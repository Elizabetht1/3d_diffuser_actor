import os
import glob
import random
from typing import List, Dict, Any, Sequence, Iterable, Tuple, Optional
from pathlib import Path
import json

import open3d
import traceback
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import einops

from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.environment import Environment
from rlbench.task_environment import TaskEnvironment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.backend.exceptions import InvalidActionError
from rlbench.demo import Demo
from pyrep.errors import IKError, ConfigurationPathError
from pyrep.const import RenderMode


# NEW COPARTICLE UTILS
from lpwm_dev.rlbench_utils.geometry import (
    action_ortho6d_to_xyzw,
    action_xyzw_to_ortho6d,
    normalize_pos,
    unnormalize_pos,
    get_gripper_loc_bounds)
from lpwm_dev.utils.util_func import create_segmentation_map
from lpwm_dev.eval.eval_particle_dreamer import plot_actions
from lpwm_dev.utils.util_func import animate_trajectories 
from transformers import T5Tokenizer, T5EncoderModel


from online_evaluation_rlbench.utils import Verify2, animate_trajectory_coparticle
from utils.mover import Mover
from datetime import datetime



ALL_RLBENCH_TASKS = [
    'basketball_in_hoop', 'beat_the_buzz', 'change_channel', 'change_clock', 'close_box',
    'close_door', 'close_drawer', 'close_fridge', 'close_grill', 'close_jar', 'close_laptop_lid',
    'close_microwave', 'hang_frame_on_hanger', 'insert_onto_square_peg', 'insert_usb_in_computer',
    'lamp_off', 'lamp_on', 'lift_numbered_block', 'light_bulb_in', 'meat_off_grill', 'meat_on_grill',
    'move_hanger', 'open_box', 'open_door', 'open_drawer', 'open_fridge', 'open_grill',
    'open_microwave', 'open_oven', 'open_window', 'open_wine_bottle', 'phone_on_base',
    'pick_and_lift', 'pick_and_lift_small', 'pick_up_cup', 'place_cups', 'place_hanger_on_rack',
    'place_shape_in_shape_sorter', 'place_wine_at_rack_location', 'play_jenga',
    'plug_charger_in_power_supply', 'press_switch', 'push_button', 'push_buttons', 'put_books_on_bookshelf',
    'put_groceries_in_cupboard', 'put_item_in_drawer', 'put_knife_on_chopping_board', 'put_money_in_safe',
    'put_rubbish_in_bin', 'put_umbrella_in_umbrella_stand', 'reach_and_drag', 'reach_target',
    'scoop_with_spatula', 'screw_nail', 'setup_checkers', 'slide_block_to_color_target',
    'slide_block_to_target', 'slide_cabinet_open_and_place_cups', 'stack_blocks', 'stack_cups',
    'stack_wine', 'straighten_rope', 'sweep_to_dustpan', 'sweep_to_dustpan_of_size', 'take_frame_off_hanger',
    'take_lid_off_saucepan', 'take_money_out_safe', 'take_plate_off_colored_dish_rack', 'take_shoes_out_of_box',
    'take_toilet_roll_off_stand', 'take_umbrella_out_of_umbrella_stand', 'take_usb_out_of_computer',
    'toilet_seat_down', 'toilet_seat_up', 'tower3', 'turn_oven_on', 'turn_tap', 'tv_on', 'unplug_charger',
    'water_plants', 'wipe_desk'
]
TASK_TO_ID = {task: i for i, task in enumerate(ALL_RLBENCH_TASKS)}


def task_file_to_task_class(task_file):
    import importlib

    name = task_file.replace(".py", "")
    class_name = "".join([w[0].upper() + w[1:] for w in name.split("_")])
    mod = importlib.import_module("rlbench.tasks.%s" % name)
    mod = importlib.reload(mod)
    task_class = getattr(mod, class_name)
    return task_class


def load_episodes() -> Dict[str, Any]:
    with open(Path(__file__).parent.parent / "data_preprocessing/episodes.json") as fid:
        return json.load(fid)



class Actioner_Coparticle:

    def __init__(
        self,
        policy=None,
        instructions=None,
        apply_cameras=("left_shoulder", "right_shoulder", "wrist"),
        action_dim=7,
        predict_trajectory=True,
        convert_6D=False,
        num_pred_steps=10,
        cond_steps=1,
        deterministic=True,
        max_length = 12,
        gripper_loc_bounds = None
    ):
        self._policy = policy
        self._instructions = instructions
        self._apply_cameras = apply_cameras
        self._action_dim = action_dim
        self._predict_trajectory = predict_trajectory

        self._actions = {}
        self._instr = None
        self._task_str = None
        self._desc = None
        self._convert_6D = convert_6D
        self._deterministic = deterministic
        
        self.num_pred_steps = num_pred_steps
        self.cond_steps = cond_steps
        self._max_length = max_length
        
        
        self._tokenizer = T5Tokenizer.from_pretrained('t5-large')
        self._t5_encoder = T5EncoderModel.from_pretrained('t5-large')
        self._t5_encoder.eval()
        

        self._policy.eval()
        
        if gripper_loc_bounds is None:
            raise ValueError("Must specify gripper location bounds")
    
    

        self.gripper_loc_bounds = torch.tensor(gripper_loc_bounds)
      
    
       
    def _embed(self, descriptions):
        desc = descriptions[np.random.randint(len(descriptions))]
        tokenized_desc = self._tokenizer(
            desc, max_length=self._max_length,
            padding='max_length', truncation=True, return_tensors='pt',
        )
        tokenized_desc = tokenized_desc['input_ids']
        with torch.no_grad():
            return self._t5_encoder(tokenized_desc).last_hidden_state.squeeze(0).float().to(self.device), desc
        
        
        
    def load_episode(self, task_str, variation, descriptions):
        self._task_str = task_str
        self._instr, desc = self._embed(descriptions)
        self._task_id = torch.tensor(TASK_TO_ID[task_str]).unsqueeze(0)
        self._actions = {}
        self._desc = desc
        return desc
        
    def load_instruction(self):
        # @TODO implement me
        pass 
    
    def _build_input_frames(
        self,
        obs_sequence: Sequence[np.ndarray],
    ) -> List[np.ndarray]:
        
        obs_sequence = list(obs_sequence)
        
        if not obs_sequence:
            raise ValueError('obs_sequence must contain at least one observation.')
    
        cond_steps = self.cond_steps
        num_pred_steps = self.num_pred_steps
        
        cond_frames = obs_sequence[-cond_steps:]
        if len(cond_frames) < cond_steps:
            raise ValueError('conditioning steps does not match conditiong data.')

        last = cond_frames[-1]
        return cond_frames + [last] * int(num_pred_steps)
    
    def _preprocess_frames(self, frames: Iterable[np.ndarray]) -> torch.Tensor:
        tensors = [self._preprocess_single_image(img) for img in frames]
        # stack: [T, n_views, C, H, W] -> unsqueeze: [1, T, n_views, H, W, C]
        frames_tensor = torch.stack(tensors, dim=0).unsqueeze(0)
        assert frames_tensor.ndim == 6  # (bs=1, T, n_views, H, W, C)
        
        frames_tensor = frames_tensor.permute(0,1,2,5,3,4)  # (bs=1, T, n_views,C, H, W,)
        
        # handle multiview reshaping
        n_views = self._policy.n_views
        
        if not frames_tensor.shape[2] == n_views:
            raise ValueError("frames do not have correct dimensionality")
        
        
        frames_tensor = frames_tensor.permute(0, 2, 1, 3, 4, 5)
        frames_tensor = frames_tensor.reshape(-1, *frames_tensor.shape[2:])  # [bs * n_views, T, ...]


    
        return frames_tensor
    
    
    def _preprocess_single_image(self, image):
        """Expect numpy array (n_views, H, W, C) in [0, 255].
        Returns tensor (n_views, C, H, W) in [0, 1] on self.device."""
        if isinstance(image, torch.Tensor):
            raise ValueError('Input image should be a numpy array')
        if image.ndim != 4:
            raise ValueError(f'Unsupported image shape: {tuple(image.shape)}')
        return torch.from_numpy(image).float().div(255.0).to(self.device)
    
    def _preprocess_actions(self,
                            actions_seq: Iterable[torch.Tensor]
                            ):
        
        
        # if not isinstance(actions_seq,np.ndarray):
        #     raise ValueError("actions seq should be np array ") 
        actions = torch.stack(actions_seq, dim=0)
        actions = actions.float().to(self.device)
        
        # convert dims as needed
        if self._convert_6D:
            actions = action_xyzw_to_ortho6d(actions)
            
        # normalize
        actions[...,:3] = normalize_pos(actions[...,:3], gripper_loc_bounds=self.gripper_loc_bounds)
        
        
        # pad as needed 
        cond_steps = self.cond_steps
        num_pred_steps = self.num_pred_steps
        target_len = cond_steps + num_pred_steps
        
       
        
        T, adim = actions.shape
        if T < target_len:
            pad_len = target_len - T
            pad = actions[-1:].repeat(pad_len, 1)
            actions = torch.cat([actions, pad], dim=0)
        return actions.unsqueeze(0)
        
    def predict(self, rgbs, pcds=None, actions=None, interpolation_length=None):
        """
        Args:
            rgbs: (bs, nhist, num_cameras, 3, H, W)
            pcds: ignored (signature compatibility with Actioner)
            actions: (bs, nhist, action_dim) tensor, or (T, action_dim) numpy array
            interpolation_length: ignored (signature compatibility with Actioner)
        Returns:
            {"action": None, "trajectory": np.ndarray}
        """
        torch.cuda.empty_cache()
        output = {"action": None, "trajectory": None, "rgb": None}

        if self._instr is None:
            raise ValueError()

        self._instr = self._instr.to(self.device)
        self._task_id = self._task_id.to(self.device)

        # perform all preprocessing
        
        rgbs = self._build_input_frames(rgbs)
        rgbs = self._preprocess_frames(rgbs)    
        actions = self._preprocess_actions(actions)
        
        with torch.no_grad():
            assert rgbs.min() >= 0 and rgbs.max() <= 1
            rec, action_rec, _, _ = self._policy.sample_from_x(
                x=rgbs,
                num_steps=self.num_pred_steps,
                cond_steps=self.cond_steps,
                deterministic=self._deterministic,
                decode=True,
                n_pred_eq_gt=False,
                return_aux_rec=True,
                actions=actions,
                lang_embed=self._instr
            )
            
            # action post processing
            # handle conversion from 6D to quarterion 
            if self._convert_6D:
                action_rec = action_ortho6d_to_xyzw(action_rec)
                
            # undo position normalization
            action_rec[...,:3] = unnormalize_pos(action_rec[...,:3], gripper_loc_bounds=self.gripper_loc_bounds)
            
            # ignore the appended reconstructions
            action_rec = action_rec[:, self.cond_steps:self.cond_steps + self.num_pred_steps] 
            trajectory = action_rec
            
            output['trajectory'] = trajectory
            
            output['rgb'] = rec[:, self.cond_steps:self.cond_steps + self.num_pred_steps] 
         
        return output

    @property
    def device(self):
        return next(self._policy.parameters()).device

class Actioner:

    def __init__(
        self,
        policy=None,
        instructions=None,
        apply_cameras=("left_shoulder", "right_shoulder", "wrist"),
        action_dim=7,
        predict_trajectory=True
    ):
        self._policy = policy
        self._instructions = instructions
        self._apply_cameras = apply_cameras
        self._action_dim = action_dim
        self._predict_trajectory = predict_trajectory

        self._actions = {}
        self._instr = None
        self._task_str = None

        self._policy.eval()

    def load_episode(self, task_str, variation,descriptions):
        self._task_str = task_str
        instructions = list(self._instructions[task_str][variation])
        self._instr = random.choice(instructions).unsqueeze(0)
        self._task_id = torch.tensor(TASK_TO_ID[task_str]).unsqueeze(0)
        self._actions = {}

    def get_action_from_demo(self, demo):
        """
        Fetch the desired state and action based on the provided demo.
            :param demo: fetch each demo and save key-point observations
            :return: a list of obs and action
        """
        key_frame = keypoint_discovery(demo)

        action_ls = []
        trajectory_ls = []
        for i in range(len(key_frame)):
            obs = demo[key_frame[i]]
            action_np = np.concatenate([obs.gripper_pose, [obs.gripper_open]])
            action = torch.from_numpy(action_np)
            action_ls.append(action.unsqueeze(0))

            trajectory_np = []
            for j in range(key_frame[i - 1] if i > 0 else 0, key_frame[i]):
                obs = demo[j]
                trajectory_np.append(np.concatenate([
                    obs.gripper_pose, [obs.gripper_open]
                ]))
            trajectory_ls.append(np.stack(trajectory_np))

        trajectory_mask_ls = [
            torch.zeros(1, key_frame[i] - (key_frame[i - 1] if i > 0 else 0)).bool()
            for i in range(len(key_frame))
        ]

        return action_ls, trajectory_ls, trajectory_mask_ls

    def predict(self, rgbs, pcds, gripper,
                interpolation_length=None):
        """
        Args:
            rgbs: (bs, num_hist, num_cameras, 3, H, W)
            pcds: (bs, num_hist, num_cameras, 3, H, W)
            gripper: (B, nhist, output_dim)
            interpolation_length: an integer

        Returns:
            {"action": torch.Tensor, "trajectory": torch.Tensor}
        """
        output = {"action": None, "trajectory": None}

        if self._instr is None:
            raise ValueError()

        self._instr = self._instr.to(rgbs.device)
        self._task_id = self._task_id.to(rgbs.device)

        # Predict trajectory
        if self._predict_trajectory:
            print('Predict Trajectory')
            fake_traj = torch.full(
                [1, interpolation_length - 1, gripper.shape[-1]], 0
            ).to(rgbs.device)
            traj_mask = torch.full(
                [1, interpolation_length - 1], False
            ).to(rgbs.device)
            output["trajectory"] = self._policy(
                fake_traj,
                traj_mask,
                rgbs[:, -1],
                pcds[:, -1],
                self._instr,
                gripper[..., :7],
                run_inference=True
            )
        else:
            print('Predict Keypose')
            pred = self._policy(
                rgbs[:, -1],
                pcds[:, -1],
                self._instr,
                gripper[:, -1, :self._action_dim],
            )
            # Hackish, assume self._policy is an instance of Act3D
            output["action"] = self._policy.prepare_action(pred)

        return output

    @property
    def device(self):
        return next(self._policy.parameters()).device


def obs_to_attn(obs, camera):
    extrinsics_44 = torch.from_numpy(
        obs.misc[f"{camera}_camera_extrinsics"]
    ).float()
    extrinsics_44 = torch.linalg.inv(extrinsics_44)
    intrinsics_33 = torch.from_numpy(
        obs.misc[f"{camera}_camera_intrinsics"]
    ).float()
    intrinsics_34 = F.pad(intrinsics_33, (0, 1, 0, 0))
    gripper_pos_3 = torch.from_numpy(obs.gripper_pose[:3]).float()
    gripper_pos_41 = F.pad(gripper_pos_3, (0, 1), value=1).unsqueeze(1)
    points_cam_41 = extrinsics_44 @ gripper_pos_41

    proj_31 = intrinsics_34 @ points_cam_41
    proj_3 = proj_31.float().squeeze(1)
    u = int((proj_3[0] / proj_3[2]).round())
    v = int((proj_3[1] / proj_3[2]).round())

    return u, v


class RLBenchEnv:

    def __init__(
        self,
        data_path,
        image_size=(128, 128),
        apply_rgb=False,
        apply_depth=False,
        apply_pc=False,
        headless=False,
        apply_cameras=("left_shoulder", "right_shoulder", "wrist", "front"),
        fine_sampling_ball_diameter=None,
        collision_checking=False,
        args=None
    ):
        

        # setup required inputs
        self.data_path = data_path
        self.apply_rgb = apply_rgb
        self.apply_depth = apply_depth
        self.apply_pc = apply_pc
        self.apply_cameras = apply_cameras
        self.fine_sampling_ball_diameter = fine_sampling_ball_diameter

        # setup RLBench environments
        self.obs_config = self.create_obs_config(
            image_size, apply_rgb, apply_depth, apply_pc, apply_cameras
        )

        self.action_mode = MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaPlanning(collision_checking=collision_checking),
            gripper_action_mode=Discrete()
        )
        self.env = Environment(
            self.action_mode, str(data_path), self.obs_config,
            headless=headless
        )
        self.image_size = image_size
        
        self.args = args

    def get_obs_action(self, obs):
        """
        Fetch the desired state and action based on the provided demo.
            :param obs: incoming obs
            :return: required observation and action list
        """

        # fetch state
        state_dict = {"rgb": [], "depth": [], "pc": []}
        for cam in self.apply_cameras:
            if self.apply_rgb:
                rgb = getattr(obs, "{}_rgb".format(cam))
                state_dict["rgb"] += [rgb]

            if self.apply_depth:
                depth = getattr(obs, "{}_depth".format(cam))
                state_dict["depth"] += [depth]

            if self.apply_pc:
                pc = getattr(obs, "{}_point_cloud".format(cam))
                state_dict["pc"] += [pc]

        # fetch action
        action = np.concatenate([obs.gripper_pose, [obs.gripper_open]])
        return state_dict, torch.from_numpy(action).float()

    def get_rgb_pcd_gripper_from_obs(self, obs):
        """
        Return rgb, pcd, and gripper from a given observation
        :param obs: an Observation from the env
        :return: rgb, pcd, gripper
        """
        state_dict, gripper = self.get_obs_action(obs)
        state = transform(state_dict, augmentation=False)
        state = einops.rearrange(
            state,
            "(m n ch) h w -> n m ch h w",
            ch=3,
            n=len(self.apply_cameras),
            m=2
        )
        rgb = state[:, 0].unsqueeze(0)  # 1, N, C, H, W
        pcd = state[:, 1].unsqueeze(0)  # 1, N, C, H, W
        gripper = gripper.unsqueeze(0)  # 1, D

        attns = torch.Tensor([])
        for cam in self.apply_cameras:
            u, v = obs_to_attn(obs, cam)
            attn = torch.zeros(1, 1, 1, self.image_size[0], self.image_size[1])
            if not (u < 0 or u > self.image_size[1] - 1 or v < 0 or v > self.image_size[0] - 1):
                attn[0, 0, 0, v, u] = 1
            attns = torch.cat([attns, attn], 1)
        rgb = torch.cat([rgb, attns], 2)

        return rgb, pcd, gripper

    def get_obs_action_from_demo(self, demo):
        """
        Fetch the desired state and action based on the provided demo.
            :param demo: fetch each demo and save key-point observations
            :param normalise_rgb: normalise rgb to (-1, 1)
            :return: a list of obs and action
        """
        key_frame = keypoint_discovery(demo)
        key_frame.insert(0, 0)
        state_ls = []
        action_ls = []
        for f in key_frame:
            state, action = self.get_obs_action(demo._observations[f])
            state = transform(state, augmentation=False)
            state_ls.append(state.unsqueeze(0))
            action_ls.append(action.unsqueeze(0))
        return state_ls, action_ls

    def get_gripper_matrix_from_action(self, action):
        action = action.cpu().numpy()
        position = action[:3]
        quaternion = action[3:7]
        rotation = open3d.geometry.get_rotation_matrix_from_quaternion(
            np.array((quaternion[3], quaternion[0], quaternion[1], quaternion[2]))
        )
        gripper_matrix = np.eye(4)
        gripper_matrix[:3, :3] = rotation
        gripper_matrix[:3, 3] = position
        return gripper_matrix

    def get_demo(self, task_name, variation, episode_index):
        """
        Fetch a demo from the saved environment.
            :param task_name: fetch task name
            :param variation: fetch variation id
            :param episode_index: fetch episode index: 0 ~ 99
            :return: desired demo
        """
        demos = self.env.get_demos(
            task_name=task_name,
            variation_number=variation,
            amount=1,
            from_episode_number=episode_index,
            random_selection=False,
        )
        return demos

    def evaluate_task_on_multiple_variations(
        self,
        task_str: str,
        max_steps: int,
        num_variations: int,  # -1 means all variations
        num_demos: int,
        actioner: Actioner,
        max_tries: int = 1,
        verbose: bool = False,
        dense_interpolation=False,
        interpolation_length=100,
        num_history=1,
        verify=True
    ):
        self.env.launch()
        task_type = task_file_to_task_class(task_str)
        task = self.env.get_task(task_type)
        task_variations = task.variation_count()

        if num_variations > 0:
            task_variations = np.minimum(num_variations, task_variations)
            task_variations = range(task_variations)
        else:
            task_variations = glob.glob(os.path.join(self.data_path, task_str, "variation*"))
            task_variations = [int(n.split('/')[-1].replace('variation', '')) for n in task_variations]

        var_success_rates = {}
        var_num_valid_demos = {}

        log_run = datetime.now().strftime("%m:%d:%Y_%I:%M_%p")
        for variation in task_variations:
            task.set_variation(variation)
            success_rate, valid, num_valid_demos = (
                self._evaluate_task_on_one_variation(
                    task_str=task_str,
                    task=task,
                    max_steps=max_steps,
                    variation=variation,
                    num_demos=num_demos // len(task_variations) + 1,
                    actioner=actioner,
                    max_tries=max_tries,
                    verbose=verbose,
                    dense_interpolation=dense_interpolation,
                    interpolation_length=interpolation_length,
                    num_history=num_history,
                    log_run = log_run,
                    verify=verify
                )
            )
            if valid:
                var_success_rates[variation] = success_rate
                var_num_valid_demos[variation] = num_valid_demos

        self.env.shutdown()

        var_success_rates["mean"] = (
            sum(var_success_rates.values()) /
            sum(var_num_valid_demos.values())
        )

        return var_success_rates
    
    def _get_gt_data(
        self,
        demo,
        camera_names: Tuple[str, ...],
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
            """Extract per-timestep GT actions and RGB frames from every demo step."""
            gt_actions: List[np.ndarray] = []
            frames_0: List[np.ndarray] = []
            frames_1: List[np.ndarray] = []
            for obs in demo:
                gt_actions.append(np.concatenate([obs.gripper_pose, [obs.gripper_open]]))
                frames_0.append(getattr(obs, f"{camera_names[0]}_rgb"))
                if len(camera_names) >= 2:
                    frames_1.append(getattr(obs, f"{camera_names[1]}_rgb"))
            cam1 = np.stack(frames_1) if frames_1 else None
            return np.stack(gt_actions), np.stack(frames_0), cam1


    @torch.no_grad()
    def _evaluate_task_on_one_variation(
        self,
        task_str: str,
        task: TaskEnvironment,
        max_steps: int,
        variation: int,
        num_demos: int,
        actioner: Actioner,
        max_tries: int = 1,
        verbose: bool = False,
        dense_interpolation=False,
        interpolation_length=50,
        num_history=0,
        coparticle = True,
        verify = True,
        log_run = None
    ):
        device = actioner.device

        success_rate = 0
        num_valid_demos = 0
        total_reward = 0

        log_run = log_run if log_run is not None else datetime.now().strftime("%m:%d:%Y_%I:%M_%p") # directory to store logs at 
        os.makedirs(f"eval_logs/{log_run}",exist_ok=True)
        
        for demo_id in range(num_demos):
            if verbose:
                print()
                print(f"Starting demo {demo_id}")

            try:
                demo = self.get_demo(task_str, variation, episode_index=demo_id)[0]
                num_valid_demos += 1
            except:
                continue
            
            # verify 
            if verify: 
                verifier = Verify2(
                    actioner=actioner,
                    demo=demo,
                    env=self,
                    task=task,
                    task_str=task_str,
                    max_steps=max_steps,
                    max_tries=max_tries,
                    verbose=verbose,
                    device=actioner.device,
                    logdir=f"eval_logs/{log_run}"
                )
                verifier.test_1_replay_demo()
                verifier.test_2_image_preprocessing()
                verifier.test_3_action_preprocessing()
                verifier.test_4_quant_conversion()
                verifier.test_5_replay_open_loop(variation=variation,demo_id=demo_id) # @TODO some sort of memory leakage – gets an OOM error after a couple of iterations 
                verifier.test_6_replay_recon(variation=variation,demo_id=demo_id)
                verifier.test_7_replay_recon_with_ctx(variation=variation,demo_id=demo_id)
                del verifier
                torch.cuda.empty_cache()
            else:
                print("[WARNING] skipping verification ... ")
            
            

            rgbs = torch.Tensor([]).to(device)
            pcds = torch.Tensor([]).to(device)
            grippers = torch.Tensor([]).to(device)

            # descriptions, obs = task.reset()
            descriptions, obs = task.reset_to_demo(demo)

            desc = actioner.load_episode(task_str, variation, descriptions)
            
            move = Mover(task, max_tries=max_tries)
            reward = 0.0
            max_reward = 0.0
            
            use_second_cam = len(self.apply_cameras) > 1
            
            actions_history : List[np.ndarray] = []
            frames_cam0 : List[np.ndarray] = []
            frames_cam1 : Optional[List[np.ndarray]] = [] if use_second_cam else None
            imagination_frames = []
            
            

            for step_id in range(max_steps):

                # Fetch the current observation, and predict one action
                if coparticle: 
                    state_dict, gripper = self.get_obs_action(obs)
                    rgbs_input = [np.stack(state_dict['rgb'],axis=0)]  # (num_cameras, H, W, C)   
                    pcds_input = None
                    gripper_input = [gripper]
                
                else: 
                    rgb, pcd, gripper = self.get_rgb_pcd_gripper_from_obs(obs)
                    rgb = rgb.to(device)
                    pcd = pcd.to(device)
                    gripper = gripper.to(device)

                    rgbs = torch.cat([rgbs, rgb.unsqueeze(1)], dim=1)
                    pcds = torch.cat([pcds, pcd.unsqueeze(1)], dim=1)
                    grippers = torch.cat([grippers, gripper.unsqueeze(1)], dim=1)

                    # Prepare proprioception history
                    if num_history < 1:
                        rgbs_input = rgbs[:, -1:][:, :, :, :3]
                        pcds_input = pcds[:, -1:]
                        gripper_input = grippers[:, -1]
                    else:
                        rgbs_input = rgbs[:, -num_history:][:, :, :, :3]
                        pcds_input = pcds[:, -num_history:]
                        gripper_input = grippers[:, -num_history:]
                        npad = num_history - gripper_input.shape[1]
                        if npad > 0:
                            rgbs_input = F.pad(
                                rgbs_input, (0, 0, 0, 0, 0, 0, npad, 0), mode='replicate'
                            )
                            pcds_input = F.pad(
                                pcds_input, (0, 0, 0, 0, 0, 0, npad, 0), mode='replicate'
                            )
                            gripper_input = F.pad(
                                gripper_input, (0, 0, npad, 0), mode='replicate'
                            )

                
                output = actioner.predict(
                    rgbs_input,
                    pcds_input,
                    gripper_input,
                    interpolation_length=interpolation_length
                )

                if verbose:
                    print(f"Step {step_id}")

                terminate = True

                # Update the observation based on the predicted action
                try:
                    # Execute entire predicted trajectory step by step
                    if output.get("trajectory", None) is not None:
                        trajectory = output["trajectory"][-1].cpu().numpy()
                        trajectory[:, -1] = trajectory[:, -1].round()
                        
                        rgb_rec = output["rgb"].cpu().numpy()
                        imagination_frames.append(rgb_rec)

                        # execute
                        for action in tqdm(trajectory):
                            #try:
                            #    collision_checking = self._collision_checking(task_str, step_id)
                            #    obs, reward, terminate, _ = move(action_np, collision_checking=collision_checking)
                            #except:
                            #    terminate = True
                            #    pass
                            collision_checking = self._collision_checking(task_str, step_id)
                            obs, reward, terminate, _ = move(action, collision_checking=collision_checking)
                            
                            actions_history.append(action)
                            
                            state_dict, gripper = self.get_obs_action(obs)
                            frames_cam0.append(state_dict['rgb'][0])
                            if use_second_cam:
                                frames_cam1.append(state_dict['rgb'][1])


                    # Or plan to reach next predicted keypoint
                    else:
                        print("Plan with RRT")
                        action = output["action"]
                        action[..., -1] = torch.round(action[..., -1])
                        action = action[-1].detach().cpu().numpy()

                        collision_checking = self._collision_checking(task_str, step_id)
                        obs, reward, terminate, _ = move(action, collision_checking=collision_checking)

                    max_reward = max(max_reward, reward)

                    if reward == 1:
                        success_rate += 1
                        break

                    if terminate:
                        print("The episode has terminated!")

                except (IKError, ConfigurationPathError, InvalidActionError) as e:
                    print(task_str, demo, step_id, success_rate, e)
                    reward = 0
                    #break

            total_reward += max_reward
            if reward == 0:
                step_id += 1

            print(
                task_str,
                "Variation",
                variation,
                "Demo",
                demo_id,
                "Reward",
                f"{reward:.2f}",
                "max_reward",
                f"{max_reward:.2f}",
                f"SR: {success_rate}/{demo_id+1}",
                f"SR: {total_reward:.2f}/{demo_id+1}",
                "# valid demos", num_valid_demos,
            )
            
            
            gt_actions, gt_cam0, gt_cam1 = self._get_gt_data(demo,actioner._apply_cameras) # @TODO don't make this verifier logic dependent

            T1,adim = gt_actions.shape
            T2 = len(actions_history)
            T= min(T1,T2)
            plot_actions(
                torch.Tensor(actions_history[:T]),
                torch.from_numpy(gt_actions[:T]),
                T,
                ndim=adim,
                id=f"{task_str}_v{variation}_d{demo_id}",
                root=f"eval_logs/{log_run}"
            )
            
            # comapre gt vs simulation rollouts 
            frames_cam0 = np.stack(frames_cam0)[:T] / 255.0
            
            gt_cam0 = gt_cam0[:T] / 255.0
            
            imagination_frames = np.concatenate(imagination_frames,axis=1)[:,:T]
            imagination_cam0 = imagination_frames[0].transpose(0,2,3,1)
            
            if use_second_cam:
                frames_cam1 = np.stack(frames_cam1)[:T] / 255.0
                gt_cam1 = gt_cam1[:T] / 255.0
                imagination_cam1 = imagination_frames[1].transpose(0,2,3,1)
            else: 
                frames_cam1 = gt_cam1 = imagination_cam1 = None
            
            
            animate_trajectories(
                orig_trajectory=gt_cam0,
                pred_trajectory=frames_cam0,
                pred_trajectory_2 = imagination_cam0,
                path=f'eval_logs/{log_run}/{task_str}_v{variation}_d{demo_id}_sim_rollout.gif',
                duration=10, # @TODO don't hardcode
                rec_to_pred_t=actioner.cond_steps,
                t1="-Sim-Rollout",
                t2="-Imagination",
                title=f"GT vs Predicted - {desc}",
                orig_trajectory2=gt_cam1, 
                pred_trajectory_12=frames_cam1,
                pred_trajectory_22= imagination_cam1
            )

        # Compensate for failed demos
        if num_valid_demos == 0:
            assert success_rate == 0
            valid = False
        else:
            valid = True

        return success_rate, valid, num_valid_demos

    def _collision_checking(self, task_str, step_id):
        """Collision checking for planner."""
        # collision_checking = True
        collision_checking = False
        # if task_str == 'close_door':
        #     collision_checking = True
        # if task_str == 'open_fridge' and step_id == 0:
        #     collision_checking = True
        # if task_str == 'open_oven' and step_id == 3:
        #     collision_checking = True
        # if task_str == 'hang_frame_on_hanger' and step_id == 0:
        #     collision_checking = True
        # if task_str == 'take_frame_off_hanger' and step_id == 0:
        #     for i in range(300):
        #         self.env._scene.step()
        #     collision_checking = True
        # if task_str == 'put_books_on_bookshelf' and step_id == 0:
        #     collision_checking = True
        # if task_str == 'slide_cabinet_open_and_place_cups' and step_id == 0:
        #     collision_checking = True
        return collision_checking

    def verify_demos(
        self,
        task_str: str,
        variation: int,
        num_demos: int,
        max_tries: int = 1,
        verbose: bool = False,
    ):
        if verbose:
            print()
            print(f"{task_str}, variation {variation}, {num_demos} demos")

        self.env.launch()
        task_type = task_file_to_task_class(task_str)
        task = self.env.get_task(task_type)
        task.set_variation(variation)  # type: ignore

        success_rate = 0.0
        invalid_demos = 0

        for demo_id in range(num_demos):
            if verbose:
                print(f"Starting demo {demo_id}")

            try:
                demo = self.get_demo(task_str, variation, episode_index=demo_id)[0]
            except:
                print(f"Invalid demo {demo_id} for {task_str} variation {variation}")
                print()
                traceback.print_exc()
                invalid_demos += 1

            task.reset_to_demo(demo)

            gt_keyframe_actions = []
            for f in keypoint_discovery(demo):
                obs = demo[f]
                action = np.concatenate([obs.gripper_pose, [obs.gripper_open]])
                gt_keyframe_actions.append(action)

            move = Mover(task, max_tries=max_tries)

            for step_id, action in enumerate(gt_keyframe_actions):
                if verbose:
                    print(f"Step {step_id}")

                try:
                    obs, reward, terminate, step_images = move(action)
                    if reward == 1:
                        success_rate += 1 / num_demos
                        break
                    if terminate and verbose:
                        print("The episode has terminated!")

                except (IKError, ConfigurationPathError, InvalidActionError) as e:
                    print(task_type, demo, success_rate, e)
                    reward = 0
                    break

            if verbose:
                print(f"Finished demo {demo_id}, SR: {success_rate}")

        # Compensate for failed demos
        if (num_demos - invalid_demos) == 0:
            success_rate = 0.0
            valid = False
        else:
            success_rate = success_rate * num_demos / (num_demos - invalid_demos)
            valid = True

        self.env.shutdown()
        return success_rate, valid, invalid_demos

    def create_obs_config(
        self, image_size, apply_rgb, apply_depth, apply_pc, apply_cameras, **kwargs
    ):
        """
        Set up observation config for RLBench environment.
            :param image_size: Image size.
            :param apply_rgb: Applying RGB as inputs.
            :param apply_depth: Applying Depth as inputs.
            :param apply_pc: Applying Point Cloud as inputs.
            :param apply_cameras: Desired cameras.
            :return: observation config
        """
        unused_cams = CameraConfig()
        unused_cams.set_all(False)
        used_cams = CameraConfig(
            rgb=apply_rgb,
            point_cloud=apply_pc,
            depth=apply_depth,
            mask=False,
            image_size=image_size,
            render_mode=RenderMode.OPENGL,
            **kwargs,
        )

        camera_names = apply_cameras
        kwargs = {}
        for n in camera_names:
            kwargs[n] = used_cams

        obs_config = ObservationConfig(
            front_camera=kwargs.get("front", unused_cams),
            left_shoulder_camera=kwargs.get("left_shoulder", unused_cams),
            right_shoulder_camera=kwargs.get("right_shoulder", unused_cams),
            wrist_camera=kwargs.get("wrist", unused_cams),
            overhead_camera=kwargs.get("overhead", unused_cams),
            joint_forces=False,
            joint_positions=False,
            joint_velocities=True,
            task_low_dim_state=False,
            gripper_touch_forces=False,
            gripper_pose=True,
            gripper_open=True,
            gripper_matrix=True,
            gripper_joint_positions=True,
        )

        return obs_config


# Identify way-point in each RLBench Demo
def _is_stopped(demo, i, obs, stopped_buffer, delta):
    next_is_not_final = i == (len(demo) - 2)
    # gripper_state_no_change = i < (len(demo) - 2) and (
    #     obs.gripper_open == demo[i + 1].gripper_open
    #     and obs.gripper_open == demo[i - 1].gripper_open
    #     and demo[i - 2].gripper_open == demo[i - 1].gripper_open
    # )
    gripper_state_no_change = i < (len(demo) - 2) and (
        obs.gripper_open == demo[i + 1].gripper_open
        and obs.gripper_open == demo[max(0, i - 1)].gripper_open
        and demo[max(0, i - 2)].gripper_open == demo[max(0, i - 1)].gripper_open
    )
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    stopped = (
        stopped_buffer <= 0
        and small_delta
        and (not next_is_not_final)
        and gripper_state_no_change
    )
    return stopped


def keypoint_discovery(demo: Demo, stopping_delta=0.1) -> List[int]:
    episode_keypoints = []
    prev_gripper_open = demo[0].gripper_open
    stopped_buffer = 0

    for i, obs in enumerate(demo):
        stopped = _is_stopped(demo, i, obs, stopped_buffer, stopping_delta)
        stopped_buffer = 4 if stopped else stopped_buffer - 1
        # If change in gripper, or end of episode.
        last = i == (len(demo) - 1)
        if i != 0 and (obs.gripper_open != prev_gripper_open or last or stopped):
            episode_keypoints.append(i)
        prev_gripper_open = obs.gripper_open

    if (
        len(episode_keypoints) > 1
        and (episode_keypoints[-1] - 1) == episode_keypoints[-2]
    ):
        episode_keypoints.pop(-2)

    return episode_keypoints


def transform(obs_dict, scale_size=(0.75, 1.25), augmentation=False):
    apply_depth = len(obs_dict.get("depth", [])) > 0
    apply_pc = len(obs_dict["pc"]) > 0
    num_cams = len(obs_dict["rgb"])

    obs_rgb = []
    obs_depth = []
    obs_pc = []
    for i in range(num_cams):
        rgb = torch.tensor(obs_dict["rgb"][i]).float().permute(2, 0, 1)
        depth = (
            torch.tensor(obs_dict["depth"][i]).float().permute(2, 0, 1)
            if apply_depth
            else None
        )
        pc = (
            torch.tensor(obs_dict["pc"][i]).float().permute(2, 0, 1) if apply_pc else None
        )

        if augmentation:
            raise NotImplementedError()  # Deprecated

        # normalise to [0, 1]
        rgb = rgb / 255.0

        obs_rgb += [rgb.float()]
        if depth is not None:
            obs_depth += [depth.float()]
        if pc is not None:
            obs_pc += [pc.float()]
    obs = obs_rgb + obs_depth + obs_pc
    return torch.cat(obs, dim=0)
