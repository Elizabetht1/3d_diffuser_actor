"""Online evaluation for an RGB-based diffusion policy on RLBench.

Replaces the 3D Diffuser Actor keypose pipeline with:
  - RGB-only observation construction (no depth / point clouds)
  - Continuous timestep-by-timestep execution with chunked model queries
  - Black-box model.sample_from_x() inference
"""

import glob
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import tap
from pyrep.errors import ConfigurationPathError, IKError
from rlbench.backend.exceptions import InvalidActionError
from rlbench.observation_config import ObservationConfig
from pyrep.const import RenderMode
from datetime import datetime

# REPLACED: DiffuserActor and Act3D imports removed.
# Those models conditioned on 3D scene tokens built from point clouds.
# The new RGB diffusion policy is loaded via load_model() instead.

# REPLACED: load_instructions removed — language embeddings are now
# pre-embedded tensors passed as language_goal (see TODO).
# REPLACED: get_gripper_loc_bounds removed — gripper workspace bounds
# are not needed without 3D scene coordinate normalisation.
from utils.common_utils import round_floats

from utils.utils_with_rlbench import (
    Mover,
    RLBenchEnv,
    load_episodes,
    task_file_to_task_class,
)

# REPLACED: Actioner removed.
# Actioner.predict(rgbs, pcds, gripper) called DiffuserActor/Act3D with
# point-cloud tensors and 3D trajectory denoising. Inference is now
# handled by query_model() via model.sample_from_x().

# @TODO add convert to 6D 
# @TODO add support for existing datasets 
# @TODO figure out language conditioning crash

from lpwm_dev.model_factory import build_model  # type: ignore  # black-box; not implemented here
from lpwm_dev.rlbench_utils.action_reconstructions import plot_actions
from lpwm_dev.utils.util_func import animate_trajectories  # type: ignore
from lpwm_dev.rlbench_utils.geometry import action_xyzw_to_ortho6d, action_ortho6d_to_xyzw
from transformers import T5Tokenizer, T5EncoderModel
from lpwm_dev.eval.eval_model import animate_trajectory_ddlp

def animate_trajectory_coparticle(model, x_horizon, actions_horizon, lang_embed, lang_str, epoch, device=torch.device('cpu'), fig_dir='./', timestep_horizon=3,
                            num_trajetories=5, accelerator=None, train=False, prefix='', cond_steps=None,
                            deterministic=True, det_and_stoch=True, use_all_ctx=True,animation_fps = 10):
    # load data
    # ds = config['ds']
    # ch = config['ch']  # image channels
    # image_size = config['image_size']
    # n_views = config.get('n_views', 1)
    # root = config['root']  # dataset root
    # duration = config['animation_fps']
    # action_condition = config.get('action_condition', False)
    # language_condition = config.get('language_condition', False)
    # img_goal_condition = config.get('image_goal_condition', False)
    # views = config.get('views',['front'])

    # # mode = 'train' if train else "valid"
    # mode = 'test'
    # # dataset = get_video_dataset(ds, root, seq_len=timestep_horizon, mode=mode, image_size=image_size,tasks=config.get('tasks',None),views=views)
    # dataset = RLBenchDataset2(root=root, 
    #                 mode=mode, 
    #                 ep_len=121,
    #                 sample_length=timestep_horizon, 
    #                 image_size=image_size,
    #                 tiny=True,
    #                 tasks=config.get('tasks',None),
    #                 multiview=len(views)>1,
    #                 views=views,
    #                 actions=config.get('actions',None),
    #                 convert_6D=config.get('convert_6D',False),
    #                 language_condition=language_condition)
    # batch_size = max(1, num_trajetories)
    # dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4, drop_last=False)
    # batch = next(iter(dataloader))
    # model_timestep_horizon = model.timestep_horizon
    # cond_steps = model_timestep_horizon if cond_steps is None else cond_steps
    # model.eval()
    # x_horizon = batch[0][:, :timestep_horizon].to(device)
    # actions_horizon = None if not action_condition else batch[1][:, :timestep_horizon].to(device)
    
    # # testing 
    # x_horizon_new = torch.zeros_like(x_horizon)
    # x_horizon_new[:,:cond_steps] = x_horizon[:,:cond_steps]
    
    # actions_horizon_new = torch.zeros_like(actions_horizon)
    # actions_horizon_new[:,:cond_steps] = actions_horizon[:,:cond_steps]

    
    
    # lang_str = None if not language_condition else batch[4]
    # lang_embed = None if not language_condition else batch[5].to(device)
    # x_goal = None if not img_goal_condition else batch[2].to(device)
    
    
    
    # if n_views > 1:
    #     # expect: [bs, T, n_views, ...]
    #     x_horizon = x_horizon.permute(0, 2, 1, 3, 4, 5)
    #     x_horizon = x_horizon.reshape(-1, *x_horizon.shape[2:])  # [bs * n_views, T, ...]
    #     x_horizon_new= x_horizon_new.permute(0, 2, 1, 3, 4, 5)
    #     x_horizon_new = x_horizon_new.reshape(-1, *x_horizon_new.shape[2:])  # [bs * n_views, T, ...]
        
    #     if x_goal is not None:
    #         x_goal = x_goal.reshape(-1, *x_goal.shape[2:])  #3 [bs * n_views, ...]
    #     if actions_horizon is not None:
    #         actions_horizon = actions_horizon.permute(0, 2, 1, 3)
    #         actions_horizon = actions_horizon.reshape(-1, *actions_horizon.shape[2:])
            
    #         actions_horizon_new = actions_horizon_new.permute(0, 2, 1, 3)
    #         actions_horizon_new = actions_horizon_new.reshape(-1, *actions_horizon_new.shape[2:])
            
    # forward pass
    model_timestep_horizon = model.timestep_horizon
    cond_steps = model_timestep_horizon if cond_steps is None else cond_steps
    n_views = model.n_views
    action_rec_1 = None
    action_rec_2 = None
    duration = animation_fps
    
    # push to device 
    x_horizon = x_horizon[:, :timestep_horizon].float().to(device)
    actions_horizon = actions_horizon[:, :timestep_horizon].float().to(device)
    
    
    with torch.no_grad():
        if det_and_stoch:
            preds_1,action_rec_1,_,_= model.sample_from_x(x_horizon, num_steps=timestep_horizon - cond_steps, deterministic=True,
                                          cond_steps=cond_steps, use_all_ctx=use_all_ctx, actions=actions_horizon,
                                          lang_embed=lang_embed,return_aux_rec=True)
            preds_2, action_rec_2, reward_rec, continue_rec = model.sample_from_x(x_horizon, num_steps=timestep_horizon - cond_steps, deterministic=False,
                                          cond_steps=cond_steps, actions=actions_horizon, lang_embed=lang_embed,
                                        return_aux_rec=True)
            
            # print(f"\n [INFO] median_actions={action_rec.mean(dim=1)} | max={action_rec.max(dim=1)} | min={action_rec.min(dim=1)} \n")
        else:
            preds_1 = model.sample_from_x(x_horizon, num_steps=timestep_horizon - cond_steps,
                                          deterministic=deterministic,
                                          cond_steps=cond_steps, actions=actions_horizon, lang_embed=lang_embed,
                                          )
            preds_2 = None
        # preds: [bs, timestep_horizon, 3, im_size, im_size]
   
    for i in range(num_trajetories):
        if n_views > 1:
            x_horizon = x_horizon.reshape(-1, n_views, *x_horizon.shape[1:])
            x_preds_1 = preds_1.reshape(-1, n_views, *preds_1.shape[1:])

            gt_traj = x_horizon[i, 0].permute(0, 2, 3, 1).data.cpu().numpy()
            pred_traj = x_preds_1[i, 0].permute(0, 2, 3, 1).data.cpu().numpy()

            gt_traj_12 = x_horizon[i, 1].permute(0, 2, 3, 1).data.cpu().numpy()
            pred_traj_12 = x_preds_1[i, 1].permute(0, 2, 3, 1).data.cpu().numpy()
        else:
            gt_traj = x_horizon[i].permute(0, 2, 3, 1).data.cpu().numpy()
            pred_traj = preds_1[i].permute(0, 2, 3, 1).data.cpu().numpy()

            gt_traj_12 = pred_traj_12 = None
        lang_str_i = lang_str[i]

        x_goal_i = x_goal_i2 = None
        if det_and_stoch:
            if n_views > 1:
                x_preds_2 = preds_2.reshape(-1, n_views, *preds_2.shape[1:])
                pred_traj_2 = x_preds_2[i, 0].permute(0, 2, 3, 1).data.cpu().numpy()
                pred_traj_22 = x_preds_2[i, 1].permute(0, 2, 3, 1).data.cpu().numpy()
            else:
                pred_traj_2 = preds_2[i].permute(0, 2, 3, 1).data.cpu().numpy()
                pred_traj_22 = None
        else:
            pred_traj_2 = pred_traj_22 = None
            
        # plot visual reconstruction quality
        if accelerator is not None:
            if accelerator.is_main_process:
                animate_trajectories(gt_traj, pred_traj, pred_traj_2,
                                     path=os.path.join(fig_dir, f'{prefix}e{epoch}_traj_anim_{i}.gif'),
                                     duration=duration, 
                                     rec_to_pred_t=cond_steps, 
                                     t1='-D', t2='-S', title=lang_str_i,
                                     goal_img=x_goal_i,
                                     orig_trajectory2=gt_traj_12, pred_trajectory_12=pred_traj_12,
                                     pred_trajectory_22=pred_traj_22, goal_img2=x_goal_i2)
        else:
            animate_trajectories(gt_traj, pred_traj, pred_traj_2,
                                 path=os.path.join(fig_dir, f'{prefix}e{epoch}_traj_anim_{i}.gif'),
                                 duration=duration, rec_to_pred_t=cond_steps, t1='-D', t2='-S', title=lang_str_i,
                                 goal_img=x_goal_i, orig_trajectory2=gt_traj_12, pred_trajectory_12=pred_traj_12,
                                 pred_trajectory_22=pred_traj_22, goal_img2=x_goal_i2)
            
        # plot action reconstruction quality
        assert action_rec_2 is not None
        assert action_rec_1 is not None
        assert actions_horizon is not None
        plot_actions(
            gen_acts = action_rec_1.squeeze(),
            gt_acts= actions_horizon.squeeze(),
            timestep_horizon = timestep_horizon,
            ndim = 10,
            root = fig_dir,
            id=f"determ_{epoch}"
        )
        
        plot_actions(
            gen_acts = action_rec_2.squeeze(),
            gt_acts= actions_horizon.squeeze(),
            timestep_horizon = timestep_horizon,
            ndim = 10,
            root = fig_dir,
            id=f"stoch_{epoch}"
        )
    return preds_2, action_rec_2
     
     

def load_model(config_fp, weights_fp):
    model = build_model(config_fp)
    model.load_state_dict(torch.load(weights_fp, map_location=torch.device('cpu')))
    return model

class Arguments(tap.Tap):
    """Command-line arguments for RGB diffusion policy evaluation."""

    # Required
    config_path: Path
    weights_path: Path
    

    # RLBench simulation
    data_dir: Path = Path(__file__).parent / "demos"
    tasks: Optional[Tuple[str, ...]] = None
    variations: Tuple[int, ...] = (1,)
    num_episodes: int = 1
    max_steps: int = 100
    max_tries: int = 10
    cameras: Tuple[str, ...] = ("left_shoulder", "right_shoulder", "wrist")
    headless: int = 1
    collision_checking: int = 0
    image_size: str = "128,128"

    # Model inference
    num_steps: int = 10   # diffusion denoising steps per query
    cond_steps: int = 1    # frames of observation history fed to model
    chunk_size: int = 8    # actions per chunk before re-querying the model
    action_dim: int = 8    # dimensionality of each action vector
    gif_fps: float = 10.0   # frame rate for saved GIF animations
    convert_to_6D: int = 0

    # Outputs
    seed: int = 2
    device: str = "cuda"
    output_file: Path = Path(__file__).parent / "eval.json"
    output_dir: Path = Path(__file__).parent / "eval_outputs"
    verbose: int = 0
    
    


def build_rlbench_env(args: Arguments) -> RLBenchEnv:
    """Construct an RLBenchEnv configured for RGB-only observation.

    # REPLACED: apply_pc=True and apply_depth=True removed.
    # 3D Diffuser Actor required depth images to reconstruct point clouds
    # for 3D scene featurisation. The RGB diffusion policy only needs
    # RGB frames, so both are disabled here.
    """
    image_size = [int(x) for x in args.image_size.split(",")]
    
    # obs_config = ObservationConfig()
    # obs_config.set_all(True)
    
    # obs_config.right_shoulder_camera.render_mode = RenderMode.OPENGL
    # obs_config.left_shoulder_camera.render_mode = RenderMode.OPENGL
    # obs_config.overhead_camera.render_mode = RenderMode.OPENGL
    # obs_config.wrist_camera.render_mode = RenderMode.OPENGL
    # obs_config.front_camera.render_mode = RenderMode.OPENGL
 
 
    return RLBenchEnv(
        data_path=args.data_dir,
        image_size=image_size,
        apply_rgb=True,
        apply_depth=False,  # REPLACED: depth removed — not needed for RGB-only policy
        apply_pc=False,     # REPLACED: point cloud removed — not needed for RGB-only policy
        headless=bool(args.headless),
        apply_cameras=args.cameras,
        collision_checking=bool(args.collision_checking),
    )


def extract_rgb_tensor(
    obs,
    camera_names: Tuple[str, ...],
    device: torch.device,
) -> torch.Tensor:
    """Extract a normalised RGB tensor from one RLBench observation.

    Args:
        obs: RLBench Observation object.
        camera_names: Ordered camera names to extract.
        device: Target device for the output tensor.

    Returns:
        Tensor of shape (num_views, 3, H, W) with values in [0, 1].
    """
    frames: List[torch.Tensor] = []
    for cam in camera_names:
        rgb_np: np.ndarray = getattr(obs, f"{cam}_rgb")  # (H, W, 3) uint8
        rgb_np = _normalize_01(rgb_np) # normalize
        rgb = torch.from_numpy(rgb_np).float().permute(2, 0, 1) 
        frames.append(rgb)
    return torch.stack(frames).to(device)  # (num_views, 3, H, W)


def init_obs_buffer(first_frame: torch.Tensor, cond_steps: int) -> torch.Tensor:
    """Initialise the observation sliding window by repeating the first frame.

    # TODO: Define observation buffer initialization strategy
    # (e.g. repeat first obs, zeros). Current implementation repeats the
    # first frame across all cond_steps slots.

    Args:
        first_frame: (num_views, C, H, W)
        cond_steps: number of history frames the model expects.

    Returns:
        Tensor of shape (num_views, cond_steps, C, H, W).
    """
    return first_frame.unsqueeze(1).expand(-1, cond_steps, -1, -1, -1).clone()


def update_obs_buffer(
    buffer: torch.Tensor,
    new_frame: torch.Tensor,
) -> torch.Tensor:
    """Slide the observation buffer left and append the new frame at the end.

    Args:
        buffer: (num_views, cond_steps, C, H, W)
        new_frame: (num_views, C, H, W)

    Returns:
        Updated buffer of shape (num_views, cond_steps, C, H, W).
    """
    return torch.cat([buffer[:, 1:], new_frame.unsqueeze(1)], dim=1)


def init_action_buffer(
    cond_steps: int, action_dim: int, device: torch.device
) -> torch.Tensor:
    """Initialise the action history buffer with zeros.

    Args:
        cond_steps: number of history steps the model expects.
        action_dim: dimensionality of each action vector.

    Returns:
        Zero tensor of shape (cond_steps, action_dim).
    """
    return torch.ones(cond_steps, action_dim, device=device)


def update_action_buffer(
    buffer: torch.Tensor,
    new_action: torch.Tensor,
) -> torch.Tensor:
    """Slide the action buffer left and append the latest executed action.

    Args:
        buffer: (cond_steps, action_dim)
        new_action: (action_dim,)

    Returns:
        Updated buffer of shape (cond_steps, action_dim).
    """
    return torch.cat([buffer[1:], new_action.unsqueeze(0)], dim=0)


@torch.no_grad()
def query_model(
    model,
    obs_buffer: torch.Tensor,
    action_buffer: torch.Tensor,
    goal_tensor: Optional[torch.Tensor],
    language_goal: Optional[torch.Tensor],
    num_steps: int,
    cond_steps: int,
    convert_to_6D: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Query the RGB diffusion model for the next action chunk.

    # REPLACED: DiffuserActor / Act3D forward pass removed.
    # Old call: policy(fake_traj, traj_mask, rgbs[:, -1], pcds[:, -1], instr, gripper)
    # That consumed point-cloud tensors (pcds) and performed 3D trajectory
    # denoising. New call: model.sample_from_x() with RGB observation buffer only.

    Args:
        obs_buffer: (num_views, cond_steps, C, H, W) normalised RGB history.
        action_buffer: (cond_steps, action_dim) action history context.
        goal_tensor: optional (C, H, W) RGB goal frame.
        language_goal: optional (max_len, embed_dim) pre-embedded language.
        num_steps: diffusion denoising steps per call.
        cond_steps: context window length expected by the model.

    Returns:
        rec: imagined future RGB frames (shape TBD — see TODO).
        action_chunk: predicted actions of shape (1, chunk_size, action_dim).
    """
    # TODO: Confirm rec shape and adjust indexing/transposing accordingly.
    b, t, c, h, w = obs_buffer.shape
    
    assert h == model.image_size and w == model.image_size, "inconsistent image sizes, please specifcy image size via command line"
    
    if goal_tensor is None and model.img_goal_condition:
        print("[WARNING] introducing dummy image conditioning data.\n")
        goal_tensor = torch.zeros(b,c,h,w).to(obs_buffer.device)
        
       
    if language_goal is None and model.language_condition:
        print("[WARNING] introducing dummy language conditioning data.\n")
        language_goal = torch.ones(1,model.language_max_len, model.language_embed_dim).to(obs_buffer.device)
       
    
    # unsqueeze actions to match dimensions expected
    action_buffer = action_buffer.unsqueeze(0) # [B,cond_steps, action_dim]
    if convert_to_6D:
        action_buffer = action_xyzw_to_ortho6d(action_buffer)
    
    
    assert model.n_views * action_buffer.shape[0] == obs_buffer.shape[0], "actions and observations have same batch size"
    assert action_buffer.shape[1] == obs_buffer.shape[1], "actions and observations have same time steps"
    assert not action_buffer.isnan().any()
    assert not obs_buffer.isnan().any()
    assert language_goal is not None
    language_goal = language_goal.unsqueeze(0)

    # rec, action_chunk, _, _ = model.sample_from_x_eval(
    #     x=obs_buffer,
    #     num_steps=num_steps,
    #     # cond_steps=cond_steps,
    #     deterministic=False,
    #     x_goal=goal_tensor,
    #     decode=True,
    #     return_context_posterior=True,
    #     return_aux_rec=True,
    #     actions=action_buffer,
    #     lang_embed=language_goal,
    #     # n_pred_eq_gt = False,
    #     pred_cond_steps=1
    # )
    rec, action_chunk, _, _ = model.sample_from_x(
        x=obs_buffer,
        num_steps=num_steps,
        cond_steps=cond_steps,
        deterministic=False,
        x_goal=goal_tensor,
        decode=True,
        return_context_posterior=True,
        return_aux_rec=True,
        actions=action_buffer,
        lang_embed=language_goal,
        n_pred_eq_gt = False,
    )
    if convert_to_6D:
        action_chunk = action_ortho6d_to_xyzw(action_chunk)
    return rec, action_chunk


def get_gt_data(
    demo,
    camera_names: Tuple[str, ...],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Extract per-timestep GT actions and RGB frames from every demo step.

    # REPLACED: Actioner.get_action_from_demo() (keypoint_discovery-based) removed.
    # That method extracted sparse keypose actions by finding stopped frames.
    # Here all timesteps are returned to support continuous dense evaluation
    # and full-episode video animation.

    Args:
        demo: RLBench Demo object.
        camera_names: cameras whose RGB frames to extract.

    Returns:
        gt_actions: (T, action_dim) array of all GT actions.
        gt_frames_cam0: (T, H, W, 3) uint8 RGB frames for camera_names[0].
        gt_frames_cam1: (T, H, W, 3) for camera_names[1], or None.
    """
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




def _normalize_01(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = arr.astype(np.float32)
    mn = float(arr.min())
    mx = float(arr.max())
    return (arr - mn) / max(mx - mn, eps)


def save_episode_visualizations(
    model,
    actions_history: np.ndarray,
    gt_actions: np.ndarray,
    gt_frames_cam0: np.ndarray,
    gt_frames_cam1: Optional[np.ndarray],
    imagined_traj: Optional[torch.Tensor],
    obs_history: np.ndarray,
    executed_steps: int,
    task_str: str,
    demo_id: int,
    cond_steps: int,
    output_dir: Path,
    gif_fps: float,
    variation : int,
    language_goal : str
) -> None:
    """Save action comparison plots and trajectory animations for one episode.

    Args:
        actions_history: (max_steps, action_dim) predicted actions this episode.
        gt_actions: (T, action_dim) all GT actions from the demo.
        gt_frames_cam0: (T, H, W, 3) GT RGB frames for the primary camera.
        gt_frames_cam1: (T, H, W, 3) GT frames for the second camera, or None.
        imagined_traj: rec tensor from the last model query (shape TBD).
        executed_steps: number of timesteps actually executed.
    """
    act_id = f"{task_str}_demo{demo_id}_var{variation}"
    # max_ts = min(executed_steps, len(gt_actions))
    max_ts = min(len(gt_actions),len(actions_history))
    

    plot_actions(
        gen_acts=actions_history[:max_ts, :],
        gt_acts=gt_actions[:max_ts, :],
        timestep_horizon=max_ts,
        id=act_id,
        root=output_dir,
        ndim=actions_history.shape[-1], 
    )

    if imagined_traj is None:
        return

    gt_traj = gt_frames_cam0[:max_ts]
    pred_traj = obs_history[:max_ts]  # placeholder until rec layout is confirmed

    # TODO: Confirm rec shape and adjust indexing/transposing accordingly.
    # TODO: Handle second camera view — conditionally pass if len(camera_names) >= 2.
    gif_path = os.path.join(output_dir, f"{act_id}_var_{variation}_demo_{demo_id}.gif")
    
    # @TODO integrate multiview
    animate_trajectories(
        orig_trajectory=_normalize_01(gt_traj[:max_ts]),
        pred_trajectory=_normalize_01(pred_traj[0,:max_ts]),
        pred_trajectory_2=_normalize_01(imagined_traj[0,:max_ts]),
        pred_trajectory_22=None, 
        path=gif_path,
        duration=gif_fps,
        rec_to_pred_t=cond_steps,
        t1="-D",
        t2="-S",
        title=f"GT vs Predicted - {language_goal}",
        goal_img=None,
        orig_trajectory2=None, 
        pred_trajectory_12=None,
        goal_img2=None,
    )


def _run_step_loop(
    task,
    mover: Mover,
    initial_obs,
    # obs_buffer: torch.Tensor,
    # action_buffer: torch.Tensor,
    model,
    language_goal: Optional[torch.Tensor],
    goal_tensor: Optional[torch.Tensor],
    camera_names: Tuple[str, ...],
    device: torch.device,
    max_steps: int,
    num_steps: int,
    cond_steps: int,
    chunk_size: int,
    action_dim: int,
    verbose: bool,
    convert_to_6D: bool,
    output_dir: str,
) -> Tuple[float, int, np.ndarray, Optional[torch.Tensor],np.ndarray, np.ndarray]:
    """Execute the continuous timestep-by-timestep rollout loop.

    # REPLACED: keypose-based trajectory execution removed.
    # Old loop: iterate over keyframes from actioner.predict(rgbs, pcds, gripper)
    # with dense interpolation between keyposes. New loop queries the model
    # every chunk_size steps and executes predicted actions one at a time.

    Returns:
        max_reward: highest reward seen across all steps.
        executed_steps: total number of env steps taken.
        actions_history: (max_steps, action_dim) all predicted actions.
        obs_history: (n_views, max_steps, C,H,W) all observed images.
        last_rec: imagined frames from the final model query, or None.
    """
    obs = initial_obs
    actions_history = np.zeros((max_steps, action_dim))
    obs_history = np.zeros((model.n_views,max_steps, *obs.front_rgb.shape))
    imagination_history = np.zeros((model.n_views,max_steps, *obs.front_rgb.shape))
    max_reward = 0.0
    last_rec: Optional[torch.Tensor] = None
    action_chunk: Optional[torch.Tensor] = None
    chunk_cursor = chunk_size  # trigger a model query on the very first step
    
    
    step_id = 0

    # intialize buffers
    obs_buffer =  extract_rgb_tensor(obs, camera_names, device).unsqueeze(0).permute(1,0,2,3,4)

    poses = np.concatenate([obs.gripper_pose, [obs.gripper_open]])
    action_buffer = torch.from_numpy(poses).unsqueeze(0).float().to(device)
    
   
    
    while step_id < max_steps:
        
        ## FOR DEBUGGING
        if chunk_cursor >= chunk_size:
            last_rec, action_chunk = query_model(
                model, obs_buffer, action_buffer,
                goal_tensor, language_goal, num_steps, cond_steps,
                convert_to_6D
            )
            chunk_cursor = 0
        
       
            
        assert action_chunk is not None
        action_chunk = action_chunk.squeeze() 
        # @TODO investigate why it returns num_steps + 1 
        
        # assert action_chunk.shape[0]  == num_steps + cond_steps and action_chunk.shape[1] == action_dim
        last_rec = last_rec.permute(1,0,2,3,4) # [Views, T, C, H , W] -> [T, Views, C, H, W]
        assert last_rec.shape[0] == action_chunk.shape[0] # should be iterating over temporal dimension
        
        # @TODO, do we chop off cond_steps at the beginning?
        for rec,action in zip(last_rec,action_chunk):
            
           
            # handle step_id updates in inner loop – may get past loop g
            if step_id >= max_steps:
                break
            
            # action: torch.Tensor = action_chunk[0, chunk_cursor]  # (action_dim,)
            chunk_cursor += 1

         
            
            
            # update histories         
          
            actions_history[step_id] = action.cpu().numpy()
            

            
            # @TODO 2x check for backward compatibility with single view

            action_np = action.cpu().numpy().copy()
            action_np[-1] = round(action_np[-1])  # binarise gripper open/close

            try:
                obs, reward, terminate, _ = mover(action_np)
                actual_action = np.concatenate([obs.gripper_pose, [obs.gripper_open]])
                actual_action_t = torch.tensor(actual_action).float().to(device)
             
                # action_buffer = update_action_buffer(action_buffer, actual_action_t)
                rgb = extract_rgb_tensor(obs, camera_names, device)
                # obs_buffer = update_obs_buffer(obs_buffer, rgb)
                
                
                # update buffers 
                obs_buffer = rgb.unsqueeze(1)
                action_buffer = actual_action_t.unsqueeze(0)
                
                # obs_buffer = torch.concat([obs_buffer,rgb.unsqueeze(1)],dim=1)
                # action_buffer = torch.concat([action_buffer,actual_action_t.unsqueeze(0)],dim=0)
                
                # if obs_buffer.shape[1] >= 1:
                #     obs_buffer = obs_buffer[:,-1:]
                    
                # if action_buffer.shape[0] >= 1:
                #     action_buffer = action_buffer[-1:]
                    
                # @TODO attempt to generate actions from ground truth in chunks – feed it (gt_obs,gt_act)_t for t = 10, 20, ... 
                # and verify it reconstructions cohesive trajectory
                
                # if that works, verify that generated_obs, generated_action are sufficiently close 
                # can also try adding noise / interpoloating between the two and seeing what happens 
                    
                    
             
                   
                   
                assert rec.shape == rgb.shape
                for view in range(model.n_views):
                    obs_history[view,step_id] = rgb[view].squeeze().permute(1,2,0).cpu().numpy() # output of predict is (C,H,W), reshape to (H,W,C)
                    imagination_history[view,step_id] = rec[view].squeeze().permute(1,2,0).cpu().numpy()
            
            
            except (IKError, ConfigurationPathError, InvalidActionError) as exc:
                print(f"  step {step_id} execution error: {exc}")
                break

            max_reward = max(max_reward, float(reward))
            if verbose:
                print(f"  step {step_id}  reward={reward:.2f}")
            if reward == 1.0 or terminate:
                break
            # update step id 
            step_id +=1 

   
    return max_reward, step_id + 1, actions_history, last_rec, obs_history, imagination_history


@torch.no_grad()
def run_episode(
    env: RLBenchEnv,
    task,
    task_str: str,
    variation: int,
    demo_id: int,
    model,
    language_goal: Optional[torch.Tensor],
    args: Arguments,
    device: torch.device,
    output_dir: Optional[str],
    variation_num:int ,
    config: Dict[str,Any]
) -> float:
    """Set up and run one demo episode; return the max reward achieved.

    Loads the GT demo, resets the task to that initial state, initialises
    observation and action buffers, runs the step loop, and saves
    per-episode visualisations.
    """
    demo = env.get_demo(task_str, variation, episode_index=demo_id)[0]
    gt_actions, gt_frames_cam0, gt_frames_cam1 = get_gt_data(demo, args.cameras)
    gt_frames_cam0 = _normalize_01(gt_frames_cam0)
    gt_frames_cam1 = _normalize_01(gt_frames_cam1)

    # TODO: Define goal frame selection heuristic.
    # Set to None to disable goal conditioning.
    goal_tensor: Optional[torch.Tensor] = None

   
                
    descriptions, obs = task.reset_to_demo(demo)
    mover = Mover(task, max_tries=args.max_tries)
    
    
    # TODO: Define language embedding source. Replace `language_goal` with either
    # a call to an encoder or a lookup from precomputed embeddings.
    if model.language_condition:
    
        tokenizer = T5Tokenizer.from_pretrained('t5-large')
        encoder = T5EncoderModel.from_pretrained('t5-large')
        encoder.eval()  # Set to evaluation mode
        desc = descriptions[np.random.randint(len(descriptions))]
        
        tokenized_desc = tokenizer(desc, max_length=model.language_max_len, padding='max_length', truncation=True, return_tensors='pt')
        tokenized_desc=  tokenized_desc['input_ids']  # Return a 1D tensor
        language_goal = encoder(tokenized_desc).last_hidden_state.squeeze(0).float().to(device)
        print(f"[DEBUG] using desc: {desc=} | {language_goal.median()=}")
    else:
        language_goal = None
    
    
    # first_rgb = extract_rgb_tensor(obs, args.cameras, device)
    # obs_buffer = init_obs_buffer(first_rgb, args.cond_steps)
    # obs_buffer = obs_buffer.repeat(1,2,1,1,1)

    # initial action buffer with first poses
    # poses = np.concatenate([obs.gripper_pose, [obs.gripper_open]])
    # poses = torch.from_numpy(poses).unsqueeze(0).float().to(device)
    # action_buffer = poses
    # action_buffer = action_buffer.repeat(2,1)
    
    # action_buffer = init_action_buffer(args.cond_steps, args.action_dim, device)
    # action_buffer = torch.tensor(gt_actions[0]).unsqueeze(0).float().to(device)
    # action_buffer = torch.zeros_like(torch.tensor(gt_actions[0]).unsqueeze(0).float().to(device))

    max_reward, executed_steps, actions_history, last_rec, obs_history, imagination_history = _run_step_loop(
        task=task, mover=mover, initial_obs=obs,
        # obs_buffer=obs_buffer, action_buffer=action_buffer,
        model=model, language_goal=language_goal, goal_tensor=goal_tensor,
        camera_names=args.cameras, device=device,
        max_steps=args.max_steps, num_steps=args.num_steps,
        cond_steps=args.cond_steps, chunk_size=args.chunk_size,
        action_dim=args.action_dim, verbose=bool(args.verbose),
        convert_to_6D=bool(args.convert_to_6D),
        output_dir=output_dir
    )

   
    save_episode_visualizations(
        model=model, actions_history=actions_history,
        gt_actions=gt_actions, gt_frames_cam0=gt_frames_cam0,
        gt_frames_cam1=gt_frames_cam1, imagined_traj=imagination_history, obs_history = obs_history,
        executed_steps=executed_steps, task_str=task_str,
        demo_id=demo_id, cond_steps=args.cond_steps,
        output_dir=output_dir,
        gif_fps=args.gif_fps,
        variation=variation_num,
        language_goal=desc
    )
    
    # comparison with vidoe generation rollout
    # animate_trajectory_ddlp(model, config, 0, device=device, fig_dir=output_dir,
    #                                     timestep_horizon= args.max_steps // 2, num_trajetories=1,
    #                                     train=True, cond_steps=args.cond_steps,use_all_ctx=False)
    
    # feed sim data into the animate trajectory function
    x_horizon = (torch.from_numpy(_normalize_01(np.stack([gt_frames_cam0,gt_frames_cam1]))).permute(0,1,4,2,3))
    actions_horizon = action_xyzw_to_ortho6d(torch.from_numpy(gt_actions)).unsqueeze(0)
    animate_trajectory_coparticle(
        model, 
        x_horizon=x_horizon, actions_horizon=actions_horizon,lang_embed=language_goal,lang_str=desc,
        epoch=variation_num,device=device,fig_dir=output_dir,timestep_horizon= args.max_steps // 2, num_trajetories=1,
        train=True, cond_steps=args.cond_steps,use_all_ctx=False
    )
    return max_reward


def evaluate_one_variation(
    env: RLBenchEnv,
    task_str: str,
    variation: int,
    task,
    model,
    language_goal: Optional[torch.Tensor],
    args: Arguments,
    device: torch.device,
    output_dir: Optional[str],
    config: Dict[str,Any]
) -> Tuple[float, int]:
    """Evaluate num_episodes demos for one task variation.

    Returns:
        total_success: number of demos where reward == 1.
        num_valid: number of demos that loaded and ran without error.
    """
    task.set_variation(variation)
    total_success = 0
    num_valid = 0

    for demo_id in range(args.num_episodes):
        # try:
        max_reward = run_episode(
            env, task, task_str, variation, demo_id,
            model, language_goal, args, device,
            output_dir,
            variation_num=variation,
            config=config
        )
        num_valid += 1

        if max_reward == 1.0:
            total_success += 1
        print(
            f"{task_str} var={variation} demo={demo_id}"
            f" max_reward={max_reward:.2f}"
            f" SR={total_success}/{num_valid}"
        )

    return total_success, num_valid


def _resolve_variations(
    env: RLBenchEnv,
    task,
    task_str: str,
    args: Arguments,
) -> List[int]:
    """Determine which variation indices to evaluate based on args.variations."""
    if args.variations[-1] > 0:
        n = min(args.variations[-1] + 1, task.variation_count())
        return list(range(n))
    found = glob.glob(os.path.join(str(env.data_path), task_str, "variation*"))
    return [int(p.split("/")[-1].replace("variation", "")) for p in found]


def evaluate_task(
    env: RLBenchEnv,
    task_str: str,
    model,
    language_goal: Optional[torch.Tensor],
    args: Arguments,
    device: torch.device,
    output_dir: Optional[str],
    config: Dict[str,Any]
) -> Dict:
    """Evaluate across all requested variations of one task.

    Launches and shuts down the RLBench environment around the full task run.

    Returns:
        Dict with per-variation success rates (float) and an overall "mean".
    """
    env.env.launch()
    task_type = task_file_to_task_class(task_str)
    task = env.env.get_task(task_type)
    variations = _resolve_variations(env, task, task_str, args)

    var_successes: Dict[int, float] = {}
    var_counts: Dict[int, int] = {}

    for variation in variations:
        total_success, num_valid = evaluate_one_variation(
            env, task_str, variation, task, model, language_goal, args, device, output_dir=output_dir,config=config
        )
        if num_valid > 0:
            var_successes[variation] = total_success / num_valid
            var_counts[variation] = num_valid

    env.env.shutdown()

    total_demos = max(sum(var_counts.values()), 1)
    mean = sum(s * var_counts[v] for v, s in var_successes.items()) / total_demos
    return {**var_successes, "mean": mean}


def main() -> None:
    """Parse arguments, load model and environment, run evaluation over all tasks."""
    args = Arguments().parse_args()
    args.cameras = tuple(c for cam in args.cameras for c in cam.split(","))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    output_dir = os.path.join(args.output_dir,f"run_{datetime.now().strftime('%d.%b.%Y_%I:%M')}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    device = torch.device(args.device)

    # REPLACED: DiffuserActor / Act3D instantiation and checkpoint loading removed.
    # Old code: manually constructed model kwargs, called load_state_dict on
    # model_dict["weight"] after stripping a "module." prefix.
    # New code: load_model() handles config parsing and weight loading internally.
    model = load_model(args.config_path, args.weights_path)
    model.to(device).eval()
    
    # NEW – want to overwrite
    with open(args.config_path, 'r') as fin:
        config = json.load(fin) 
    args.cameras = config.get('views',None)
    if args.cameras is None:
        raise Exception("No views specified")
    

    # TODO: Define language embedding source. Replace `language_goal` with either
    # a call to an encoder or a lookup from precomputed embeddings.
    language_goal: Optional[torch.Tensor] = None

    env = build_rlbench_env(args)
    max_eps_dict = load_episodes()["max_episode_length"]
    task_success_rates: Dict = {}
    
    if args.tasks == []:
        raise Exception("No tasks specified!")
    
   
    
    for task_str in (args.tasks or []):
        if args.max_steps == -1:
            args.max_steps = max_eps_dict.get(task_str, 25)

        result = evaluate_task(env, task_str, model, language_goal, args, device,output_dir = output_dir,config=config)
        print(f"\n{task_str} per-variation SR: {round_floats(result)}")
        print(f"{task_str} mean SR: {round_floats(result['mean'])}")

        task_success_rates[task_str] = result
        with open(os.path.join(output_dir,'eval.json'), "w") as f:
            json.dump(round_floats(task_success_rates), f, indent=4)


if __name__ == "__main__":
    main()


# @TODO 
# 1) feed sim data into animate_trajectories_ddlp 
# if the generation remains good => eval loop problem. Just call animate_trajectories ddlp. 
# if the generation deteirorites => it is a data issue. Use animate_trajectories_ddlp to figure out data flaw. 
# potential hypotheses: we are not trained on intial obs but instead first actions 
# hypothesis – unnormalized ?? 

# @TODO check cond_steps in evaluate script 
# @TODO reduce action chunk based on reconstruction quality
# @TODO check camera image quality