"""Verification tests for the evaluate_policy_coparticle pipeline.

Each test imports and uses the exact same processing functions as the inference
pipeline, so there is no risk of evaluating a different code path.
"""

import numpy as np
import torch
from typing import Optional, List, Tuple

from utils.mover import Mover


from pyrep.errors import ConfigurationPathError, IKError
from rlbench.backend.exceptions import InvalidActionError


# coparticle utils
from lpwm_dev.rlbench_utils.geometry import action_xyzw_to_ortho6d, action_ortho6d_to_xyzw
from lpwm_dev.rlbench_utils.action_reconstructions import eval_action_recon 
from lpwm_dev.eval.eval_particle_dreamer import plot_actions
import numpy as np


class Verify2:
    """Diagnostic tests for the coparticle evaluation pipeline.

    Takes a pre-loaded actioner and demo.  All model/env parameters are
    derived from actioner attributes so no separate args object is needed.
    Helper methods (image transform, env reset, step loop) are defined as
    instance methods to keep the class self-contained.
    """

    def __init__(
        self,
        actioner,
        demo,
        env=None,
        task=None,
        task_str: str = "",
        max_steps: int = 25,
        max_tries: int = 10,
        verbose: bool = False,
        device: torch.device = torch.device("cpu"),
        logdir: str = "",
    ):
        self.env = env
        self.model = actioner._policy
        self.task = task
        self.task_str = task_str
        self.device = device
        self.logdir = logdir
        self.actioner = actioner
        self.demo = demo

        self.max_steps = max_steps
        self.max_tries = max_tries
        self.verbose = verbose

        # Derive model/env config from actioner
        self.cameras = actioner._apply_cameras
        self.num_steps = actioner.num_pred_steps
        self.cond_steps = actioner.cond_steps
        self.action_dim = actioner._action_dim
        self.convert_6D = actioner._convert_6D

        self.gt_actions, self.gt_frames_cam0, self.gt_frames_cam1 = self._get_gt_data(
            self.demo, camera_names=self.cameras
        )

    # ----------------------------------------------------------- helpers

    def _build_image_transform(self, image_size: int):
        import torchvision.transforms as transforms
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(image_size, antialias=True),
            transforms.CenterCrop(image_size),
        ])

    def _extract_rgb_tensor(self, obs, camera_names, device, transform=None) -> torch.Tensor:
        """Extract preprocessed RGB tensor from one obs. Returns (num_views, 3, H, W)."""
        frames: List[torch.Tensor] = []
        for cam in camera_names:
            rgb_np: np.ndarray = getattr(obs, f"{cam}_rgb")
            if transform is not None:
                rgb = transform(rgb_np)
            else:
                rgb = torch.from_numpy(rgb_np.astype(np.float32) / 255.0).permute(2, 0, 1)
            frames.append(rgb)
        return torch.stack(frames).to(device)

    def _reset_env_to_demo(self, task, demo):
        descriptions, obs = task.reset_to_demo(demo)
        return descriptions, obs


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

    # ------------------------------------------------------------------ tests

    def test_1_replay_demo(self, variation: int = 0, demo_id: int = 0) -> bool:
        """Verify that ground truth actions are replayable and result in 100% success rate."""
        gt_actions, _, _ = self._get_gt_data(self.demo, self.cameras)

        _, obs = self._reset_env_to_demo(self.task, self.demo)
        mover = Mover(self.task, max_tries=self.max_tries)

        reward = 0.0
        for action_np in gt_actions:
            try:
                obs, reward, terminate, _ = mover(action_np)
            except (IKError, ConfigurationPathError, InvalidActionError) as e:
                print(f"[test_1] FAIL: execution error: {e}")
                return False
            if reward == 1.0:
                print("[test_1] PASS: demo replay achieved reward=1.0")
                return True

        passed = reward == 1.0
        print(f"[test_1] {'PASS' if passed else 'FAIL'}: max_reward={reward:.2f}")
        return passed

    def test_2_image_preprocessing(self, variation: int = 0, demo_id: int = 0) -> bool:
        """Verify image preprocessing in inference matches training (size, color scale)."""
        image_size = getattr(self.model, 'image_size', 128) if self.model else 128
        transform = self._build_image_transform(image_size)
        passed = True

        for cam in self.cameras:
            raw: np.ndarray = getattr(self.demo[0], f"{cam}_rgb")
            if raw.dtype != np.uint8:
                print(f"[test_2] FAIL {cam}: dtype={raw.dtype}, expected uint8")
                passed = False
                continue

            frame = self._extract_rgb_tensor(self.demo[0], (cam,), self.device, transform)
            # frame: (1, 3, H, W)

            if frame.shape != (1, 3, image_size, image_size):
                print(f"[test_2] FAIL {cam}: shape {tuple(frame.shape)} != (1,3,{image_size},{image_size})")
                passed = False
            if frame.min() < 0.0 or frame.max() > 1.0:
                print(f"[test_2] FAIL {cam}: values outside [0,1]: "
                      f"min={frame.min():.4f} max={frame.max():.4f}")
                passed = False

        if passed:
            print(f"[test_2] PASS: all cameras -> ({image_size}x{image_size}), values in [0,1]")
        return passed

    def test_3_action_preprocessing(self, variation: int = 0, demo_id: int = 0) -> bool:
        """Verify action preprocessing: ranges, quaternion norms, gripper binarisation."""
        gt_actions, _, _ = self._get_gt_data(self.demo, self.cameras)
        passed = True

        pos = gt_actions[:, :3]
        if pos.max() > 5.0 or pos.min() < -5.0:
            print(f"[test_3] FAIL: position out of expected workspace: "
                  f"min={pos.min():.3f} max={pos.max():.3f}")
            passed = False

        quats = gt_actions[:, 3:7]
        norms = np.linalg.norm(quats, axis=-1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            print(f"[test_3] FAIL: quaternion norms not unit: "
                  f"min={norms.min():.4f} max={norms.max():.4f}")
            passed = False

        gripper = gt_actions[:, 7]
        if np.any((gripper != 0.0) & (gripper != 1.0)):
            print(f"[test_3] WARN: gripper not strictly binary: {np.unique(gripper)}")

        _, obs = self._reset_env_to_demo(self.task, self.demo)
        expected_init = np.concatenate([obs.gripper_pose, [obs.gripper_open]])
        if not np.allclose(expected_init, gt_actions[0], atol=1e-4):
            print(f"[test_3] FAIL: initial obs pose != gt_actions[0]\n"
                  f"  obs:        {expected_init}\n"
                  f"  gt_actions: {gt_actions[0]}")
            passed = False

        if passed:
            print(f"[test_3] PASS: T={len(gt_actions)}, "
                  f"pos=[{pos.min():.3f},{pos.max():.3f}], quat norms~1, init buffer OK")
        return passed

    def test_4_quant_conversion(self, n_samples: int = 1000) -> bool:
        """Verify quaternion == to_xyzw(to_6D(action)) round-trip (up to sign flip)."""
        pos = torch.randn(n_samples, 3)
        quats = torch.randn(n_samples, 4)
        quats = quats / quats.norm(dim=-1, keepdim=True)
        gripper = torch.randint(0, 2, (n_samples, 1)).float()
        actions = torch.cat([pos, quats, gripper], dim=-1)

        actions_6d = action_xyzw_to_ortho6d(actions)
        actions_back = action_ortho6d_to_xyzw(actions_6d)

        pos_ok = torch.allclose(actions[:, :3], actions_back[:, :3], atol=1e-5)
        grip_ok = torch.allclose(actions[:, 7:], actions_back[:, 7:], atol=1e-5)
        dot = (actions[:, 3:7] * actions_back[:, 3:7]).sum(-1).abs()
        quat_ok = torch.allclose(dot, torch.ones_like(dot), atol=1e-4)

        passed = pos_ok and grip_ok and quat_ok
        print(f"[test_4] {'PASS' if passed else 'FAIL'}: "
              f"pos={pos_ok}, quat={quat_ok}, gripper={grip_ok} ({n_samples} samples)")
        return passed

    def test_5_replay_open_loop(
        self,
        variation: int = 0,
        demo_id: int = 0,
    ) -> bool:
        """Generate a full open-loop trajectory and play it in simulation.

        Queries the model once at t=0 (chunk_size=num_steps) and executes all
        predicted actions without re-querying.
        """
        descs, obs = self._reset_env_to_demo(self.task, self.demo)
        mover = Mover(self.task, max_tries=self.max_tries)
        language_goal = self.actioner._instr

        # @TODO implement me 
        self.actioner.predict( ) 

        T = self.gt_actions.shape[0]
        plot_actions(
            actions_history[:T],
            self.gt_actions,
            T,
            ndim=8,
            id="test_5",
            root=self.logdir
        )

        passed = max_reward == 1.0
        print(f"[test_5] {'PASS' if passed else 'FAIL'}: "
              f"open-loop max_reward={max_reward:.2f} over {executed_steps} steps")
        return passed

    def test_6_replay_recon(self, variation: int = 0, demo_id: int = 0):
        gt_actions, _, _ = self._get_gt_data(self.demo, self.cameras)

        parsed_gt_actions = torch.from_numpy(gt_actions).float().to(self.device)
        parsed_gt_actions = action_xyzw_to_ortho6d(parsed_gt_actions).unsqueeze(1)
        recon_actions = eval_action_recon(self.model, parsed_gt_actions, deterministic=True,
                                         plot=True, logdir=self.logdir, epoch=6, id="test_6")
        recon_actions = action_ortho6d_to_xyzw(recon_actions).cpu().numpy()

        _, obs = self._reset_env_to_demo(self.task, self.demo)
        mover = Mover(self.task, max_tries=self.max_tries)

        reward = 0.0
        for action_np in recon_actions:
            try:
                obs, reward, terminate, _ = mover(action_np)
            except (IKError, ConfigurationPathError, InvalidActionError) as e:
                print(f"[test_6] FAIL: execution error: {e}")
                return False
            if reward == 1.0:
                print("[test_6] PASS: demo replay achieved reward=1.0")
                return True

        passed = reward == 1.0
        print(f"[test_6] {'PASS' if passed else 'FAIL'}: max_reward={reward:.2f}")
        return passed

    def test_7_replay_recon_with_ctx(self, variation: int = 0, demo_id: int = 0):
        """Evaluate reconstruction fidelity with ground-truth latent context."""
        desc, obs = self._reset_env_to_demo(self.task, self.demo)
        mover = Mover(self.task, max_tries=self.max_tries)

        T, adim = self.gt_actions.shape

        parsed_gt_actions = torch.from_numpy(self.gt_actions).float().to(self.device)
        parsed_gt_actions = action_xyzw_to_ortho6d(parsed_gt_actions).unsqueeze(0)

        parsed_gt_frames = (
            torch.from_numpy(np.stack([self.gt_frames_cam0, self.gt_frames_cam1]))
            .float()
            .permute(0, 1, 4, 2, 3)
            .to(self.device)
        )

        lang_embed = self.actioner._instr
        ts_horizon = T

        recon_actions = None
        for ts in range(0, T, ts_horizon):
            pred_horizon = min(T - ts, ts_horizon)
            with torch.no_grad():
                _, recon_actions_chunk, _, _ = self.model.sample_from_x(
                    parsed_gt_frames[:, ts:ts + pred_horizon],
                    num_steps=pred_horizon,
                    deterministic=True,
                    cond_steps=self.cond_steps,
                    use_all_ctx=True,
                    actions=parsed_gt_actions[:, ts:ts + pred_horizon],
                    lang_embed=lang_embed,
                    return_aux_rec=True,
                    n_pred_eq_gt=False,
                )
            recon_actions_chunk = recon_actions_chunk[:, self.cond_steps:]
            if recon_actions is None:
                recon_actions = recon_actions_chunk
            else:
                recon_actions = torch.cat([recon_actions, recon_actions_chunk], dim=1)

        recon_actions = action_ortho6d_to_xyzw(recon_actions).squeeze().cpu().numpy()

        reward = 0.0
        for action_np in recon_actions:
            try:
                obs, reward, terminate, _ = mover(action_np)
            except (IKError, ConfigurationPathError, InvalidActionError) as e:
                print(f"[test_7] FAIL: execution error: {e}")
                return False
            if reward == 1.0:
                print("[test_7] PASS: demo replay achieved reward=1.0")
                return True

        plot_actions(
            recon_actions.squeeze(),
            self.gt_actions.squeeze(),
            T,
            ndim=8,
            id="test_7",
            root=self.logdir
        )

        passed = reward == 1.0
        print(f"[test_7] {'PASS' if passed else 'FAIL'}: max_reward={reward:.2f}")
        return passed

class Verify:
    """Diagnostic tests for the evaluate_policy_coparticle pipeline.

    Constructor args match the objects already built in main():
        env, model, task, task_str, args, device.

    All simulation tests (1-3, 5) require env, task, task_str to be set.
    test_4 is a pure numerical check and requires only lpwm geometry imports.
    """

    def __init__(
        self,
        build_image_transform,
        extract_rgb_tensor,
        get_gt_data,
        query_model,
        _reset_env_to_demo,
        _run_step_loop,
        embed,
        env = None,
        model=None,
        task=None,
        task_str: str = "",
        args=None,
        device: torch.device = torch.device("cpu"),
        logdir = ""
    ):
        self._build_image_transform = build_image_transform
        self._extract_rgb_tensor = extract_rgb_tensor
        self._get_gt_data = get_gt_data
        self._reset_env_to_demo = _reset_env_to_demo
        self._run_step_loop = _run_step_loop
        self.embed = embed

        self.env = env
        self.model = model
        self.task = task
        self.task_str = task_str
        self.args = args
        self.device = device
        self.logdir = logdir 

    def _load_demo(self, variation: int, demo_id: int):
        self.task.set_variation(variation)
        
        return self.env.get_demo(self.task_str, variation, episode_index=demo_id)[0]
    
    def _get_gt_data(
        demo,
        camera_names: Tuple[str, ...],
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Extract per-timestep GT actions and RGB frames from every demo step.

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

    # ------------------------------------------------------------------ tests

    def test_1_replay_demo(self, variation: int = 0, demo_id: int = 0) -> bool:
        """Verify that ground truth actions are replayable and result in 100% success rate."""
        demo = self._load_demo(variation, demo_id)
        gt_actions, _, _ = self._get_gt_data(demo, self.args.cameras)

        _, obs = self._reset_env_to_demo(self.task, demo)
        mover = Mover(self.task, max_tries=self.args.max_tries)

        reward = 0.0
        for action_np in gt_actions:
            try:
                obs, reward, terminate, _ = mover(action_np)
            except (IKError, ConfigurationPathError, InvalidActionError) as e:
                print(f"[test_1] FAIL: execution error: {e}")
                return False
            if reward == 1.0:
                print("[test_1] PASS: demo replay achieved reward=1.0")
                return True

        passed = reward == 1.0
        print(f"[test_1] {'PASS' if passed else 'FAIL'}: max_reward={reward:.2f}")
        return passed

    def test_2_image_preprocessing(self, variation: int = 0, demo_id: int = 0) -> bool:
        """Verify image preprocessing in inference matches training (size, color scale)."""
        demo = self._load_demo(variation, demo_id)
        image_size = getattr(self.model, 'image_size', 128) if self.model else 128
        transform = self._build_image_transform(image_size)   # exact same function as inference
        passed = True

        for cam in self.args.cameras:
            raw: np.ndarray = getattr(demo[0], f"{cam}_rgb")
            if raw.dtype != np.uint8:
                print(f"[test_2] FAIL {cam}: dtype={raw.dtype}, expected uint8")
                passed = False
                continue

            # Use extract_rgb_tensor — same call path as _run_step_loop
            frame = self._extract_rgb_tensor(demo[0], (cam,), self.device, transform)
            # frame: (1, 3, H, W)

            if frame.shape != (1, 3, image_size, image_size):
                print(f"[test_2] FAIL {cam}: shape {tuple(frame.shape)} != (1,3,{image_size},{image_size})")
                passed = False
            if frame.min() < 0.0 or frame.max() > 1.0:
                print(f"[test_2] FAIL {cam}: values outside [0,1]: "
                      f"min={frame.min():.4f} max={frame.max():.4f}")
                passed = False

        if passed:
            print(f"[test_2] PASS: all cameras → ({image_size}x{image_size}), values in [0,1]")
        return passed

    def test_3_action_preprocessing(self, variation: int = 0, demo_id: int = 0) -> bool:
        """Verify action preprocessing matches inference: ranges, quaternion norms,
        gripper binarisation, and initial action buffer seeding."""
        demo = self._load_demo(variation, demo_id)

        # get_gt_data is the exact same call used in run_episode()
        gt_actions, _, _ = self._get_gt_data(demo, self.args.cameras)  # (T, 8)
        passed = True

        pos = gt_actions[:, :3]
        if pos.max() > 5.0 or pos.min() < -5.0:
            print(f"[test_3] FAIL: position out of expected workspace: "
                  f"min={pos.min():.3f} max={pos.max():.3f}")
            passed = False

        quats = gt_actions[:, 3:7]
        norms = np.linalg.norm(quats, axis=-1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            print(f"[test_3] FAIL: quaternion norms not unit: "
                  f"min={norms.min():.4f} max={norms.max():.4f}")
            passed = False

        gripper = gt_actions[:, 7]
        if np.any((gripper != 0.0) & (gripper != 1.0)):
            print(f"[test_3] WARN: gripper not strictly binary: {np.unique(gripper)}")

        # Verify the initial action buffer seeded in _run_step_loop matches demo[0]
        _, obs = self._reset_env_to_demo(self.task, demo)
        expected_init = np.concatenate([obs.gripper_pose, [obs.gripper_open]])
        if not np.allclose(expected_init, gt_actions[0], atol=1e-4):
            print(f"[test_3] FAIL: initial obs pose ≠ gt_actions[0]\n"
                  f"  obs:        {expected_init}\n"
                  f"  gt_actions: {gt_actions[0]}")
            passed = False

        if passed:
            print(f"[test_3] PASS: T={len(gt_actions)}, "
                  f"pos=[{pos.min():.3f},{pos.max():.3f}], quat norms~1, init buffer OK")
        return passed

    def test_4_quant_conversion(self, n_samples: int = 1000) -> bool:
        """Verify quaternion == to_xyzw(to_6D(action)) round-trip (up to sign flip)."""
        pos = torch.randn(n_samples, 3)
        quats = torch.randn(n_samples, 4)
        quats = quats / quats.norm(dim=-1, keepdim=True)
        gripper = torch.randint(0, 2, (n_samples, 1)).float()
        actions = torch.cat([pos, quats, gripper], dim=-1)       # (N, 8)

        actions_6d = action_xyzw_to_ortho6d(actions)             # (N, 10)
        actions_back = action_ortho6d_to_xyzw(actions_6d)        # (N, 8)

        pos_ok = torch.allclose(actions[:, :3], actions_back[:, :3], atol=1e-5)
        grip_ok = torch.allclose(actions[:, 7:], actions_back[:, 7:], atol=1e-5)
        # q and -q encode the same rotation — check |cos| ≈ 1
        dot = (actions[:, 3:7] * actions_back[:, 3:7]).sum(-1).abs()
        quat_ok = torch.allclose(dot, torch.ones_like(dot), atol=1e-4)

        passed = pos_ok and grip_ok and quat_ok
        print(f"[test_4] {'PASS' if passed else 'FAIL'}: "
              f"pos={pos_ok}, quat={quat_ok}, gripper={grip_ok} ({n_samples} samples)")
        return passed

    def test_5_replay_open_loop(
        self,
        variation: int = 0,
        demo_id: int = 0,
        language_goal: Optional[torch.Tensor] = None,
    ) -> bool:
        """Generate a full open-loop trajectory and play it in simulation.

        Uses _run_step_loop with chunk_size=num_steps so the model is queried
        exactly once at t=0 and all predicted actions are executed without
        re-querying — identical to the closed-loop pipeline except for the
        chunk_size override.
        """
        demo = self._load_demo(variation, demo_id)
        descs, obs = self._reset_env_to_demo(self.task, demo)
        mover = Mover(self.task, max_tries=self.args.max_tries)
        
        language_goal = self.embed(descs,device=self.device)

        max_reward, executed_steps, actions_history, last_rec, obs_history, imagination_history = self._run_step_loop(
            task=self.task,
            mover=mover,
            initial_obs=obs,
            model=self.model,
            language_goal=language_goal,
            goal_tensor=None,
            camera_names=self.args.cameras,
            device=self.device,
            max_steps=self.args.max_steps,   # execute only one chunk worth of steps
            num_steps=self.args.num_steps,
            cond_steps=self.args.cond_steps,
            chunk_size=self.args.num_steps,  # query model once at t=0, then open-loop
            action_dim=self.args.action_dim,
            verbose=bool(self.args.verbose),
            convert_to_6D=bool(self.args.convert_to_6D),
            output_dir="",
        )
        
        gt_actions, _,_ = self._get_gt_data(demo,camera_names=self.args.cameras)
        T = gt_actions.shape[0]
        plot_actions(
            actions_history[:T],
            gt_actions,
            T,
            ndim=8,
            id="test_5",
            root=self.logdir
        )


        passed = max_reward == 1.0
        print(f"[test_5] {'PASS' if passed else 'FAIL'}: "
              f"open-loop max_reward={max_reward:.2f} over {executed_steps} steps")
        return passed
    
    def test_6_replay_recon(self, variation: int = 0,demo_id: int = 0):
        
        demo = self._load_demo(variation, demo_id)
        gt_actions, _, _ = self._get_gt_data(demo, self.args.cameras)
        
        # generate the reconstruction to rollout 
        
        parsed_gt_actions = torch.from_numpy(gt_actions).float().to(self.device)
        parsed_gt_actions = action_xyzw_to_ortho6d(parsed_gt_actions).unsqueeze(1)
        recon_actions = eval_action_recon(self.model,parsed_gt_actions,deterministic=True,plot=True,logdir=self.logdir,epoch=6,id="test_6")
        recon_actions = action_ortho6d_to_xyzw(recon_actions).cpu().numpy()

        # intialize enviornment mover 
        _, obs = self._reset_env_to_demo(self.task, demo)
        mover = Mover(self.task, max_tries=self.args.max_tries)

        reward = 0.0
        for action_np in recon_actions:
            try:
                obs, reward, terminate, _ = mover(action_np)
            except (IKError, ConfigurationPathError, InvalidActionError) as e:
                print(f"[test_6] FAIL: execution error: {e}")
                return False
            if reward == 1.0:
                print("[test_6] PASS: demo replay achieved reward=1.0")
                return True

        passed = reward == 1.0
        print(f"[test_6] {'PASS' if passed else 'FAIL'}: max_reward={reward:.2f}")
        return passed
    
    def test_7_replay_recon_with_ctx(self, variation: int = 0,demo_id: int = 0):
        """ evaluate reconstruction fidelity when we only have access to ground truth latent ctx instead of latent actions """
        
        demo = self._load_demo(variation, demo_id)
        gt_actions, gt_frames_cam0, gt_frames_cam1 = self._get_gt_data(demo, self.args.cameras)
        
        # intialize enviornment mover 
        desc, obs = self._reset_env_to_demo(self.task, demo)
        mover = Mover(self.task, max_tries=self.args.max_tries)
        
        # generate the reconstruction to rollout 
        # we allow access to ground truth ctx 
        T, adim = gt_actions.shape
        
        # prepare actions 
        parsed_gt_actions = torch.from_numpy(gt_actions).float().to(self.device)
        parsed_gt_actions = action_xyzw_to_ortho6d(parsed_gt_actions).unsqueeze(0)
        
        # prepare images 
        parsed_gt_frames = torch.from_numpy(np.stack([gt_frames_cam0,gt_frames_cam1])).float().permute(0,1,4,2,3).to(self.device)
        
        # prepare language embedding 
        lang_embed = self.embed(descriptions=desc,device=self.device,max_length=self.model.language_max_len).unsqueeze(0)
        
        # predict in chunks
        ts_horizon = T
        
        recon_actions = None
        for ts in range(0,T,ts_horizon):
            pred_horizon = min(T - ts, ts_horizon)
            with torch.no_grad():
                _,recon_actions_chunk,_,_ = self.model.sample_from_x(parsed_gt_frames[:,ts:ts+pred_horizon], num_steps= pred_horizon, deterministic=True,
                                                cond_steps=self.args.cond_steps, use_all_ctx=True, actions=parsed_gt_actions[:,ts:ts+pred_horizon],
                                                lang_embed=lang_embed, return_aux_rec=True,n_pred_eq_gt=False)
            recon_actions_chunk = recon_actions_chunk[:,self.args.cond_steps:] # @TODO verify this makes sense 
            # my assumption is that the first action is the ground truth reconstruction, we don't want that, we want the prediction
            if recon_actions is None:
                recon_actions = recon_actions_chunk
            else:
                
                recon_actions = torch.cat([recon_actions,recon_actions_chunk],dim=1)
        

        
            
            
            
            
            
            
            
        recon_actions = action_ortho6d_to_xyzw(recon_actions).squeeze().cpu().numpy()
        
        reward = 0.0
        for action_np in recon_actions:
            try:
                obs, reward, terminate, _ = mover(action_np)
            except (IKError, ConfigurationPathError, InvalidActionError) as e:
                print(f"[test_7] FAIL: execution error: {e}")
                return False
            if reward == 1.0:
                print("[test_7] PASS: demo replay achieved reward=1.0")
                return True

        plot_actions(
            recon_actions.squeeze(),
            gt_actions.squeeze(),
            T,
            ndim=8,
            id="test_7",
            root=self.logdir
        )

        passed = reward == 1.0
        print(f"[test_7] {'PASS' if passed else 'FAIL'}: max_reward={reward:.2f}")
        return passed
    
        
