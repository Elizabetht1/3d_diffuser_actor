"""Verification tests for the evaluate_policy_coparticle pipeline.

Each test imports and uses the exact same processing functions as the inference
pipeline, so there is no risk of evaluating a different code path.
"""

import numpy as np
import torch
from typing import Optional

from utils.utils_with_rlbench import Mover, RLBenchEnv
from lpwm_dev.rlbench_utils.geometry import action_xyzw_to_ortho6d, action_ortho6d_to_xyzw
from pyrep.errors import ConfigurationPathError, IKError
from rlbench.backend.exceptions import InvalidActionError


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
        env: Optional[RLBenchEnv] = None,
        model=None,
        task=None,
        task_str: str = "",
        args=None,
        device: torch.device = torch.device("cpu"),
    ):
        self._build_image_transform = build_image_transform
        self._extract_rgb_tensor = extract_rgb_tensor
        self._get_gt_data = get_gt_data
        self._reset_env_to_demo = _reset_env_to_demo
        self._run_step_loop = _run_step_loop

        self.env = env
        self.model = model
        self.task = task
        self.task_str = task_str
        self.args = args
        self.device = device

    def _load_demo(self, variation: int, demo_id: int):
        self.task.set_variation(variation)
        return self.env.get_demo(self.task_str, variation, episode_index=demo_id)[0]

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
        _, obs = self._reset_env_to_demo(self.task, demo)
        mover = Mover(self.task, max_tries=self.args.max_tries)

        max_reward, executed_steps, *_ = self._run_step_loop(
            task=self.task,
            mover=mover,
            initial_obs=obs,
            model=self.model,
            language_goal=language_goal,
            goal_tensor=None,
            camera_names=self.args.cameras,
            device=self.device,
            max_steps=self.args.num_steps,   # execute only one chunk worth of steps
            num_steps=self.args.num_steps,
            cond_steps=self.args.cond_steps,
            chunk_size=self.args.num_steps,  # query model once at t=0, then open-loop
            action_dim=self.args.action_dim,
            verbose=bool(self.args.verbose),
            convert_to_6D=bool(self.args.convert_to_6D),
            output_dir="",
        )

        passed = max_reward == 1.0
        print(f"[test_5] {'PASS' if passed else 'FAIL'}: "
              f"open-loop max_reward={max_reward:.2f} over {executed_steps} steps")
        return passed
