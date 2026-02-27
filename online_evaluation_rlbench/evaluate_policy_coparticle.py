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
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import tap
from pyrep.errors import ConfigurationPathError, IKError
from rlbench.backend.exceptions import InvalidActionError
from rlbench.observation_config import ObservationConfig
from pyrep.const import RenderMode

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

from lpwm_dev.model_factory import build_model  # type: ignore  # black-box; not implemented here
from lpwm_dev.rlbench_utils.action_reconstructions import plot_actions
from lpwm_dev.utils.util_func import animate_trajectories  # type: ignore

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
    max_steps: int = 25
    max_tries: int = 10
    cameras: Tuple[str, ...] = ("left_shoulder", "right_shoulder", "wrist")
    image_size: str = "64,64"
    headless: int = 1
    collision_checking: int = 0

    # Model inference
    num_steps: int = 50    # diffusion denoising steps per query
    cond_steps: int = 1    # frames of observation history fed to model
    chunk_size: int = 8    # actions per chunk before re-querying the model
    action_dim: int = 8    # dimensionality of each action vector
    gif_fps: float = 10.0   # frame rate for saved GIF animations

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
        rgb = torch.from_numpy(rgb_np).float().permute(2, 0, 1) / 255.0
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
    return torch.zeros(cond_steps, action_dim, device=device)


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
    _, _, c, h, w = obs_buffer.shape
    if goal_tensor is None and model.img_goal_condition:
        print("[WARNING] introducing dummy image conditioning data.\n")
        goal_tensor = torch.zeros(1,c,h,w).to(obs_buffer.device)
        
       
    if language_goal is None and model.language_condition:
        print("[WARNING] introducing dummy language conditioning data.\n")
        language_goal = torch.zeros(1,model.language_max_len, model.language_embed_dim).to(obs_buffer.device)
       
    
    # unsqueeze actions to match dimensions expected
    action_buffer = action_buffer.unsqueeze(0) # [B,cond_steps, action_dim]

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
        n_pred_eq_gt = False
    )
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
    executed_steps: int,
    task_str: str,
    demo_id: int,
    cond_steps: int,
    output_dir: Path,
    gif_fps: float,
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
    act_id = f"{task_str}_demo{demo_id}"
    # max_ts = min(executed_steps, len(gt_actions))
    max_ts = min(len(gt_actions),len(actions_history))
    

    plot_actions(
        gen_acts=actions_history[:max_ts, :],
        gt_acts=gt_actions[:max_ts, :],
        timestep_horizon=max_ts,
        id=act_id,
        root=output_dir,
        ndim=model.action_dim,
    )

    if imagined_traj is None:
        return

    gt_traj = gt_frames_cam0[:max_ts]
    pred_traj = gt_traj  # placeholder until rec layout is confirmed

    # TODO: Confirm rec shape and adjust indexing/transposing accordingly.
    # TODO: Handle second camera view — conditionally pass if len(camera_names) >= 2.
    gif_path = output_dir / f"{act_id}.gif"
    
    animate_trajectories(
        orig_trajectory=_normalize_01(gt_traj),
        pred_trajectory=_normalize_01(pred_traj),
        pred_trajectory_2=imagined_traj[0, :max_ts].permute(0,2,3,1).detach().cpu().numpy(),
        pred_trajectory_22=None, # imagined_traj[1, :max_ts].permute(0,2,3,1)
        path=gif_path,
        duration=gif_fps,
        rec_to_pred_t=cond_steps,
        t1="-D",
        t2="-S",
        title=f"GT vs Predicted - {act_id}",
        goal_img=None,
        orig_trajectory2=gt_frames_cam1[:max_ts] if gt_frames_cam1 is not None else None,
        pred_trajectory_12=None,
        goal_img2=None,
    )


def _run_step_loop(
    task,
    mover: Mover,
    initial_obs,
    obs_buffer: torch.Tensor,
    action_buffer: torch.Tensor,
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
) -> Tuple[float, int, np.ndarray, Optional[torch.Tensor]]:
    """Execute the continuous timestep-by-timestep rollout loop.

    # REPLACED: keypose-based trajectory execution removed.
    # Old loop: iterate over keyframes from actioner.predict(rgbs, pcds, gripper)
    # with dense interpolation between keyposes. New loop queries the model
    # every chunk_size steps and executes predicted actions one at a time.

    Returns:
        max_reward: highest reward seen across all steps.
        executed_steps: total number of env steps taken.
        actions_history: (max_steps, action_dim) all predicted actions.
        last_rec: imagined frames from the final model query, or None.
    """
    obs = initial_obs
    actions_history = np.zeros((max_steps, action_dim))
    max_reward = 0.0
    last_rec: Optional[torch.Tensor] = None
    action_chunk: Optional[torch.Tensor] = None
    chunk_cursor = chunk_size  # trigger a model query on the very first step
    step_id = -1

    for step_id in range(max_steps):
        rgb = extract_rgb_tensor(obs, camera_names, device)
        obs_buffer = update_obs_buffer(obs_buffer, rgb)

        if chunk_cursor >= chunk_size:
            last_rec, action_chunk = query_model(
                model, obs_buffer, action_buffer,
                goal_tensor, language_goal, num_steps, cond_steps,
            )
            chunk_cursor = 0

        assert action_chunk is not None
        action: torch.Tensor = action_chunk[0, chunk_cursor]  # (action_dim,)
        chunk_cursor += 1

        action_buffer = update_action_buffer(action_buffer, action)
        actions_history[step_id] = action.cpu().numpy()

        action_np = action.cpu().numpy().copy()
        action_np[-1] = round(action_np[-1])  # binarise gripper open/close

        try:
            obs, reward, terminate, _ = mover(action_np)
        except (IKError, ConfigurationPathError, InvalidActionError) as exc:
            print(f"  step {step_id} execution error: {exc}")
            break

        max_reward = max(max_reward, float(reward))
        if verbose:
            print(f"  step {step_id}  reward={reward:.2f}")
        if reward == 1.0 or terminate:
            break

    return max_reward, step_id + 1, actions_history, last_rec


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
) -> float:
    """Set up and run one demo episode; return the max reward achieved.

    Loads the GT demo, resets the task to that initial state, initialises
    observation and action buffers, runs the step loop, and saves
    per-episode visualisations.
    """
    demo = env.get_demo(task_str, variation, episode_index=demo_id)[0]
    gt_actions, gt_frames_cam0, gt_frames_cam1 = get_gt_data(demo, args.cameras)

    # TODO: Define goal frame selection heuristic.
    # Set to None to disable goal conditioning.
    goal_tensor: Optional[torch.Tensor] = None

    # TODO: Define language embedding source. Replace `language_goal` with either
    # a call to an encoder or a lookup from precomputed embeddings.

    _, obs = task.reset_to_demo(demo)
    mover = Mover(task, max_tries=args.max_tries)
    


    first_rgb = extract_rgb_tensor(obs, args.cameras, device)
    obs_buffer = init_obs_buffer(first_rgb, args.cond_steps)
    action_buffer = init_action_buffer(args.cond_steps, args.action_dim, device)

    
    max_reward, executed_steps, actions_history, last_rec = _run_step_loop(
        task=task, mover=mover, initial_obs=obs,
        obs_buffer=obs_buffer, action_buffer=action_buffer,
        model=model, language_goal=language_goal, goal_tensor=goal_tensor,
        camera_names=args.cameras, device=device,
        max_steps=args.max_steps, num_steps=args.num_steps,
        cond_steps=args.cond_steps, chunk_size=args.chunk_size,
        action_dim=args.action_dim, verbose=bool(args.verbose),
    )

   
    save_episode_visualizations(
        model=model, actions_history=actions_history,
        gt_actions=gt_actions, gt_frames_cam0=gt_frames_cam0,
        gt_frames_cam1=gt_frames_cam1, imagined_traj=last_rec,
        executed_steps=executed_steps, task_str=task_str,
        demo_id=demo_id, cond_steps=args.cond_steps,
        output_dir=args.output_dir,
        gif_fps=args.gif_fps,
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
        
        max_reward = run_episode(
            env, task, task_str, variation, demo_id,
            model, language_goal, args, device,
        )
        num_valid += 1
        # except Exception as exc:
        #     print(f"  demo {demo_id} failed: {exc}")
        #     continue

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
            env, task_str, variation, task, model, language_goal, args, device,
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

    os.makedirs(args.output_dir, exist_ok=True)
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

        result = evaluate_task(env, task_str, model, language_goal, args, device)
        print(f"\n{task_str} per-variation SR: {round_floats(result)}")
        print(f"{task_str} mean SR: {round_floats(result['mean'])}")

        task_success_rates[task_str] = result
        with open(args.output_file, "w") as f:
            json.dump(round_floats(task_success_rates), f, indent=4)


if __name__ == "__main__":
    main()
