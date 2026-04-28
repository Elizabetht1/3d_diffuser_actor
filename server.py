import os
import glob
import itertools
import threading
import queue
import multiprocessing
from dataclasses import dataclass, field
from fastapi import FastAPI, Request
import requests
import uvicorn
import json
from online_evaluation_rlbench.evaluate_policy import evaluate_single_variation, Arguments
import subprocess

import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SERVER_PORT = 8765

app = FastAPI()

# --- SERVER (machine B) ---

_DEVICES = ['cuda:0', 'cuda:1']
_WORKERS_PER_GPU = 3
data_dir = '/data/rlbench2/val'
trials_per_variation = 5
gripper_loc_bounds_file = "tasks/18_peract_tasks_location_bounds_corrected.json"
use_instruction = 1
max_tries = 2
verbose = 1
single_task_gripper_loc_bounds = 0
seed = 0
quaternion_format = "wxyz"  # IMPORTANT: change this to be the same as the training script IF you're not using our checkpoint
headless = 1
image_size = (128, 128)


# ---------------------------------------------------------------------------
# Experiment-level aggregation
# ---------------------------------------------------------------------------

@dataclass
class ExperimentContext:
    experiment_id: str
    callback_url: str
    epoch: int
    config_root: str
    total: int
    results: dict = field(default_factory=dict)   # {(task, variation): {'success_rate', 'num_valid_demos'} | None}
    completed: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, task: str, variation: int, result) -> bool:
        """Record one (task, variation) result. Returns True when all items are done."""
        with self.lock:
            self.results[(task, variation)] = result
            self.completed += 1
            return self.completed == self.total


def _aggregate_and_post(ctx: ExperimentContext):
    task_success: dict = {}
    task_valid_demos: dict = {}
    for (task, var), result in ctx.results.items():
        if result is None:
            continue
        task_success.setdefault(task, {})[var] = result['success_rate']
        task_valid_demos.setdefault(task, {})[var] = result['num_valid_demos']

    aggregated = {}
    for task in task_success:
        var_rates = task_success[task]
        total_valid = sum(task_valid_demos[task].values())
        mean = sum(var_rates.values()) / total_valid if total_valid > 0 else 0.0
        aggregated[task] = {**var_rates, 'mean': mean}

    payload = {
        'results': aggregated,
        'epoch': ctx.epoch,
        'loc': ctx.config_root,
    }
    logger.info(f"Posting results for {ctx.experiment_id} to {ctx.callback_url}")
    requests.post(ctx.callback_url, json=payload)


# ---------------------------------------------------------------------------
# Variation discovery + episode distribution
# ---------------------------------------------------------------------------

def discover_variations(task: str) -> list:
    dirs = glob.glob(os.path.join(data_dir, task, "variation*"))
    return sorted(int(d.split("variation")[-1]) for d in dirs)



# ---------------------------------------------------------------------------
# Args builder (runs in setup worker, once per experiment)
# ---------------------------------------------------------------------------

def _build_args(experiment_id: str) -> Arguments:
    weight_dir = os.path.join('test_weights', experiment_id, 'recent.pth')
    config_fp = os.path.join('test_weights', experiment_id, 'hparams.json')

    with open(config_fp, 'r') as fin:
        config = json.load(fin)

    tasks = config['tasks']
    cameras = config['views']
    embed_type = config.get('embed_type', 't5')

    args = Arguments().parse_args([])
    args.tasks = tuple(tasks)
    args.cameras = tuple(cameras)
    args.embed_type = embed_type
    args.checkpoint = weight_dir
    args.config = config_fp
    args.data_dir = data_dir
    args.num_episodes = trials_per_variation
    args.gripper_loc_bounds_file = gripper_loc_bounds_file
    args.use_instruction = use_instruction
    args.max_tries = max_tries
    args.verbose = verbose
    args.single_task_gripper_loc_bounds = single_task_gripper_loc_bounds
    args.seed = seed
    args.quaternion_format = quaternion_format
    args.headless = headless
    args.image_size = f'{image_size[0]},{image_size[1]}'
    args.verify = 1
    args.max_steps = 45
    args.diffusion_timesteps = 100
    args.num_history = 3
    args.rotation_parametrization = '6D'
    args.dense_interpolation = 1
    args.collision_checking = 0
    args.predict_trajectory = 1
    args.action_dim = 8
    return args


# ---------------------------------------------------------------------------
# Setup worker: SCP → build args → discover variations → enqueue per-var items
# ---------------------------------------------------------------------------

_setup_queue: queue.Queue = queue.Queue()
_gpu_queues = [queue.Queue(), queue.Queue()]
_job_counter = itertools.count()


def _setup_worker():
    logger.info("Setup worker started")
    while True:
        caller_ip, config_root, experiment_id, callback_url, epoch = _setup_queue.get()
        logger.info(f"Setting up {experiment_id}")
        try:
            path = os.path.join("test_weights", experiment_id)
            os.makedirs(path, exist_ok=True)

            pem = os.path.expanduser("~/.ssh/lambda_keys/tallambda.pem")
            src = f"ubuntu@{caller_ip}"
            _scp = ["scp", "-i", pem, "-o", "ConnectTimeout=30", "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=3"]
            subprocess.run([*_scp, f"{src}:{config_root}/hparams.json", path], check=True, timeout=120)
            subprocess.run([*_scp, f"{src}:{config_root}/saves/rlbench_gddlp{experiment_id}.pth", path], check=True, timeout=600)

            for fname in os.listdir(path):
                if not fname.endswith(".pth"):
                    continue
                dest = "best.pth" if fname.endswith("best.pth") else "recent.pth"
                os.rename(os.path.join(path, fname), os.path.join(path, dest))

            args = _build_args(experiment_id)

            items = []
            for task in args.tasks:
                variations = discover_variations(task)
                if not variations:
                    logger.warning(f"No variations found for task {task} in {data_dir}")
                    continue
                for var in variations:
                    items.append((task, var, trials_per_variation))

            if not items:
                logger.error(f"No (task, variation) items found for {experiment_id}; skipping")
                _setup_queue.task_done()
                continue

            ctx = ExperimentContext(
                experiment_id=experiment_id,
                callback_url=callback_url,
                epoch=epoch,
                config_root=config_root,
                total=len(items),
            )
            for task, var, n_demos in items:
                gpu_idx = next(_job_counter) % 2
                _gpu_queues[gpu_idx].put((args, task, var, n_demos, ctx))

            logger.info(f"Enqueued {len(items)} items for {experiment_id}")
        except Exception as e:
            logger.error(f"Setup failed for {experiment_id}: {e}")
        finally:
            _setup_queue.task_done()


# ---------------------------------------------------------------------------
# GPU workers: one spawned process per (task, variation)
# ---------------------------------------------------------------------------

def _gpu_worker(gpu_queue: queue.Queue, device: str):
    logger.info(f"GPU worker started for {device}")
    while True:
        args, task, var, n_demos, ctx = gpu_queue.get()
        logger.info(f"Starting {ctx.experiment_id} | {task} var={var} on {device} ({gpu_queue.qsize()} queued)")
        try:
            result_q = multiprocessing.Queue()
            proc = multiprocessing.get_context('spawn').Process(
                target=evaluate_single_variation,
                args=(args, task, var, n_demos, device, result_q),
                daemon=True,
            )
            proc.start()
            proc.join()
            result = result_q.get() if not result_q.empty() else None
        except Exception as e:
            logger.error(f"Worker error for {ctx.experiment_id} | {task} var={var}: {e}")
            result = None

        all_done = ctx.record(task, var, result)
        logger.info(f"Finished {ctx.experiment_id} | {task} var={var} on {device} (done={ctx.completed}/{ctx.total})")

        if all_done:
            try:
                _aggregate_and_post(ctx)
            except Exception as e:
                logger.error(f"Callback failed for {ctx.experiment_id}: {e}")

        gpu_queue.task_done()


# ---------------------------------------------------------------------------
# Worker startup (main process only)
# ---------------------------------------------------------------------------

if multiprocessing.current_process().name == 'MainProcess':
    threading.Thread(target=_setup_worker, daemon=True).start()
    for _q, _dev in zip(_gpu_queues, _DEVICES):
        for _ in range(_WORKERS_PER_GPU):
            threading.Thread(target=_gpu_worker, args=(_q, _dev), daemon=True).start()


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@app.post("/evaluate")
def evaluate(body: dict, request: Request):
    _setup_queue.put((request.client.host, body["config_root"], body["experiment_id"], body["callback_url"], body['epoch']))
    return {"status": "ack", "setup_queue_depth": _setup_queue.qsize()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
