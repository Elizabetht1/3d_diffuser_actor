import os
import itertools
import threading
import queue
import multiprocessing
from fastapi import FastAPI, Request
import requests
import uvicorn
import json
from online_evaluation_rlbench.evaluate_policy import main_coparticle, Arguments
import subprocess

import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SERVER_PORT = 8765

app = FastAPI()

# --- SERVER (machine B) ---

_DEVICES = ['cuda:0', 'cuda:1']
_WORKERS_PER_GPU = 3
data_dir='/data/rlbench2/val'
num_episodes=10
gripper_loc_bounds_file="tasks/18_peract_tasks_location_bounds_corrected.json"
use_instruction=1
max_tries=2
verbose=1
single_task_gripper_loc_bounds=0
seed=0
quaternion_format="wxyz"  # IMPORTANT: change this to be the same as the training script IF you're not using our checkpoint
headless=1
image_size=(128,128)
variations=tuple(range(60))

def run_experiment(experiment_id, device):

    weight_dir = os.path.join('test_weights', experiment_id, 'recent.pth')
    config_fp = os.path.join('test_weights', experiment_id, 'hparams.json')

    # infer from hparams
    with open(config_fp, 'r') as fin:
        config = json.load(fin)

    tasks = config['tasks']
    cameras = config['views']
    embed_type = config.get('embed_type','t5')

    print(f"exp_id:     {experiment_id}")
    print(f"tasks:      {tasks}")
    print(f"cameras:    {cameras}")
    print(f"embed_type: {embed_type}")
    print(f"device:     {device}")

    os.environ['DISPLAY'] = ':1'

    args = Arguments().parse_args([])
    args.tasks = tuple(tasks)
    args.cameras = tuple(cameras)
    args.embed_type = embed_type
    args.checkpoint = weight_dir
    args.config = config_fp
    args.device = device
    args.data_dir = data_dir
    args.num_episodes = num_episodes
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
    args.variations = variations
    args.max_steps = 45
    args.diffusion_timesteps = 100
    args.num_history = 3
    args.rotation_parametrization = '6D'
    args.dense_interpolation = 1
    args.collision_checking = 0
    args.predict_trajectory = 1
    args.action_dim = 8
    # args.instructions = 'instructions/peract/instructions.pkl'

    return main_coparticle(args)



def _run_eval(caller_ip: str, config_root: str, experiment_id: str, callback_url: str, epoch: int, device: str):
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

        
        result = run_experiment(experiment_id, device)
        if result is None:
            logging.error(f"run_experiment returned None for {experiment_id}; skipping callback")
            return
        sim_results, sim_results_loc = result
        


        results = {
                'results': sim_results, 
                'epoch': epoch,
                'loc': config_root,
                'sim_results_loc': sim_results_loc
                }

        logging.info(f"posting results to {callback_url} ")
        requests.post(callback_url, json=results)
    except Exception as e:
        logging.error(f"Error encountered: {e}")


_gpu_queues = [queue.Queue(), queue.Queue()]
_job_counter = itertools.count()


def _gpu_worker(gpu_queue: queue.Queue, device: str):
    logger.info(f"GPU worker started for {device}")
    while True:
        caller_ip, config_root, experiment_id, callback_url, epoch = gpu_queue.get()
        logger.info(f"Starting {experiment_id} on {device} ({gpu_queue.qsize()} queued on {device})")
        proc = multiprocessing.get_context('spawn').Process(
            target=_run_eval,
            args=(caller_ip, config_root, experiment_id, callback_url, epoch, device),
            daemon=True,
        )
        proc.start()
        proc.join()
        logger.info(f"Finished {experiment_id} on {device}")
        gpu_queue.task_done()


if multiprocessing.current_process().name == 'MainProcess':
    for _q, _dev in zip(_gpu_queues, _DEVICES):
        for _ in range(_WORKERS_PER_GPU):
            threading.Thread(target=_gpu_worker, args=(_q, _dev), daemon=True).start()


@app.post("/evaluate")
def evaluate(body: dict, request: Request):
    gpu_idx = next(_job_counter) % 2
    _gpu_queues[gpu_idx].put((request.client.host, body["config_root"], body["experiment_id"], body["callback_url"], body['epoch']))
    return {"status": "ack", "gpu": _DEVICES[gpu_idx], "queue_depth": _gpu_queues[gpu_idx].qsize()}



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)