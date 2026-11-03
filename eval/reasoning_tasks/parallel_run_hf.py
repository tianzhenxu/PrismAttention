#!/usr/bin/env python3
import subprocess
import os
import sys
import argparse
import time
from collections import deque # Use deque for efficient pop/append


def choose_task_config(model_size, output_dir):
    output_dir = output_dir.lower()
    if model_size != "32B":
        task_config = {
            "aime24": {"bs": 10, "total_run": 64},
            "aime25": {"bs": 10, "total_run": 64},
            "math": {"bs": 50, "total_run": 8},
            "gpqa": {"bs": 20, "total_run": 16},
            "olympiadbench": {"bs": 15, "total_run": 8},
            "livecodebench": {"bs": 15, "total_run": 8},
        }
    else:
        raise ValueError(f"Not support model_size: {model_size}")

    return task_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tasks using subprocess.")
    parser.add_argument("--model_dir", type=str,
                        default="Qwen/Qwen3-4B",
                        help="Model directory path")
    parser.add_argument("--model_size", type=str, default="14B", help="model_size")
    parser.add_argument("--tasks", type=str, default="aime",
                        help="Comma-separated list of tasks (e.g., aime,math,gpqa)")
    parser.add_argument("--output_dir", type=str, default="./results/aime",
                        help="Directory to store output results")
    parser.add_argument("--attention_method", type=str, default="prism",
                        help="attention implementations")
    parser.add_argument("--limit", type=int, default=-1,
                        help="Limit for the number of samples to process")
    parser.add_argument("--num_gpus", default=8, type=int)
    parser.add_argument("--token_budget", default="2048", type=str)
    parser.add_argument("--max_tokens", default="32768", type=str)
    parser.add_argument("--start_layer", type=int, default=0, help="Start sparse layer, '0' means all layers")
    args = parser.parse_args()

    limit = args.limit
    num_gpus = args.num_gpus
    max_tokens = args.max_tokens

    model_dir = args.model_dir
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    token_budgets = [t.strip() for t in args.token_budget.split(",") if t.strip()]

    model_subfolder = os.path.basename(model_dir.rstrip('/'))
    output_dir = os.path.join(args.output_dir, model_subfolder)
    attention_method = args.attention_method

    task_config = choose_task_config(args.model_size, output_dir)

    for task in tasks:
        if task not in task_config:
            print(f"Error: Unknown task '{task}'")
            sys.exit(1)

        bs = task_config[task]["bs"]
        total_run = task_config[task]["total_run"]

        print(f"\n{'=' * 40}")
        print(f"Starting task: {task}")
        print(f"Batch size: {bs} | total_run: {total_run}")

        for token_budget in token_budgets:
            param_desc = f"token_budget={token_budget}"

            print(f"\n{'─' * 30}")
            print(f"Processing task={task} | attention={attention_method} | {param_desc}")

            output_config_subdir = os.path.join(
                output_dir,
                f"{task}_bs{bs}_budget{token_budget}_start{args.start_layer}_{attention_method}",
            )
            os.makedirs(output_config_subdir, exist_ok=True)

            overall_summary_filepath = os.path.join(output_config_subdir, "overall_summary.txt")
            if os.path.exists(overall_summary_filepath):
                print(f"Skip {param_desc} because overall_summary.txt already exists.")
                continue

            active_procs = {}
            available_gpus = deque(range(num_gpus))
            run_counter = 0

            while run_counter < total_run or active_procs:

                for proc, info in list(active_procs.items()):
                    if proc.poll() is None:
                        continue
                    print(f"Run {info['run_id']} on GPU {info['gpu_id']} "
                          f"finished with exit code {proc.returncode}.")
                    available_gpus.append(info["gpu_id"])
                    del active_procs[proc]


                while run_counter < total_run and available_gpus:
                    gpu_id = available_gpus.popleft()
                    current_run_id = run_counter
                    run_counter += 1

                    print(f"Launching run {current_run_id} on GPU {gpu_id}...")

                    env = os.environ.copy()
                    cmd = [
                        sys.executable, "eval_hf.py",
                        "--model_name_or_path", model_dir,
                        "--data_name", task,
                        "--batch_size", str(bs),
                        "--limit", str(limit),
                        "--output_dir", output_config_subdir,
                        "--attention_method", attention_method,
                        "--use_batch_exist",
                        "--use_fused_kernel",
                        "--surround_with_messages",
                        "--rank", str(gpu_id),
                        "--token_budget", str(token_budget),
                        "--run_id", str(current_run_id),
                        "--max_tokens", str(max_tokens),
                        "--start_layer", str(args.start_layer),
                    ]

                    proc = subprocess.Popen(cmd, env=env)
                    active_procs[proc] = {
                        "gpu_id": gpu_id,
                        "run_id": current_run_id,
                    }

                if ((run_counter < total_run and not available_gpus)
                        or (run_counter >= total_run and active_procs)):
                    time.sleep(5)

            if task != "livecodebench":
                get_results_cmd = [
                    sys.executable, "summary_results.py",
                    "--model_name_or_path", model_dir,
                    "--data_name", task,
                    "--limit", str(limit),
                    "--output_dir", output_config_subdir,
                    "--total_run", str(total_run),
                ]

                try:
                    subprocess.run(get_results_cmd, check=True)
                    print(f"Successfully generated results for {param_desc}")
                except subprocess.CalledProcessError as error:
                    print(f"Error generating results: {error}")

            print(f"\nCompleted {param_desc}")

        print(f"\nCompleted task={task}")

    print("\nAll tasks and configurations completed!")
