import argparse
import os
import subprocess
import sys

import submitit
import torch
import yaml


# Check if the experiments are already done, before starting any slurm jobs
# It checks by seeing if an indicator file has been saved
# In this case, "done.json"
def already_done(experiment_path, config_names):
    all_done = True
    for c in config_names:
        full_path = os.path.join(experiment_path, c)
        best_trial_file = os.path.join(full_path, "done.json")
        if not os.path.isfile(best_trial_file):
            all_done = False
            break
    return all_done


# for making each experiment use a different ordering of gpus
def rotate(l, n):
    return l[n:] + l[:n]


# The command (in string form) for running an experiment
def base_command(config_name, experiment_path, cfg):
#     return f"python main.py --dataset {cfg.dataset} --max_epochs {cfg.max_epochs} --optimizer {cfg.optimizer} \
# --important1 {cfg.important1} --important2 {cfg.important2} --config {config_name} --experiment_path {experiment_path}"
    return f"mmf_run config={cfg.config_path} \
             dataset={cfg.dataset} \
             model={cfg.model} \
             run_type={cfg.run_type} \
             env.data_dir={cfg.data_dir} \
             env.save_dir={os.path.join(cfg.root_experiment_folder, cfg.name)}"

# Uses submitit to run experiments using slurm
def exp_launcher(cfg, experiment_path):
    num_gpus = torch.cuda.device_count()
    print("num gpus available in exp_launcher =", num_gpus)

    job_env = submitit.JobEnvironment()
    local_rank = job_env.local_rank
    config_name = cfg.config_names[local_rank]
    gpu_list = list(range(num_gpus))
    use_devices = ",".join([str(x) for x in rotate(gpu_list, local_rank)])
    
    command = base_command(config_name, experiment_path, cfg)

    full_command = f"bash -i ./scripts/{cfg.script_wrapper} {cfg.conda_env} {use_devices}".split(" ")

    ### If using loop_script_wrapper ###
    # full_command = f"bash -i ./scripts/{cfg.script_wrapper} {config_name} {str(cfg.script_wrapper_timeout)} {experiment_path} {cfg.conda_env} {use_devices}".split(
    #     " "
    # )
    
    full_command += [command]
    subprocess.run(full_command)


def main(cfg, slurm_args):
    experiment_path = os.path.join(cfg.root_experiment_folder, cfg.name)

    if already_done(experiment_path, cfg.config_names):
        print("These experiments are already done. Exiting.")
        return

    num_tasks = len(cfg.config_names)
    executor = submitit.AutoExecutor(folder=os.path.join(experiment_path, "slurm_logs"))
    executor.update_parameters(
        timeout_min=0,
        tasks_per_node=num_tasks,
        slurm_additional_parameters=slurm_args,
    )
    job = executor.submit(exp_launcher, cfg, experiment_path)
    jobid = job.job_id
    print(f"running job_id = {jobid}")
    all_jobids_filename = os.path.join(cfg.root_experiment_folder, "all_jobids.txt")
    with open(all_jobids_filename, "a") as fd:
        fd.write(f"{jobid}\n")


if __name__ == "__main__":
    with open("constants.yaml", "r") as f:
        constants = yaml.safe_load(f)

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--script_wrapper_timeout", type=int, default=1200)
    parser.add_argument(
        "--root_experiment_folder",
        type=str,
        default=constants["experiment_folder"],
    )
    parser.add_argument(
        "--dataset_folder", type=str, default=constants["dataset_folder"]
    )
    parser.add_argument("--conda_env", type=str, default=constants["conda_env"])
    parser.add_argument("--config_names", nargs="+", type=str, default=["none"])
    parser.add_argument("--script_wrapper", type=str, default="script_wrapper.sh")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--model", type=str, default="xgen")
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--run_type", type=str, default="train_val")
    # parser.add_argument("--max_epochs", type=int, required=True)
    # parser.add_argument("--optimizer", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    args, unknown_args = parser.parse_known_args()

    slurm_args = {}
    for s in unknown_args:
        if s == "":
            continue
        k, v = s.split("=")
        slurm_args[k.lstrip("--")] = v

    main(args, slurm_args)
