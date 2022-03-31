import argparse
import os
import subprocess

import yaml


def main(cfg):
    curr_dir = os.path.abspath(os.path.dirname(__file__))

    command = "bash -i ./scripts/upload_logs.sh {0} {1} {2} {3} {4}".format(
        cfg.root_experiment_folder,
        cfg.slurm_folder,
        cfg.gdrive_folder,
        cfg.sleep_time,
        curr_dir,
    )
    subprocess.run(command.split(" "))


if __name__ == "__main__":
    with open("constants.yaml", "r") as f:
        constants = yaml.safe_load(f)

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--root_experiment_folder",
        type=str,
        default=constants["experiment_folder"],
    )
    parser.add_argument(
        "--slurm_folder",
        type=str,
        default=constants["slurm_folder"],
    )
    parser.add_argument("--gdrive_folder", type=str, default="19RHtq1y2tR7tmKdV9oHeocfLPeCcrGkW")
    parser.add_argument("--sleep_time", type=str, default="240m")
    args = parser.parse_args()
    main(args)
