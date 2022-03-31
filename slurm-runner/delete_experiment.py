import argparse
import glob
import os
import shutil

import yaml


def main(cfg):
    experiment_paths = sorted(glob.glob(f"{cfg.root_experiment_folder}/*"))
    all_folders = {}
    for p in experiment_paths:
        if os.path.isdir(p):
            exps = glob.glob(f"{p}/*")
            for e in exps:
                basename = os.path.basename(e)
                if basename == cfg.name:
                    print(e)
                    if cfg.delete:
                        shutil.rmtree(e)


if __name__ == "__main__":
    with open("constants.yaml", "r") as f:
        constants = yaml.safe_load(f)

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--root_experiment_folder",
        type=str,
        default=constants["experiment_folder"],
    )
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--delete", action="store_true")
    args = parser.parse_args()
    main(args)
