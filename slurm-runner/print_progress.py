import argparse
import glob
import json
import os

import yaml


def count_exp_folders(contents):
    num_folders = 0
    for x in contents:
        if os.path.isdir(x) and os.path.basename(x).isdigit():
            num_folders += 1
    return num_folders


def print_folder_progress(cfg, exps):
    output = {}
    for e in exps:
        e = os.path.normpath(e)
        contents = glob.glob(f"{e}/*")
        num_folders = count_exp_folders(contents)
        output[os.path.basename(e)] = num_folders
    return output


def main(cfg):
    experiment_paths = sorted(glob.glob(f"{cfg.root_experiment_folder}/*"))
    all_folders = {}
    for p in experiment_paths:
        if os.path.isdir(p):
            exps = sorted(glob.glob(f"{p}/*"))
            all_folders[os.path.basename(p)] = print_folder_progress(cfg, exps)

    out_string = json.dumps(all_folders, indent=4)
    if cfg.save_to_file:
        with open(cfg.save_to_file, "w") as f:
            f.write(out_string)
    else:
        print(out_string)


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
        "--save_to_file",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    main(args)
