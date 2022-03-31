import argparse
import os

def main(args):
    print(args)
    full_path = os.path.join(args.experiment_path, args.config)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    done_path = os.path.join(full_path, "done.json")
    with open(done_path, 'w') as f:
        f.write("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--experiment_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max_epochs", type=int, required=True)
    parser.add_argument("--optimizer", type=str, required=True)
    parser.add_argument("--important1", type=str, required=True)
    parser.add_argument("--important2", type=str, required=True)
    args = parser.parse_args()
    main(args)
