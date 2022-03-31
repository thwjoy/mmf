import subprocess

with open("constants.yaml", "r") as f:
    content = f.readlines()

conda_env = [
    c.strip().split(":")[1].lstrip(" ") for c in content if c.startswith("conda_env")
][0]

command = "bash -i ./scripts/create_env.sh {0}".format(conda_env)
subprocess.run(command.split(" "))
