# slurm-example

## The goal
The goal is to put all the experiment commands in a single .sh file:
```
python launch_multiple.py --config dataset1_configs/config1 ...
python launch_multiple.py --config dataset1_configs/config2 ...
...
```
Then the remote user just has to:
1. run the .sh file
2. run ```python upload_logs.py ...```


## Overview

The code in ```launch_one.py``` and in ```slurm_configs``` assumes you're running experiments with the following folder structure:
```root_experiment_folder/experiment_group/experiment_name```

If you look at [```slurm_configs/dataset1_configs/config1.yaml```](https://github.com/KevinMusgrave/slurm-example/blob/main/slurm_configs/dataset1_configs/config1.yaml):

 - the "experiment group" is defined by the ```important1``` and ```important2``` flags. (Only because this is how the folders are defined in ```launch_one.py```. There is nothing special about the flag names, it's just an example.)
 - the experiment name is defined by the config name. (It's assumed that you have config files for running individual experiments.)

Running this slurm config file will create the following folders:
```
-root_experiment_folder
--dog_cat
---alpha
---beta
---gamma
---omega
---zeta
--zebra_giraffe
---alpha
---beta
---gamma
---omega
---zeta
```

It will run:
- ```dog_cat/(alpha, beta, gamma)``` as 1 slurm job
- ```dog_cat/(omega, zeta)``` as 1 slurm job
- ```zebra_giraffe/(alpha, beta, gamma)``` as 1 slurm job
- ```zebra_giraffe/(omega, zeta)``` as 1 slurm job

To specify number of gpus, cpus etc, append the appropriate slurm flag when you call ```launch_multiple.py```. See below for details



### [```constants.yaml```](https://github.com/KevinMusgrave/slurm-example/blob/main/constants.yaml)
Define the root experiment folder, conda environment name etc.

Try not to push any changes to this file to your git repo. Otherwise, the remote user will have merge conflicts when they pull.

### [```install_env.py```](https://github.com/KevinMusgrave/slurm-example/blob/main/install_env.py)
Installs packages into the conda environment specified in ```constants.yaml```. This script calls [```scripts/create_env.sh```](https://github.com/KevinMusgrave/slurm-example/blob/main/scripts/create_env.sh). Modify the bash script and [```requirements.txt```](https://github.com/KevinMusgrave/slurm-example/blob/main/requirements.txt) to control the details. You might also want to use a [conda environment file](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#exporting-the-environment-yml-file).

It's best to include ```submitit``` and ```PyYAML``` in your environment, since those are required to run ```launch_multiple.py```. That way, if the remote user activates your environment before running ```launch_multiple.py```, there won't be any problems.


### [```launch_multiple.py```](https://github.com/KevinMusgrave/slurm-example/blob/main/launch_multiple.py)
Reads in a yaml file (e.g. config1.yaml) that contains multiple commands, and sends each one to ```launch_one.py```.

### [```scripts/example_jobs.sh```](https://github.com/KevinMusgrave/slurm-example/blob/main/scripts/example_jobs.sh)
This runs some experiments using ```launch_multiple.py```. See this file for an example of slurm flags. When you append slurm flags, make sure they start with ```--``` and that they have ```=``` between the flag and the value. This is important for the parsing done in ```launch_one.py```.

### [```launch_one.py```](https://github.com/KevinMusgrave/slurm-example/blob/main/launch_one.py)
Launches the slurm jobs. You should modify the ```base_command``` function to call your main experiment script. Also check the argparse arguments at the bottom of the file. Any unknown arguments that are passed in will be assumed to be for slurm. Every slurm jobid is saved in ```<root_experiment_folder>/all_jobids.txt```

### [```kill_all.py```](https://github.com/KevinMusgrave/slurm-example/blob/main/kill_all.py)
This will kill all slurm jobids in ```all_jobids.txt```

### [```delete_experiment.py```](https://github.com/KevinMusgrave/slurm-example/blob/main/delete_experiment.py)
Will delete all sub experiments that match the name. For example if the experiment directories are:

```
-root_experiment_folder
--dog_cat
---alpha
---beta
---gamma
--zebra_giraffe
---alpha
---beta
---gamma
```
Then ```python delete.py --name alpha --delete``` will delete ```dog_cat/alpha``` and ```zebra_giraffe/alpha```.
You can get a preview of what it will delete by omitting the ```--delete``` flag.

### [```print_progress.py```](https://github.com/KevinMusgrave/slurm-example/blob/main/print_progress.py)
Will print the number of subfolders in each experiment folder. This might not be applicable to your setup.

### [```upload_logs.py```](https://github.com/KevinMusgrave/slurm-example/blob/main/upload_logs.py)
### !!!WARNING: before every upload, this will delete all existing files in the target Google Drive folder!!!
Uploads files to google drive. The gdrive folder address is the random characters following ```folders/``` in the web address of a google drive folder.

```python upload_logs.py --gdrive_folder folder_address_goes_here```

The python script calls ```scripts/upload_logs.sh```

### [```scripts/upload_logs.sh```](https://github.com/KevinMusgrave/slurm-example/blob/main/scripts/upload_logs.sh)
Called by ```upload_logs.py```. You can customize the "find" command to change what gets zipped and uploaded. Before uploading, it will delete everything in the target google drive folder. After uploading, it sleeps (default sleep time = 4 hours).

### [```scripts/script_wrapper.sh```](https://github.com/KevinMusgrave/slurm-example/blob/main/scripts/script_wrapper.sh)
Called by ```launch_one.py```. This runs inside a slurm job. It activates the conda environment, and exports ```CUDA_VISIBLE_DEVICES```, before running the experiment.

### [```scripts/loop_script_wrapper.sh```](https://github.com/KevinMusgrave/slurm-example/blob/main/scripts/loop_script_wrapper.sh)
This will run the experiment, then check the experiment folder occassionally to see if the folder is getting updated. If the folder is not updated, this script will kill the experiment and restart it. I find this useful because PyTorch dataloaders sometimes hang. If you want to use this instead of ```script_wrapper.sh```, you need to make some changes to the ```exp_launcher``` function in ```launch_one.py```.

### [```scripts/process_checker.sh```](https://github.com/KevinMusgrave/slurm-example/blob/main/scripts/process_checker.sh)
Used by ```scripts/loop_script_wrapper.sh```.

