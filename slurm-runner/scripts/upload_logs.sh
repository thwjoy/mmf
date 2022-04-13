while :
do
    cd "$5" # goes to the slurm-example folder
    # put anything here that needs to be done while in slurm-example
    cd "$1" # goes to the root experiment folder
    echo $PWD
    echo "Removing existing zip"
    rm all_logs.zip
    echo "Zipping files"
    # finds all *.err and *.out files and zips them
    # find -maxdepth 4 -name *.err -o -name *.out | zip -rv all_logs -@
    zip -r all_logs.zip . -x "*.ckpt*"
    echo "Deleting existing files"
    gdrive list -q "'$3' in parents" --no-header --max 0 | cut -d" " -f1 - | xargs -L 1 gdrive delete
    echo "Starting upload $3"
    gdrive upload --parent "$3" all_logs.zip
    echo "Sleeping for $4"
    sleep "$4"
done