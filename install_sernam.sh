#!/bin/bash
conda create -n mmf python=3.7
conda activate mmf
pip install --editable .
pip install submitit
pip install future
pip install pyyaml