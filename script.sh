#!/bin/sh
#An example for gpu job.
#SBATCH -J DiffBrep
#SBATCH -o job-%j.log
#SBATCH -e job-%j.err
#SBATCH --qos=gpu_8a100
#SBATCH -N 1 -n 1 -p GPU-8A100 --gres=gpu:1

python train_gdm.py
