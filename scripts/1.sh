#!/bin/bash

# Base directory containing the groups
BASE_DIR="/home/jing/PythonProjects/BrepGDM/comparison/datas/deepcad/reference_test"

# Loop through 0000 to 0099
for i in $(seq -f "%04g" 1 99); do
    IN_DIR="$BASE_DIR/$i"
    OUT_DIR="$BASE_DIR/$i"

    # Run the Python script with the specified directories
    python -m comparison.sample_points --in_dir "$IN_DIR" --out_dir "$OUT_DIR"
done
