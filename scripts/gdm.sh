#!/bin/bash\

python train_gdm.py --data data_process/furniture_parsed \
    --train_list data_process/furniture_data_split_6bit.pkl --option feTopo --gpu 1 \
    --env furniture_gdm_fbTopo --train_epochs 3000 --test_epochs 50 --save_epochs 500 --batch_size 64 \
    --max_face 50 --max_edge 30

python train_gdm.py --data data_process/furniture_parsed \
    --train_list data_process/furniture_data_split_6bit.pkl --option faceGeom --gpu 2 3 \
    --env furniture_gdm_faceGeom --train_epochs 3000 --test_epochs 50 --save_epochs 500 --batch_size 64 \
    --max_face 50 --max_edge 30

python train_gdm.py --data data_process/furniture_parsed \
    --train_list data_process/furniture_data_split_6bit.pkl --option edgeGeom --gpu 2 3 \
    --env furniture_gdm_edgeGeom --train_epochs 1000 --test_epochs 50 --save_epochs 100 --batch_size 32 \
    --max_face 50 --max_edge 30