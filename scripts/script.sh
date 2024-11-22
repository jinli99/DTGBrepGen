#!/bin/bash\

# Train Furniture
python -m topology.train_topo --name furniture --batch_size 16 --option faceEdge --train_epochs 2000
python -m topology.train_topo --name furniture --batch_size 16 --option edgeVert --train_epochs 1000
python -m geometry.train_geom --name furniture --batch_size 512 --option faceBbox --train_epochs 3000
python -m geometry.train_geom --name furniture --batch_size 512 --option vertGeom --train_epochs 3000
python -m geometry.train_geom --name furniture --batch_size 512 --option edgeGeom --train_epochs 3000

# Train DeepCAD
python -m topology.train_topo --name deepcad --batch_size 512 --option faceEdge --train_epochs 2000
python -m topology.train_topo --name deepcad --batch_size 512 --option edgeVert --train_epochs 1000
python -m geometry.train_geom --name deepcad --batch_size 512 --option faceBbox --train_epochs 3000
python -m geometry.train_geom --name deepcad --batch_size 512 --option vertGeom --train_epochs 3000
python -m geometry.train_geom --name deepcad --batch_size 512 --option edgeGeom --train_epochs 3000

# Train ABC
python -m topology.train_topo --name abc --batch_size 512 --option faceEdge --train_epochs 1000
python -m topology.train_topo --name abc --batch_size 512 --option edgeVert --train_epochs 500 --save_epochs 100
python -m geometry.train_geom --name abc --batch_size 512 --option faceBbox --train_epochs 2000
python -m geometry.train_geom --name abc --batch_size 512 --option vertGeom --train_epochs 2000
python -m geometry.train_geom --name abc --batch_size 512 --option edgeGeom --train_epochs 2000