#!/bin/bash

### Dedupliate DeepCAD ###
# Deduplicate repeat CAD B-rep (GDM training)
python ../data_process/deduplicate.py --name cad --data deepcad_parsed --bit 6 --option 'deepcad'
# Deduplicate repeated surface & edge (VAE training)
python ../data_process/deduplicate.py --name facEdge --data deepcad_parsed --list deepcad_data_split_6bit.pkl --bit 6 --type face --option 'deepcad'
python ../data_process/deduplicate.py --name facEdge --data deepcad_parsed --list deepcad_data_split_6bit.pkl --bit 6 --type edge --option 'deepcad'


### Dedupliate ABC ###
# Deduplicate repeat CAD B-rep (GDM training)
python ../data_process/deduplicate.py --name cad --data abc_parsed --bit 6 --option 'abc'
# Deduplicate repeated surface & edge (VAE training)
python ../data_process/deduplicate.py --name facEdge --data abc_parsed --list abc_data_split_6bit.pkl --bit 6 --type face --option 'abc'
python ../data_process/deduplicate.py --name facEdge --data abc_parsed --list abc_data_split_6bit.pkl --bit 6 --type edge --option 'abc'


### Dedupliate Furniture ###
# Deduplicate repeat CAD B-rep (GDM training)
python ../data_process/deduplicate.py --name cad --data furniture_parsed --bit 6 --option 'furniture'
# Deduplicate repeated surface & edge (VAE training)
python ../data_process/deduplicate.py --name facEdge --data furniture_parsed --list furniture_data_split_6bit.pkl --bit 6 --type face --option 'furniture'
python ../data_process/deduplicate.py --name facEdge --data furniture_parsed --list furniture_data_split_6bit.pkl --bit 6 --type edge --option 'furniture'