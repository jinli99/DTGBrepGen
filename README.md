# B-rep Generation Based on Decoupling Topology and Geometry

### Dependencies

Install PyTorch and other dependencies:
```
conda create --name BrepGDM python=3.10.13 -y
conda activate BrepGDM

pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
pip install chamferdist
```

If `chamferdist` fails to install here are a few options to try:

- If there is a CUDA version mismatch error, then try setting the `CUDA_HOME` environment variable to point to CUDA installation folder. The CUDA version of this folder must match with PyTorch's version i.e. 11.8.

- Try [building from source](https://github.com/krrish94/chamferdist?tab=readme-ov-file#building-from-source).

Install OCCWL following the instruction [here](https://github.com/AutodeskAILab/occwl).
```
conda install -c lambouj -c conda-forge occwl
```

If conda is stuck in "Solving environment..." there are two options to try:

- Try using `mamba` as suggested in occwl's README.

- Install pythonOCC: https://github.com/tpaviot/pythonocc-core?tab=readme-ov-file#install-with-conda and occwl manually: `pip install git+https://github.com/AutodeskAILab/occwl`.

