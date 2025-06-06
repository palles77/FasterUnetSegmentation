# MUSOCS
Multi-Scale U-Net Segmentation Optimized by CNN-Based Quality Scoring

# Cloning

You need GIT-LFS installed in order to clone this repository

# Preparing the environment

1. Install Anaconda with the latest version of Python (in our case it was Python 3.8.20)
2. Create Anaconda environment
```
conda create -n unet-segmentation-article python=3.11.5
conda activate unet-segmentation-article
```
3. Make sure your Python version is 3.11 or newer
```
python --version
```
4.1. Install relevant requirements from the current directory on Windows:
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r src/requirements.txt
```
4.2. Install relevant requirements from the current directory on Linux:
```
bash src/pytorch_linux.sh
pip install -r src/requirements.txt
```

# Downloading the data

Refer to [data/README.md](data/README.md)

# More 

Detailed understanding can be revised from [.vscode/launch.json](.vscode/launch.json)

# Last update
2025/05/29
