
## 1. Create conda env and install basic packages
```bash
conda env create -f env.yml
```
## 2. Install sdf
```bash
conda activate dax
pip install git+https://github.com/fogleman/sdf.git -c constraints.txt 
```
## 3. Install JAX

Here we use JAX version 0.4.13, and we suggest using CUDA12.

```bash
pip install jaxlib==0.4.13+cuda12.cudnn89 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install jax==0.4.13
```

Run a simple python file to test whether JAX is installed appropriately on GPU(s).

```bash
python test_jax.py
```
You may get results like this:
```
JAX version: 0.4.13
JAX devices: [gpu(id=0)]
Device types: ['NVIDIA GeForce RTX 4090']
GPU computation test: [5. 7. 9.]
```

## 4. Install the DaxBench package
```bash
pip install . --no-deps
```

## 5. Test
```bash
python daxbench/core/envs/fold_cloth1_env.py
```

## Trouble Shooting

### 1.ImportError: Library "GLU" not found

```bash
sudo apt-get update
sudo apt-get install -y \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libglu1-mesa \
    libglu1-mesa-dev \
    libglvnd-dev \
    libx11-dev \
    libxcursor-dev \
    libxinerama-dev \
    libxrandr-dev \
    mesa-common-dev \
    pkg-config
```

### 2.FAILED_PRECONDITION: DNN library initialization failed / cudaGetErrorName symbol not found
That is may because you are using CUDA11 or ealier version. In our test, these versions are not compatible with DaxBench. Therefore, uninstall the related CUDA drivers and NVIDIA drivers, and reinstall the CUDA12 version. 
Take CUDA11.5 for example:

```bash
sudo apt-get --purge remove "*cuda-11-5*"
sudo apt-get --purge remove "*cublas-11-5*"
sudo apt-get --purge remove "*cufft-11-5*"
sudo apt-get --purge remove "*curand-11-5*"
sudo apt-get --purge remove "*cusolver-11-5*"
sudo apt-get --purge remove "*cusparse-11-5*"
sudo apt-get --purge remove "*npp-11-5*"
sudo apt-get --purge remove "*nvjpeg-11-5*"

sudo apt-get --purge remove "*nvidia*"

sudo apt-get autoremove
sudo apt-get autoclean

#remove PATH and LD_LIBRARY_PATH in bashrc

sudo reboot

#install new drivers (take ubuntu22.04 for example)
#reference: https://developer.nvidia.com/cuda-12-4-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda-repo-ubuntu2204-12-4-local_12.4.0-550.54.14-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-4-local_12.4.0-550.54.14-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4
sudo apt-get install -y nvidia-driver-550-open
sudo apt-get install -y cuda-drivers-550


vim ~/.bashrc
export PATH=/usr/local/cuda-12.4/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
source ~/.bashrc
sudo reboot
```


