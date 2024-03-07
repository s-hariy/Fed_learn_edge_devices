# FEDERATED LEARNING WITH JETSON AND LOCAL PC CLIENTS

# SERVER SETUP
## Hardware Requirement
1. Intel or AMD CPU (min 4 Cores)
2. GPU (>= 8GB)
3. 32 GB RAM
## Software Requirement
1. Ubuntu >=18.04
2. Python >= 3.8
3. Install server_requirements.txt

# JETSON XAVIER SETUP

## INSTALL JETPACK
1. Install Nvidia SDK manager: https://docs.nvidia.com/jetson/jetpack/install-jetpack/index.html#sdk-manager
2. Create a NVIDIA developer account.
3. Flash Jetson Xavier with Jetpack 5.1.2 using SDK Manager. Follow the tutorial in the link: https://docs.nvidia.com/sdk-manager/install-with-sdkm-jetson/index.html


## INSTALL PYTORCH
1. Follow the instructions in the link: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048 OR Follow the below steps.
2. First check the compatability of the Jetpack version with pyTorch version: https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html#pytorch-jetson-rel
3. Example: Install pytorch 2.0.0 and torchvision 0.15.1 for Jetpack 5.1.2.
4. Install system packages for Pytorch.
```
sudo apt-get -y update; 
sudo apt-get -y python3-pip 
sudo apt-get libopenblas-dev;
```
5. Install using the Wheel File.
```
export TORCH_INSTALL=https://developer.download.nvidia.cn/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
```
`v511` is the jetpack version 5.1.1. Note we can install 5.1.1 version pytorch for 5.1.2 without any adverse effects, vice versa not possible.

6. Install Pytorch.
```
python3 -m pip install --upgrade pip; python3 -m pip install numpy python3 -m pip install --no-cache $TORCH_INSTALL
```

Note: Step 6 can be done in virtual environment also.

## INSTALL TORCHVISION
1. Follow the instructions in the link: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048 OR Follow the below steps.

2. Install system packages for torchvision.
```
sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev
```
3. Select the correct torchvision version for the pytorch and clone the repo.
```
git clone --branch <version> https://github.com/pytorch/vision torchvision
```
4. Build torchvision from the source.
```
cd torchvision
export BUILD_VERSION=0.x.0  # where 0.x.0 is the torchvision version  
python3 setup.py install --user
cd ../  
python3 -m pip install pillow
```
## INSTALL TENSORFLOW
1. Install system packages
```
$ sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
```
2. Install and upgrade pip and python setuptools
```
sudo apt-get install python3-pip
sudo python3 -m pip install --upgrade pip
sudo pip3 install -U testresources setuptools==65.5.0
```
3. Install the python dependent packages.

```
sudo pip3 install -U numpy==1.22 future==0.18.2 mock==3.0.5 keras_preprocessing==1.1.2 keras_applications==1.0.8 gast==0.4.0 protobuf pybind11 cython pkgconfig packaging h5py==3.7.0
```
4. Install Tensorflow
```
sudo pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v512 tensorflow==2.12.0+nv23.06
```

Note: Remove sudo and follow the same steps to install python packages in virtual environment.
```
pip3 install -U numpy grpcio absl-py py-cpuinfo psutil portpicker six mock requests gast h5py astor termcolor protobuf keras-applications keras-preprocessing wrapt google-pasta

pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v51 tensorflow==$TF_VERSION+nv$NV_VERSION
```

Note: grpcio - building takes a long time in this setup.

5. Install python packages for federated learning
```
pip3 install -r jetson_requirements.txt
```
6. Download and extract data: https://drive.google.com/file/d/15uNMnR3IoZaQYPoIAHJJeEfgACM5RBH3/view?usp=sharing

# LOCAL PC/LAPTOP SETUP
## Hardware Requirement
1. Intel or AMD CPU (min 4 Cores)
2. GPU (>= 8GB)
3. 32 GB RAM
## Software Requirement
1. Ubuntu >=18.04
2. Python >= 3.8
3. Install client_requirements.txt
4. Download and extract data: https://drive.google.com/file/d/15uNMnR3IoZaQYPoIAHJJeEfgACM5RBH3/view?usp=sharing

# RUNNING FEDERATED LEARNING

## Server
Set firewall permission to allow incoming traffic to port 8080
```bash
sudo iptables -A INPUT -p tcp --dport 8080 -j ACCEPT\

sudo ufw allow 8080/tcp
```
```bash
# Launch your server.
# Will wait for at least 2 clients to be connected, then will train for 3 FL rounds
# The command below will sample all clients connected (since sample_fraction=1.0)
python server.py --rounds 3 --min_num_clients 2 --sample_fraction 1.0
```
Example:
```bash
(flwr) srihari@fmlpc:~/git/embedded-devices$ python3 server.py -h
usage: server.py [-h] [--server_address SERVER_ADDRESS] [--rounds ROUNDS]
                 [--sample_fraction SAMPLE_FRACTION]
                 [--min_num_clients MIN_NUM_CLIENTS]

Embedded devices

options:
  -h, --help            show this help message and exit
  --server_address SERVER_ADDRESS
                        gRPC server address (deafault '0.0.0.0:8080')
  --rounds ROUNDS       Number of rounds of federated learning (default: 5)
  --sample_fraction SAMPLE_FRACTION
                        Fraction of available clients used for fit/evaluate
                        (default: 1.0)
  --min_num_clients MIN_NUM_CLIENTS
                        Minimum number of available clients required for
                        sampling (default: 2)

```
## Client (any device)

```bash
# Run the default FedISIC dataset
python3 client.py --cid=<CLIENT_ID> --server_address=<SERVER_ADDRESS>

```
Example:
```bash
(flwr) srihari@fmlpc:~/git/embedded-devices$ python3 client.py -h
usage: client.py [-h] [--server_address SERVER_ADDRESS] --cid CID
                 [--data_path DATA_PATH]

Embedded devices

options:
  -h, --help            show this help message and exit
  --server_address SERVER_ADDRESS
                        gRPC server address (default '0.0.0.0:8080')
  --cid CID             Client id. Should be an integer between 0 and
                        NUM_CLIENTS
  --data_path DATA_PATH
                        absolute path to data

```

Default centres: 
```
    centers = [1,2] if args.cid == 0 else [3,4] if args.cid == 1 else [0,5]
```
Modify Line 158 in client.py to set custom centers. Note there are totally 6 centers (0 to 5).

NUM_CLIENTS default value: 50. Change this if the clients exceed 50 otherwise retain this value.
