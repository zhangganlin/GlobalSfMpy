# GlobalSfMpy

This repo is the implementation of the paper "[Revisiting Rotation Averaging: Uncertainties and Robust Losses](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Revisiting_Rotation_Averaging_Uncertainties_and_Robust_Losses_CVPR_2023_paper.pdf)".

If you find our code or paper useful, please cite
```bibtex
@inproceedings{zhang2023revisiting,
  title={Revisiting Rotation Averaging: Uncertainties and Robust Losses},
  author={Zhang, Ganlin and Larsson, Viktor and Barath, Daniel},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={17215--17224},
  year={2023}
}
```

## Tested Environment
Ubuntu 22.04

## Dependency
* TheiaSfM (Recommand version 0.7.0)
* Ceres (Recommand version 1.14.0)
* pybind11 (Recommand version 2.9.2)
* OpenCV
* (Optional) COLMAP (Recommand version 3.6)

### Ceres
More details can be found on the [Ceres official website](http://ceres-solver.org/installation.html)
```bash
# install dependencies of Ceres Solver
sudo apt-get install cmake libgoogle-glog-dev libgflags-dev libatlas-base-dev libeigen3-dev libsuitesparse-dev

# download and install Ceres Solver
wget https://github.com/ceres-solver/ceres-solver/archive/refs/tags/1.14.0.tar.gz
tar zxf 1.14.0.tar.gz
mkdir ceres-bin
cd ceres-bin
cmake ../ceres-solver-1.14.0
make -j3
make install
```
### TheiaSfM
More details can be found on the [TheiaSfM official website](http://theia-sfm.org/building.html)
```bash
# install dependencies of Theia
sudo apt-get install libopenimageio-dev librocksdb-dev libatlas-base-dev rapidjson-dev libgtest-dev libyaml-cpp-dev
cd thirdparty/TheiaSfM
mkdir build && cd build
cmake ..
make -j4
sudo make install
```

### PyBind11
```bash
git clone git@github.com:pybind/pybind11.git
cd pybind11 && mkdir build && cd build
cmake .. && make -j4
sudo make install
```

### OpenCV
More details can be found on the [OpenCV official website](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html)
```bash
# Install minimal prerequisites (Ubuntu 18.04 as reference)
sudo apt update && sudo apt install -y cmake g++ wget unzip
# Download and unpack sources
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.x.zip
unzip opencv.zip
unzip opencv_contrib.zip
# Create build directory and switch into it
mkdir -p build && cd build
# Configure
cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.x/modules ../opencv-4.x
# Build
cmake --build .
```

## Install
```bash
mkdir build && cd build
cmake ..
make
```


## Demo
### 1DSfM Madrid_Metropolis
```bash
cd script
python sfm_pipeline.py ../flags_1dsfm.yaml
```
### ETH3D terrace
First, use ```COLMAP``` extract the feature points and two-view matches. When extracting the feature, please use the ```PINHOLE``` camera model. Here a bash script is provided to run COLMAP for extracting the two-view information:

Download the datasets from [here](https://www.eth3d.net/data/terrace_dslr_undistorted.7z) and put the images inside the ```datasets/terrace/images``` folder, i.e.
```bash
.
└── datasets
    └── terrace
        └── images
            └── *.JPG
```

First change [the colmap executable path](https://github.com/zhangganlin/GlobalSfMpy/blob/main/scripts/colmapFeatureMatching.sh#L1) in the script, then
```bash
cd script
bash colmapFeatureMatching.sh ../datasets/terrace
```
After running COLMAP, the sturcture of the dataset folder should be as following:

```bash
.
└── datasets
    └── terrace
        ├── cameras.txt
        ├── colmap
        │   ├── 0
        │   │   ├── cameras.bin
        │   │   ├── images.bin
        │   │   ├── points3D.bin
        │   │   └── project.ini
        │   └── database.db
        ├── covariance_rot.txt
        ├── images
        │   └── *.JPG
        ├── images.txt
        ├── points3D.txt
        └── two_views.txt

```
Then, use GobalSfMpy to reconstruct the scene. The following commands are run inside ```script``` folder.
```bash
python read_colmap_database.py --dataset_path ../datasets/terrace
python get_covariance_from_colmap.py --dataset_path ../datasets/terrace
python sfm_with_colmap_feature.py --dataset_path ../datasets/terrace
```

The reconstruction is stored in ```output``` folder. The format of the result is the same as what it is in TheiaSfM. The Theia application ```view_reconstruction``` can be used to visualize the result. 
```bash
./view_reconstruction --reconstruction <RESULT_FILE>
```
E.g. For ```1DSfM Madrid_Metropolis```
```bash
./thirdparty/TheiaSfM/build/bin/view_reconstruction --reconstruction output/Madrid_Metropolis
```
Example visualization of the output of ```1DSfM Madrid_Metropolis```:
![demo](https://github.com/zhangganlin/GlobalSfMpy/assets/32034109/750de1f2-36b5-485c-982b-2e06fce6cffb)

Example visualization of the output of ```ETH3D terrace```
![demo](https://github.com/zhangganlin/GlobalSfMpy/assets/32034109/a8e58390-2ffe-44b4-94b6-614ef7ec2b7e)
