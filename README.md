# GlobalSfMpy

## Tested Environment
Ubuntu 22.04

## Dependency
* TheiaSfM (Recommand version 0.7.0)
* Ceres (Recommand version 1.14.0)
* pybind11 (Recommand version 2.9.2)
* OpenCV
* (Optional) COLMAP (Recommand version 3.6)

## Demo
### 1DSfM Madrid_Metropolis
```bash
cd script
python sfm_pipeline.py ../flags_1dsfm.yaml
```
### ETH3D facade
First, use ```COLMAP``` extract the feature points and two-view matches. Put the COLMAP results inside the ```datasets/facade/colmap``` folder.
i.e.


```bash
.
└── datasets
    └── facade
        ├── cameras.txt
        ├── colmap
        │   ├── cameras.txt
        │   ├── database.db
        │   ├── images.txt
        │   ├── points3D.txt
        │   └── project.ini
        ├── images
        │   └── *.JPG
        └── images.txt
```
Then, use GobalSfMpy to reconstruct the scene.
```bash
cd script
python read_colmap_database.py --dataset_path ../datasets/facade
python get_covariance_from_colmap.py
python sfm_with_colmap_feature.py
```

The reconstruction is stored in ```output``` folder. The format of the result is the same as what it is in TheiaSfM. The Theia application ```view_reconstruction``` can be used to visualize the result. 
```bash
./view_reconstruction --reconstruction <RESULT_FILE>
```

