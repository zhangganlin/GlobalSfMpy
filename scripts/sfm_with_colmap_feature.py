import sys
import yaml
import os
sys.path.append('../build')
import GlobalSfMpy as sfm
from loss_functions import *
from sfm_pipeline import *


flagfile = "../flags_1dsfm.yaml"
f = open(flagfile,"r")
config = yaml.safe_load(f)
glog_directory = config['glog_directory']
glog_verbose = config['v']
log_to_stderr = config['log_to_stderr']
sfm.InitGlog(glog_verbose,log_to_stderr,glog_directory)

dataset_name = "facade"
# dataset_path = "../datasets/"+dataset_name
dataset_path = "/home/zhangganlin/Desktop/CVG/final_version_globalsfmpy/datasets/"+dataset_name

colmap_path = dataset_path+"/colmap/images.txt"
output_reconstruction = "../output/"+dataset_name

rotation_error_type = sfm.RotationErrorType.ANGLE_AXIS_COVARIANCE
position_error_type = sfm.PositionErrorType.BASELINE

reconstruction = sfm_pipeline(flagfile,dataset_path,
                                            MAGSACWeightBasedLoss(0.02),HuberLoss(0.1),
                                            rotation_error_type,position_error_type,
                                            onlyRotationAvg=False)
if os.path.exists(output_reconstruction):
    os.remove(output_reconstruction)
sfm.WriteReconstruction(reconstruction, output_reconstruction)

sfm.StopGlog()



