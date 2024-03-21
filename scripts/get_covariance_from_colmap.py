import sys
import yaml
import os
sys.path.append('../build')
import GlobalSfMpy as sfm

dataset_name = "facade"
flagfile = "../flags_1dsfm.yaml"
dataset_path = "../datasets/"+dataset_name

if os.path.exists(dataset_path+"/covariance_rot.txt"):
    print("Covariance already exists!")
    exit()
    

f = open(flagfile,"r")
config = yaml.safe_load(f)
glog_directory = config['glog_directory']
glog_verbose = config['v']
log_to_stderr = False
    
sfm.InitGlog(glog_verbose,log_to_stderr,glog_directory)

database = sfm.FeaturesAndMatchesDatabase(
    dataset_path+"/database")
options = sfm.ReconstructionBuilderOptions()
sfm.load_1DSFM_config(flagfile,options)
reconstruction_builder = sfm.ReconstructionBuilder(options,database)
sfm.AddColmapMatchesToReconstructionBuilder(dataset_path+"/two_views.txt",dataset_path+"/images/*.JPG",reconstruction_builder)
        

reconstruction_builder.CheckView()
view_graph = reconstruction_builder.get_view_graph()
reconstruction = reconstruction_builder.get_reconstruction()

sfm.store_covariance_rot(dataset_path,reconstruction,view_graph)


sfm.StopGlog()
