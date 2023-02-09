import sys
import yaml
import os
import shutil
sys.path.append('../build')
import GlobalSfMpy as sfm
from loss_functions import *
    
def sfm_with_1dsfm_dataset(flagfile,dataset_path,
                           loss_func_rotation,
                           loss_func_position,
                           rotation_error_type=sfm.RotationErrorType.ANGLE_AXIS,
                           position_error_type=sfm.PositionErrorType.BASELINE,
                           onlyRotationAvg=False):
    return sfm_pipeline(flagfile,dataset_path,
                           loss_func_rotation,
                           loss_func_position,
                           rotation_error_type,
                           position_error_type,
                           onlyRotationAvg,
                           use1DSfM = True)

def sfm_pipeline(flagfile,dataset_path,
                           loss_func_rotation,
                           loss_func_position,
                           rotation_error_type=sfm.RotationErrorType.ANGLE_AXIS,
                           position_error_type=sfm.PositionErrorType.BASELINE,
                           onlyRotationAvg=False,
                           use1DSfM = False):
    if(use1DSfM):
        options = sfm.ReconstructionBuilderOptions()
        sfm.load_1DSFM_config(flagfile,options)
        reconstruction = sfm.Reconstruction()
        view_graph = sfm.ViewGraph()
        rot_covariances = sfm.MapEdgesCovariance()
        sfm.Read1DSFM(dataset_path,reconstruction,view_graph,rot_covariances)
        reconstruction_builder = sfm.ReconstructionBuilder(options,reconstruction,view_graph)
    else:
        database = sfm.FeaturesAndMatchesDatabase(
            dataset_path+"/database")
        options = sfm.ReconstructionBuilderOptions()
        sfm.load_1DSFM_config(flagfile,options)
        rot_covariances = sfm.MapEdgesCovariance()
        
        sfm.ReadCovariance(dataset_path,rot_covariances)
        reconstruction_builder = sfm.ReconstructionBuilder(options,database)
        sfm.AddColmapMatchesToReconstructionBuilder(dataset_path+"/two_views.txt",dataset_path+"/images/*.JPG",reconstruction_builder)
        
    
    reconstruction_builder.CheckView()
    view_graph = reconstruction_builder.get_view_graph()
    reconstruction = reconstruction_builder.get_reconstruction()
    reconstruction_estimator = sfm.GlobalReconstructionEstimator(options.reconstruction_estimator_options)
        
    # Step 1. Filter the initial view graph and remove any bad two view geometries.
    # Step 2. Calibrate any uncalibrated cameras.
    reconstruction_estimator.FilterInitialViewGraphAndCalibrateCameras(view_graph,reconstruction)

    # Step 3. Estimate global rotations.
    assert(reconstruction_estimator.EstimateGlobalRotationsUncertainty(loss_func_rotation,
                                                            rot_covariances,
                                                            rotation_error_type))
    
    if(onlyRotationAvg):
        # Set the poses in the reconstruction object.
        sfm.SetOrientations(
            reconstruction_estimator.orientations,
            reconstruction
        )
        return reconstruction
    
    
    # Step 4. Filter bad rotations.
    reconstruction_estimator.FilterRotations()

    # Step 5. Optimize relative translations.
    reconstruction_estimator.OptimizePairwiseTranslations()

    # Step 6. Filter bad relative translations.
    reconstruction_estimator.FilterRelativeTranslation()

    # Step 7. Estimate global positions.
    assert(reconstruction_estimator.EstimatePosition(loss_func_position,position_error_type))

    # Set the poses in the reconstruction object.
    sfm.SetReconstructionFromEstimatedPoses(
        reconstruction_estimator.orientations,
        reconstruction_estimator.positions,
        reconstruction
    )
    
    # Always triangulate once, then retriangulate and remove outliers depending
    # on the reconstruciton estimator options.
    for i in range(1+reconstruction_estimator.options.num_retriangulation_iterations):
        
        # Step 8. Triangulate features.
        reconstruction_estimator.EstimateStructure()
        sfm.SetUnderconstrainedAsUnestimated(reconstruction)
        
        # Do a single step of bundle adjustment where only the camera positions and
        # 3D points are refined. This is only done for the very first bundle
        # adjustment iteration.
        if i == 0 and reconstruction_estimator.options.refine_camera_positions_and_points_after_position_estimation:
            reconstruction_estimator.BundleAdjustCameraPositionsAndPoints()
        
        # Step 9. Bundle Adjustment.
        reconstruction_estimator.BundleAdjustmentAndRemoveOutlierPoints()
        
    if (not use1DSfM) and os.path.exists(dataset_path+"/database"):
        shutil.rmtree(dataset_path+"/database") 
    
    return reconstruction

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: sfm_with_1dsfm_dataset.py <flagfile>')
        sys.exit(1)
    flagfile = sys.argv[1]
    
    f = open(flagfile,"r")
    config = yaml.safe_load(f)
    dataset_path = config['1dsfm_dataset_directory']
    output_reconstruction = config['output_reconstruction']
    glog_directory = config['glog_directory']
    glog_verbose = config['v']
    log_to_stderr = config['log_to_stderr']
        
    sfm.InitGlog(glog_verbose,log_to_stderr,glog_directory)
    
    position_error_type = sfm.PositionErrorType.BASELINE    
    rotation_error_type = sfm.RotationErrorType.ANGLE_AXIS_COVARIANCE
    
    if not os.path.exists(dataset_path+"/covariance_rot.txt"):
        sfm.CalcCovariance(dataset_path)
    
    loss_func = MAGSACWeightBasedLoss(0.02)
        
    reconstruction = sfm_with_1dsfm_dataset(flagfile,dataset_path,
                                            loss_func,HuberLoss(0.1),
                                            rotation_error_type,position_error_type,
                                            onlyRotationAvg=False)
    
    if os.path.exists(output_reconstruction):
        os.remove(output_reconstruction)
    sfm.WriteReconstruction(reconstruction, output_reconstruction)
    
    sfm.StopGlog()
