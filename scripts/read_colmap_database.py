from colmap_database import *
import os
import argparse
from scipy.spatial.transform import Rotation
import cv2 as cv
import poselib


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", default="../datasets/facade")
args = parser.parse_args()

dataset_path = args.dataset_path


class Pose:
    def __init__(self,qw,qx,qy,qz,tx,ty,tz):
        self.qw = qw
        self.qx = qx
        self.qy = qy
        self.qz = qz
        self.tx = tx
        self.ty = ty
        self.tz = tz
    def rotation_matrix(self):
        R = Rotation.from_quat([self.qx,self.qy,self.qz,self.qw])
        return R.as_matrix()
    def translation(self):
        return np.array([[self.tx],[self.ty],[self.tz]])
        

f_camera = open(dataset_path+"/cameras.txt")
cameras = {}
lines = f_camera.readlines()
for line in lines[3:]:
    words = line.split()
    camera_id = np.uint32(words[0])
    camera_width = np.int32(words[2])
    camera_height = np.int32(words[3])
    fx = np.float64(words[4])
    fy = np.float64(words[5])
    ux = np.float64(words[6])
    uy = np.float64(words[7])
    cameras[camera_id] = [camera_width,camera_height,fx,fy,ux,uy]
f_camera.close()

f_images = open(dataset_path+"/images.txt")
images = {}
lines = f_images.readlines()[4:]
for i in range(len(lines)):
    if i%2 == 1:
        continue 
    words = lines[i].split()
    image_name = words[-1].split("/")[-1]
    camera_id = np.uint32(words[-2])
    qw = np.float64(words[1])
    qx = np.float64(words[2])
    qy = np.float64(words[3])
    qz = np.float64(words[4])
    tx = np.float64(words[5])
    ty = np.float64(words[6])
    tz = np.float64(words[7])
    images[image_name]=[camera_id,Pose(qw,qx,qy,qz,tx,ty,tz)]
f_images.close()

# Open the database.
database_path = dataset_path+"/colmap/database.db"
db = COLMAPDatabase.connect(database_path)

two_view_geometries = db.execute("SELECT * FROM two_view_geometries")

write_path = dataset_path + "/two_views.txt"
f_write = open(write_path,"w")

f_write.write("# img_name1 image_name2 f1 f2 num_inlier rot[0] rot[1] rot[2] trans[0] trans[1] trans[2]\n"
        +"# features1 [p0x p0y p1x p1y ...]\n"+ "# features2 [p0x p0y p1x p1y ...]\n") 

for two_view_geometry in two_view_geometries:
    pair_id = two_view_geometry[0]
    if two_view_geometry[3]== None:
        continue
    matches = blob_to_array(two_view_geometry[3],np.uint32,(-1,2))
    image_id1,image_id2 = pair_id_to_image_ids(pair_id)
    image_id1 = np.uint32(image_id1)
    
    image1 = db.execute("SELECT name FROM images WHERE image_id={0}".format(image_id1))
    image2 = db.execute("SELECT name FROM images WHERE image_id={0}".format(image_id2))
    image1_name = next(image1)[0]
    image2_name = next(image2)[0]
    image1_name = image1_name.split("/")[-1]
    image2_name = image2_name.split("/")[-1]

    F = blob_to_array(two_view_geometry[5],np.float64,(3,3))
    
    feature1 = db.execute("SELECT data FROM keypoints WHERE image_id={0}".format(image_id1))
    feature1_cols = db.execute("SELECT cols FROM keypoints WHERE image_id={0}".format(image_id1))
    feature1_cols = next(feature1_cols)[0]
    feature1 = next(feature1)[0]
    feature1 = blob_to_array(feature1,np.float32,(-1,feature1_cols))
    feature2 = db.execute("SELECT data FROM keypoints WHERE image_id={0}".format(image_id2))
    feature2_cols = db.execute("SELECT cols FROM keypoints WHERE image_id={0}".format(image_id2))
    feature2_cols = next(feature2_cols)[0]
    feature2 = next(feature2)[0]
    feature2 = blob_to_array(feature2,np.float32,(-1,feature2_cols))
    
    feature1 = feature1[:,0:2]
    feature2 = feature2[:,0:2]
    

    corrresponding1 = feature1[matches[:,0],:]
    corrresponding2 = feature2[matches[:,1],:]
    
    camera1 = cameras[images[image1_name][0]]
    camera2 = cameras[images[image2_name][0]]
    pose1 = images[image1_name][1]
    pose2 = images[image2_name][1]
    camera1 = {'model': 'PINHOLE', 'width': camera1[0], 'height': camera1[1], 'params': [camera1[2],camera1[3],camera1[4],camera1[5]]}
    camera2 = {'model': 'PINHOLE', 'width': camera2[0], 'height': camera2[1], 'params': [camera2[2],camera2[3],camera2[4],camera2[5]]}
    
    K1 = np.array([[camera1['params'][0],0.0,camera1['params'][2]],
                   [0.0,camera1['params'][1],camera1['params'][3]],
                   [0,0,1]])
    K2 = np.array([[camera2['params'][0],0.0,camera2['params'][2]],
                   [0.0,camera2['params'][1],camera2['params'][3]],
                   [0,0,1]])
    
    result = poselib.estimate_relative_pose(corrresponding1,corrresponding2,camera1,camera2)
    relative_pose = result[0]
    
    num_inliers = result[1]["num_inliers"]
    inliers_mask = result[1]["inliers"]
    
    point1 = corrresponding1[inliers_mask,:]
    point2 = corrresponding2[inliers_mask,:]
    rotation = Rotation.from_quat([relative_pose.q[1],relative_pose.q[2],relative_pose.q[3],relative_pose.q[0]]).as_rotvec()
    translation = relative_pose.t
    f1 = (camera1['params'][0]+camera1['params'][1])/2.0
    f2 = (camera2['params'][0]+camera2['params'][1])/2.0
    
    f_write.write("{} {} {} {} {} {} {} {} {} {} {}\n".format(image1_name,image2_name,f1,f2,num_inliers,
                                                    rotation[0],rotation[1],rotation[2],
                                                    translation[0],translation[1],translation[2]))
    point1_string = ""
    point2_string = ""
    for i in range(num_inliers):
        point1_string += "{} {} ".format(point1[i,0],point1[i,1])
        point2_string += "{} {} ".format(point2[i,0],point2[i,1])
    point1_string+="\n"
    point2_string+="\n"
    f_write.write(point1_string)
    f_write.write(point2_string)
    
    
f_write.close()
db.close()
