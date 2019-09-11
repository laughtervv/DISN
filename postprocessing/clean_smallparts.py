import argparse
import numpy as np
import tensorflow as tf
import pymesh
import os
import sys
from joblib import Parallel, delayed
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR) # model
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'data'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'preprocessing'))
slim = tf.contrib.slim



parser = argparse.ArgumentParser()
parser.add_argument('--src_dir', type=str, default="", help='src directory, before clean')
parser.add_argument('--tar_dir', type=str, default="", help='where to store')
parser.add_argument('--thread_n', type=int, default=10, help='parallelism')

FLAGS = parser.parse_args()
print('pid: %s'%(str(os.getpid())))
print(FLAGS)

def separate_single_mesh(src_mesh, tar_mesh):
    src_mesh = pymesh.load_mesh(src_mesh)
    dis_meshes = pymesh.separate_mesh(src_mesh, connectivity_type='auto')
    pymesh.save_mesh_raw(tar_mesh+".obj", src_mesh.vertices, src_mesh.faces)
    count=0
    for dis_mesh in dis_meshes:
        print("dis_mesh.vertices.shape",dis_mesh.vertices.shape)
        print("dis_mesh.faces.shape",dis_mesh.faces.shape)
        pymesh.save_mesh_raw(tar_mesh+"_"+str(count)+".obj", dis_mesh.vertices, dis_mesh.faces)
        count+=1

def clean_single_mesh(src_mesh, tar_mesh, dist_thresh, num_thresh):
    src_mesh_obj = pymesh.load_mesh(src_mesh)
    dis_meshes = pymesh.separate_mesh(src_mesh_obj, connectivity_type='auto')
    max_mesh_verts = 0
    for dis_mesh in dis_meshes:
       if dis_mesh.vertices.shape[0] > max_mesh_verts:
           max_mesh_verts = dis_mesh.vertices.shape[0]

    collection=[]
    for dis_mesh in dis_meshes:
       if dis_mesh.vertices.shape[0] > max_mesh_verts*num_thresh:
           centroid = np.mean(dis_mesh.vertices, axis=0)
           if np.sqrt(np.sum(np.square(centroid))) < dist_thresh:
            collection.append(dis_mesh)
    tar_mesh_obj = pymesh.merge_meshes(collection)
    pymesh.save_mesh_raw(tar_mesh, tar_mesh_obj.vertices, tar_mesh_obj.faces)
    print("threshes:", str(dist_thresh), str(num_thresh), " clean: ", src_mesh, " create: ",tar_mesh)

def clean_meshes(cats, src_dir, tar_dir,dist_thresh=0.5, num_thresh=0.3, thread_n=12):
    for cat_nm, cat_id in cats.items():
        src_cat_dir=os.path.join(src_dir,cat_id)
        tar_cat_dir=os.path.join(tar_dir,cat_id)
        os.makedirs(tar_cat_dir, exist_ok=True)
        _,_,src_file_lst, tar_file_lst = build_file_dict(src_cat_dir, tar_cat_dir)
        dist_thresh_lst=[dist_thresh for i in range(len(src_file_lst))]
        num_thresh_lst=[num_thresh for i in range(len(src_file_lst))]
        with Parallel(n_jobs=thread_n) as parallel:
            parallel(delayed(clean_single_mesh)
                     (src_mesh, tar_mesh, dist_thresh, num_thresh)
                     for src_mesh, tar_mesh, dist_thresh, num_thresh in
                     zip(src_file_lst, tar_file_lst, dist_thresh_lst,num_thresh_lst))
        print("done with ", cat_nm, cat_id)
    print("done!")

def build_file_dict(src_dir, tar_dir):
    src_file_dict = {}
    tar_file_dict = {}
    src_file_lst=[]
    tar_file_lst=[]
    for file in os.listdir(src_dir):
        src_full_path = os.path.join(src_dir, file)
        tar_full_path = os.path.join(tar_dir, file)
        if os.path.isfile(src_full_path):
            obj_id = file.split("_")[1]
            if obj_id in src_file_dict.keys():
                src_file_dict[obj_id].append(src_full_path)
            else:
                src_file_dict[obj_id] = [src_full_path]
            src_file_lst.append(src_full_path)
            if obj_id in tar_file_dict.keys():
                tar_file_dict[obj_id].append(tar_full_path)
            else:
                tar_file_dict[obj_id] = [tar_full_path]
            tar_file_lst.append(tar_full_path)

    return src_file_dict, tar_file_dict,src_file_lst,tar_file_lst


if __name__ == "__main__":


    cats = {
        # "chair": "03001627",
        # "bench": "02828884",
        # "car": "02958343",
        # "airplane": "02691156",
        # "sofa": "04256520",
        # "table": "04379243",
        # "phone": "04401088",
        "cabinet": "02933112",
        "display": "03211117",
        # "lamp": "03636649",
        "speaker": "03691459",
        "rifle": "04090263",
        "watercraft": "04530566"
    }
    # src_dir = "checkpoint/all_best/sdf_2d_sdfproj_twostream_wd_2048_weight10_ftprev_inweight/test_objs/65_0.0"
    # tar_dir = "checkpoint/all_best/sdf_2d_sdfproj_twostream_wd_2048_weight10_ftprev_inweight/test_objs/65_0.0_sep"
    clean_meshes(cats, FLAGS.src_dir, FLAGS.tar_dir, dist_thresh=0.5, num_thresh=0.3, thread_n=FLAGS.thread_n)

