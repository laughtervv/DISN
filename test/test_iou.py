import argparse
import math
from datetime import datetime
import numpy as np
import random
import tensorflow as tf
import socket
import pymesh
import os
import sys
from joblib import Parallel, delayed
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(BASE_DIR) # model
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'data'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'preprocessing'))
from tensorflow.contrib.framework.python.framework import checkpoint_utils
import create_file_lst

slim = tf.contrib.slim
lst_dir, cats, all_cats, raw_dirs = create_file_lst.get_all_info()

parser = argparse.ArgumentParser()
parser.add_argument('--store', action='store_true')
parser.add_argument('--max_epoch', type=int, default=1, help='Epoch to run [default: 201]')
parser.add_argument('--img_h', type=int, default=137, help='Image Height')
parser.add_argument('--img_w', type=int, default=137, help='Image Width')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate [default: 0.001]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.9, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--num_classes', type=int, default=1024, help='vgg global embedding dimensions')
parser.add_argument('--num_points', type=int, default=1, help='Point Number [default: 2048]')
parser.add_argument('--dim', type=int, default=110)
parser.add_argument('--alpha', action='store_true')
parser.add_argument('--rot', action='store_true')
parser.add_argument('--tanh', action='store_true')
parser.add_argument('--cat_limit', type=int, default=168000, help="balance each category, 1500 * 24 = 36000")
parser.add_argument('--sdf_res', type=int, default=64, help='sdf grid')
parser.add_argument('--binary', action='store_true')
parser.add_argument('--num_sample_points', type=int, default=2048, help='Sample Point Number for each obj to test[default: 2048]')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 32]')
parser.add_argument('--cam_est', action='store_true')


parser.add_argument('--log_dir', default='checkpoint/exp_200', help='Log dir [default: log]')
parser.add_argument('--test_lst_dir', default=lst_dir, help='test mesh data list')
parser.add_argument('--threedcnn', action='store_true')
parser.add_argument('--img_feat_onestream', action='store_true')
parser.add_argument('--img_feat_twostream', action='store_true')
parser.add_argument('--category', default="all", help='Which single class to train on [default: None]')
parser.add_argument('--view_num', type=int, default=24, help="how many views do you want to create for each obj")
parser.add_argument('--cal_dir', type=str, default="", help="target obj directory that needs to be tested")

FLAGS = parser.parse_args()
print('pid: %s'%(str(os.getpid())))
print(FLAGS)

EPOCH_CNT = 0
GPU_INDEX = FLAGS.gpu
PRETRAINED_MODEL_PATH = FLAGS.log_dir
LOG_DIR = FLAGS.log_dir
SDF_WEIGHT = 10.

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_INDEX
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
RESULT_PATH = os.path.join(LOG_DIR, 'test_results_allpts')
if not os.path.exists(RESULT_PATH): os.mkdir(RESULT_PATH)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_test.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')


HOSTNAME = socket.gethostname()
print("HOSTNAME:", HOSTNAME)
VV = False
VV = VV and (HOSTNAME == "ubuntu")

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


if FLAGS.threedcnn:
    info = {'rendered_dir': raw_dirs["renderedh5_dir_v2"],
            'sdf_dir': raw_dirs["3dnnsdf_dir"],
            'gt_marching_cube':raw_dirs['norm_mesh_dir_v2']}
elif FLAGS.img_feat_onestream or FLAGS.img_feat_twostream:
    info = {'rendered_dir': raw_dirs["renderedh5_dir"],
            'sdf_dir': raw_dirs["sdf_dir"],
            'gt_marching_cube':raw_dirs['norm_mesh_dir']}
    if FLAGS.cam_est:
        info['rendered_dir']= raw_dirs["renderedh5_dir_est"]
else:
    info = {'rendered_dir': raw_dirs["renderedh5_dir_v2"],
            'sdf_dir': raw_dirs['sdf_dir_v2'],
            'gt_marching_cube':raw_dirs['norm_mesh_dir_v2']}

print(info)

def load_model(sess, LOAD_MODEL_FILE, prefixs, strict=False):

    vars_in_pretrained_model = dict(checkpoint_utils.list_variables(LOAD_MODEL_FILE))
    vars_in_defined_model = []

    for var in tf.trainable_variables():
        if isinstance(prefixs, list):
            for prefix in prefixs:
                if (var.op.name.startswith(prefix)) and (var.op.name in vars_in_pretrained_model.keys()) and ('logits' not in var.op.name):
                    if (list(var.shape) == vars_in_pretrained_model[var.op.name]):
                        vars_in_defined_model.append(var)
        else:
            if (var.op.name.startswith(prefixs)) and (var.op.name in vars_in_pretrained_model.keys()) and ('logits' not in var.op.name):
                if (list(var.shape) == vars_in_pretrained_model[var.op.name]):
                    vars_in_defined_model.append(var)
    saver = tf.train.Saver(vars_in_defined_model)
    saver.restore(sess, LOAD_MODEL_FILE)
    print( "Model loaded in file: %s" % (LOAD_MODEL_FILE))

    return True

def build_file_dict(dir):
    file_dict = {}
    for file in os.listdir(dir):
        full_path = os.path.join(dir, file)
        if os.path.isfile(full_path) and os.stat(full_path)[6] > 200:
            obj_id = file.split("_")[1]
            if obj_id in file_dict.keys():
                file_dict[obj_id].append(full_path)
            else:
                file_dict[obj_id] = [full_path]
    for obj_id in file_dict.keys():
        paths = file_dict[obj_id]
        if len(paths) == 0:
            print("{} no ok files ".format(obj_id))
        elif len(paths) < FLAGS.view_num:
            choice = np.random.randint(len(paths), size=FLAGS.view_num)
            # print(obj_id," choice: ",choice)
            file_dict[obj_id] = []
            for ind in choice:
                file_dict[obj_id].append(paths[ind])
        else:
            file_dict[obj_id] = random.sample(paths, FLAGS.view_num)
    return file_dict

class NoStdStreams(object):
    def __init__(self,stdout = None, stderr = None):
        self.devnull = open(os.devnull,'w')
        self._stdout = stdout or self.devnull or sys.stdout
        self._stderr = stderr or self.devnull or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush(); self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush(); self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        self.devnull.close()


def iou_all(cats, pred_dir, gt_dir, test_lst_dir, dim=110):
    for cat_nm, cat_id in cats.items():
        pred_dir_cat = os.path.join(pred_dir, cat_id)
        gt_dir_cat = os.path.join(gt_dir, cat_id)
        test_lst_f = os.path.join(test_lst_dir, cat_id + "_test.lst")
        iou_avg, best_iou_pred_lst = iou_cat(pred_dir_cat, gt_dir_cat, test_lst_f, dim=dim)
        print("cat_nm: {}, cat_id: {}, iou_avg: {}".format(cat_nm, cat_id, iou_avg))
    print("done!")

def iou_cat(pred_dir, gt_dir, test_lst_f, dim=110):
    pred_dict = build_file_dict(pred_dir)
    iou_sum = 0.0
    count = 0.0
    best_iou_pred_lst = []
    with open(test_lst_f, "r") as f:
        test_objs = f.readlines()
        for obj_id in test_objs:
            obj_id = obj_id.rstrip('\r\n')
            src_path = os.path.join(gt_dir, obj_id, "isosurf.obj")
            src_path_lst = [src_path for i in range(len(pred_dict[obj_id]))]
            dim_lst = [dim for i in range(len(pred_dict[obj_id]))]
            if obj_id not in pred_dict.keys():
                print("skip error obj id, no key:", obj_id)
                continue
            pred_path_lst = pred_dict[obj_id]
            if len(pred_path_lst) == 0:
                print("skip error obj id:", obj_id)
                continue
            with Parallel(n_jobs=min(12, FLAGS.view_num)) as parallel:
                result_lst = parallel(delayed(iou_pymesh)
                         (src_path, pred_path, dim)
                         for src_path, pred_path, dim in
                         zip(src_path_lst, pred_path_lst, dim_lst))
            iou_vals = np.asarray([result[0] for result in result_lst], dtype=np.float32)
            sum_iou = np.sum(iou_vals)
            iou_sum += sum_iou
            count += len(iou_vals)
            avg_iou = np.mean(iou_vals)
            ind = np.argmax(iou_vals)
            best_iou_pred_lst.append(result_lst[ind])
            print("obj_id iou avg: ", avg_iou, " best pred: ", result_lst[ind])
    return iou_sum / count, best_iou_pred_lst

def iou_pymesh(mesh_src, mesh_pred, dim=FLAGS.dim):
    try:
        mesh1 = pymesh.load_mesh(mesh_src)
        grid1 = pymesh.VoxelGrid(2./dim)
        grid1.insert_mesh(mesh1)
        grid1.create_grid()

        ind1 = ((grid1.mesh.vertices + 1.1) / 2.4 * dim).astype(np.int)
        v1 = np.zeros([dim, dim, dim])
        v1[ind1[:,0], ind1[:,1], ind1[:,2]] = 1


        mesh2 = pymesh.load_mesh(mesh_pred)
        grid2 = pymesh.VoxelGrid(2./dim)
        grid2.insert_mesh(mesh2)
        grid2.create_grid()

        ind2 = ((grid2.mesh.vertices + 1.1) / 2.4 * dim).astype(np.int)
        v2 = np.zeros([dim, dim, dim])
        v2[ind2[:,0], ind2[:,1], ind2[:,2]] = 1

        intersection = np.sum(np.logical_and(v1, v2))
        union = np.sum(np.logical_or(v1, v2))
        return [float(intersection) / union, mesh_pred]
    except:
        print("error mesh {} / {}".format(mesh_src, mesh_pred))


if __name__ == "__main__":


############################################################

    cats_all = {
        "watercraft": "04530566",
        "rifle": "04090263",
        "display": "03211117",
        "lamp": "03636649",
        "speaker": "03691459",
        "chair": "03001627",
        "bench": "02828884",
        "cabinet": "02933112",
        "car": "02958343",
        "airplane": "02691156",
        "sofa": "04256520",
        "table": "04379243",
        "phone": "04401088"
    }
    if FLAGS.category == "all":
        cats = cats_all
    elif FLAGS.category == "clean":
        cats = {"cabinet": "02933112",
                "display": "03211117",
                "lamp": "03636649",
                "speaker": "03691459",
                "rifle": "04090263",
                "watercraft": "04530566"
                }
    else:
        cats = {FLAGS.category: cats_all[FLAGS.category]}

    iou_all(cats, FLAGS.cal_dir, info["gt_marching_cube"], FLAGS.test_lst_dir, dim=110)

