import argparse
import numpy as np
import random
import tensorflow as tf
import socket
import pymesh
import os
import sys
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(BASE_DIR) # model
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'data'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'preprocessing'))
from tensorflow.contrib.framework.python.framework import checkpoint_utils

import models.tf_ops.nn_distance.tf_nndistance as tf_nndistance
import create_file_lst
slim = tf.contrib.slim

parser = argparse.ArgumentParser()
lst_dir, cats, all_cats, raw_dirs = create_file_lst.get_all_info()

parser.add_argument('--store', action='store_true')
parser.add_argument('--max_epoch', type=int, default=1, help='Epoch to run [default: 201]')
parser.add_argument('--img_h', type=int, default=137, help='Image Height')
parser.add_argument('--img_w', type=int, default=137, help='Image Width')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate [default: 0.001]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.9, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--num_classes', type=int, default=1024, help='vgg global embedding dimensions')
parser.add_argument('--num_points', type=int, default=1, help='Point Number [default: 2048]')
parser.add_argument('--mask_tp', type=str, default="neg_two_sides")
parser.add_argument('--mask_rt', type=int, default=40000)
parser.add_argument('--alpha', action='store_true')
parser.add_argument('--rot', action='store_true')
parser.add_argument('--tanh', action='store_true')
parser.add_argument('--cat_limit', type=int, default=168000, help="balance each category, 1500 * 24 = 36000")
parser.add_argument('--sdf_res', type=int, default=64, help='sdf grid')
parser.add_argument('--binary', action='store_true')
parser.add_argument('--truethreshold', type=float, default=2.5, help='if distance smaller than this value, its true')

parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
parser.add_argument('--batch_size', type=int, default=24, help='Batch Size during training [default: 32]')
parser.add_argument('--log_dir', default='checkpoint/exp_200', help='Log dir [default: log]')
parser.add_argument('--test_lst_dir', default=lst_dir, help='test mesh data list')
parser.add_argument('--num_sample_points', type=int, default=2048, help='Sample Point Number for each obj to test[default: 2048]')
parser.add_argument('--threedcnn', action='store_true')
parser.add_argument('--img_feat_onestream', action='store_true')
parser.add_argument('--img_feat_far', action='store_true')
parser.add_argument('--img_feat_twostream', action='store_true')
parser.add_argument('--category', default="all", help='Which single class to train on [default: None]')
parser.add_argument('--view_num', type=int, default=24, help="how many views do you want to create for each obj")
parser.add_argument('--cam_est', action='store_true')
parser.add_argument('--cal_dir', type=str, default="", help="target obj directory that needs to be tested")

FLAGS = parser.parse_args()
print('pid: %s'%(str(os.getpid())))
print(FLAGS)

EPOCH_CNT = 0
BATCH_SIZE = FLAGS.batch_size
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

IMG_SIZE = FLAGS.img_h
VV =False
HOSTNAME = socket.gethostname()

TEST_LISTINFO = []


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
        if os.path.isfile(full_path):
            obj_id = file.split("_")[1]
            if obj_id in file_dict.keys():
                file_dict[obj_id].append(full_path)
            else:
                file_dict[obj_id] = [full_path]
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


def cal_f_score_all_cat(cats, pred_dir, gt_dir, test_lst_dir,threshold_lst, side_len):
    precision_lst = []
    recall_lst = []
    cnt_lst = []
    for cat_nm, cat_id in cats.items():
        pred_dir_cat = os.path.join(pred_dir, cat_id)
        gt_dir_cat = os.path.join(gt_dir, cat_id)
        test_lst_f = os.path.join(test_lst_dir, cat_id + "_test.lst")
        thresholds = np.asarray(threshold_lst, dtype=np.float32) * 0.01 * side_len
        precision_avg, recall_avg, cnt \
            = f_score_cat(cat_id, cat_nm, pred_dir_cat, gt_dir_cat, test_lst_f, thresholds)
        precision_lst.append(precision_avg)
        recall_lst.append(recall_avg)
        cnt_lst.append(cnt)
        print("{}, {}, precision_avg {}, recal_avg{}, count {}"
              .format(cat_nm, cat_id, precision_avg, recall_avg, cnt))
    print("done!")
    precision = np.asarray(precision_lst) # 13 * 5
    recall = np.asarray(recall_lst)
    pre_w_avg = np.average(precision, axis=0, weights=cnt_lst)
    rec_w_avg = np.average(recall, axis=0, weights=cnt_lst)
    f_score = 2 * (pre_w_avg * rec_w_avg) / (pre_w_avg + rec_w_avg)
    print("pre_w_avg {}, rec_w_avg {}, f_score {}".format(pre_w_avg, rec_w_avg, f_score))

def f_score_cat(cat_id, cat_nm, pred_dir, gt_dir, test_lst_f, thresholds):
    pred_dict = build_file_dict(pred_dir)
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            sess = tf.Session(config=config)
            sampled_pc = tf.placeholder(tf.float32, shape=(FLAGS.batch_size + 1, FLAGS.num_sample_points, 3))
            #
            dists_forward_sqrt, dists_backward_sqrt \
                = get_points_distance(sampled_pc)
            count = 0
            precision_sum = 0
            recall_sum = 0
            with open(test_lst_f, "r") as f:
                test_objs = f.readlines()
                for obj_id in test_objs:
                    obj_id = obj_id.rstrip('\r\n')
                    pred_pnt_dir = os.path.join(os.path.dirname(pred_dir),
                        "pnt_{}_{}".format(FLAGS.num_sample_points, cat_id))
                    forfl = os.path.join(pred_pnt_dir, "for_dist_{}.txt".format(obj_id))
                    backfl = os.path.join(pred_pnt_dir, "bac_dist_{}.txt".format(obj_id))
                    if not os.path.exists(forfl):
                        gt_pnt_path = os.path.join(gt_dir, obj_id, "pnt_{}.txt".format(FLAGS.num_sample_points))
                        # npnts, 3
                        gt_pnts = np.loadtxt(gt_pnt_path,dtype=float, delimiter=',')
                        pred_path_lst = pred_dict[obj_id]
                        verts_batch = np.zeros((FLAGS.view_num + 1, FLAGS.num_sample_points, 3), dtype=np.float32)
                        verts_batch[0, ...] = gt_pnts
                        for i in range(len(pred_path_lst)):
                            pred_mesh_fl = pred_path_lst[i]
                            view_id = pred_mesh_fl[-6:-4]
                            pred_pnt_path = os.path.join(pred_pnt_dir, "pnt_{}_{}.txt".format(obj_id, view_id))
                            pred_pnts = np.loadtxt(pred_pnt_path,dtype=float, delimiter=',')
                            verts_batch[i + 1, ...] = pred_pnts
                        if FLAGS.batch_size == FLAGS.view_num:
                            feed_dict = {sampled_pc: verts_batch}
                            # view * npnt
                            dists_forward_sqrt_val, dists_backward_sqrt_val\
                                = sess.run([dists_forward_sqrt, dists_backward_sqrt], feed_dict=feed_dict)
                        else:
                            raise NotImplementedError
                        np.savetxt(forfl, dists_forward_sqrt_val)
                        np.savetxt(backfl, dists_backward_sqrt_val)
                    else:
                        dists_forward_sqrt_val = np.loadtxt(forfl)
                        dists_backward_sqrt_val = np.loadtxt(backfl)
                    dists_forward_sqrt_val = np.tile(dists_forward_sqrt_val, [thresholds.shape[0], 1])
                    dists_backward_sqrt_val = np.tile(dists_backward_sqrt_val, [thresholds.shape[0], 1])
                    pre_sum_val = np.sum(np.less(dists_forward_sqrt_val, thresholds), axis=1)
                    rec_sum_val = np.sum(np.less(dists_backward_sqrt_val, thresholds), axis=1)
                    precision = pre_sum_val / (dists_forward_sqrt_val.shape[1])
                    recall = rec_sum_val / (dists_backward_sqrt_val.shape[1])
                    print("cat_id {}, obj_id {}: pre_sum {}, rec_sum {}, precision {}, recall {}"
                          .format(cat_id, obj_id, pre_sum_val, rec_sum_val, precision, recall))
                    precision_sum += precision
                    recall_sum += recall
                    count += 1
    return precision_sum/count, recall_sum/count, count

def get_points_distance(sampled_pc):
    src_pc = tf.tile(tf.expand_dims(sampled_pc[0, :, :], axis=0), (FLAGS.batch_size, 1, 1))
    if sampled_pc.get_shape().as_list()[0] == 2:
        pred = tf.expand_dims(sampled_pc[1, :, :], axis=0)
    else:
        pred = sampled_pc[1:, :, :]
    print(src_pc.get_shape())
    print(pred.get_shape())
    dists_forward, _, dists_backward, _ = tf_nndistance.nn_distance(pred, src_pc)
    dists_forward_sqrt = tf.sqrt(dists_forward)
    dists_backward_sqrt = tf.sqrt(dists_backward)
    dists_forward_sqrt = tf.reshape(dists_forward_sqrt, [-1])
    dists_backward_sqrt = tf.reshape(dists_backward_sqrt, [-1])
    return dists_forward_sqrt, dists_backward_sqrt


if __name__ == "__main__":
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
        cats=cats_all
    elif FLAGS.category == "clean":
        cats ={ "cabinet": "02933112",
                "display": "03211117",
                "speaker": "03691459",
                "rifle": "04090263",
                "watercraft": "04530566"
        }
    else:
        cats={FLAGS.category: cats_all[FLAGS.category]}

    # calculate distance
    cal_f_score_all_cat(cats, FLAGS.cal_dir, info["gt_marching_cube"],
        FLAGS.test_lst_dir, [[0.5], [1], [2], [5], [10], [20]], FLAGS.truethreshold)

# nohup python -u test/test_cd_emd.py --gpu 0 --threedcnn --num_points 2048 --category all --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/main/3dcnn/test_objs/65_0.0 &> log/f_3dcnn_pnt.log &
# nohup python -u test/test_cd_emd.py --gpu 1 --num_points 2048 --category all --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/main/IM-SVR/test_objs/65_0.0_comb &> log/f_IM-SVR_pnt.log &
# nohup python -u test/test_cd_emd.py --gpu 2 --img_feat_twostream --num_points 2048 --category all --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/main/DISN/test_objs/65_0.0_comb &> log/f_DISN_pnt.log &
# nohup python -u test/test_cd_emd.py --gpu 3 --cam_est --img_feat_twostream --num_points 2048 --category all --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/main/DISN/test_objs/camest_65_0.0_comb &> log/f_DISNcamest_pnt.log &


# nohup python -u test/test_cd_emd.py --gpu 0 --threedcnn --num_points 2048 --category all --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/main/3dcnn/test_objs/65_0.0 &> log/f_3dcnn_pnt_1.6.log &
# nohup python -u test/test_cd_emd.py --gpu 1 --num_points 2048 --category all --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/main/IM-SVR/test_objs/65_0.0_comb &> log/f_IM-SVR_pnt_1.6.log &
# nohup python -u test/test_cd_emd.py --gpu 2 --img_feat_twostream --num_points 2048 --category all --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/main/DISN/test_objs/65_0.0_comb &> log/f_DISN_pnt_1.6.log &
# nohup python -u test/test_cd_emd.py --gpu 3 --cam_est --img_feat_twostream --num_points 2048 --category all --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/main/DISN/test_objs/camest_65_0.0_comb &> log/f_DISNcamest_pnt_1.6.log &




# nohup python -u test/test_cd_emd.py --gpu 0 --binary --img_feat_twostream --num_points 2048 --category chair --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/ablation/loctwobin/test_objs/65_0.0 &> log/f_binary_chair_pnt_1.6.log &
# nohup python -u test/test_cd_emd.py --gpu 2 --binary --cam_est --img_feat_twostream --num_points 2048 --category chair --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/ablation/loctwobin/test_objs/camest_65_0.0 &> log/f_binarycamest_chair_pnt_1.6.log &
# nohup python -u test/test_cd_emd.py --gpu 1  --num_points 2048 --category chair --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/ablation/noloco/test_objs/65_0.0 &> log/f_global_chair_pnt_1.6.log &
# nohup python -u test/test_cd_emd.py --gpu 2 --img_feat_onestream --num_points 2048 --category chair --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/ablation/onestream/test_objs/65_0.0 &> log/f_onestream_chair_pnt_1.6.log &
# nohup python -u test/test_cd_emd.py --gpu 2 --cam_est --img_feat_onestream --num_points 2048 --category chair --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/ablation/onestream/test_objs/camest_65_0.0 &> log/f_onestreamcamest_chair_pnt_1.6.log &
# nohup python -u test/test_cd_emd.py --gpu 3 --img_feat_twostream --num_points 2048 --category chair --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/ablation/DISNChair/test_objs/65_0.0 &> log/f_DISN_chair_pnt_1.6.log &
# nohup python -u test/test_cd_emd.py --gpu 3 --cam_est --img_feat_twostream --num_points 2048 --category chair --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/ablation/DISNChair/test_objs/camest_65_0.0 &> log/f_DISNcamest_chair_pnt_1.6.log &



# 2.5
# #
# nohup python -u test/test_cd_emd.py --gpu 2 --threedcnn --num_points 2048 --category all --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/main/3dcnn/test_objs/65_0.0 &> log/f_3dcnn_pnt_2.5.log &
# nohup python -u test/test_cd_emd.py --gpu 3 --num_points 2048 --category all --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/main/IM-SVR/test_objs/65_0.0_comb &> log/f_IM-SVR_pnt_2.5.log &
# nohup python -u test/test_cd_emd.py --gpu 2 --img_feat_twostream --num_points 2048 --category all --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/main/DISN/test_objs/65_0.0_comb &> log/f_DISN_pnt_2.5.log &
# nohup python -u test/test_cd_emd.py --gpu 3 --cam_est --img_feat_twostream --num_points 2048 --category all --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/main/DISN/test_objs/camest_65_0.0_comb &> log/f_DISNcamest_pnt_2.5.log &
# nohup python -u test/test_cd_emd.py --gpu 2 --binary --img_feat_twostream --num_points 2048 --category chair --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/ablation/loctwobin/test_objs/65_0.0 &> log/f_binary_chair_pnt_2.5.log &
# nohup python -u test/test_cd_emd.py --gpu 3 --binary --cam_est --img_feat_twostream --num_points 2048 --category chair --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/ablation/loctwobin/test_objs/camest_65_0.0 &> log/f_binarycamest_chair_pnt_2.5.log &
# nohup python -u test/test_cd_emd.py --gpu 2  --num_points 2048 --category chair --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/ablation/noloco/test_objs/65_0.0 &> log/f_global_chair_pnt_2.5.log &
# nohup python -u test/test_cd_emd.py --gpu 3 --img_feat_onestream --num_points 2048 --category chair --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/ablation/onestream/test_objs/65_0.0 &> log/f_onestream_chair_pnt_2.5.log &
# nohup python -u test/test_cd_emd.py --gpu 2 --cam_est --img_feat_onestream --num_points 2048 --category chair --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/ablation/onestream/test_objs/camest_65_0.0 &> log/f_onestreamcamest_chair_pnt_2.5.log &
# nohup python -u test/test_cd_emd.py --gpu 3 --img_feat_twostream --num_points 2048 --category chair --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/ablation/DISNChair/test_objs/65_0.0 &> log/f_DISN_chair_pnt_2.5.log &
# nohup python -u test/test_cd_emd.py --gpu 2 --cam_est --img_feat_twostream --num_points 2048 --category chair --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/ablation/DISNChair/test_objs/camest_65_0.0 &> log/f_DISNcamest_chair_pnt_2.5.log &
# #
