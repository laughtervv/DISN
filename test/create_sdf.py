import argparse
from datetime import datetime
import numpy as np
import random
import tensorflow as tf
import socket
import os
import sys
import h5py
import struct
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR) # model
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'data'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'preprocessing'))
import model_normalization as model
from concurrent.futures import ThreadPoolExecutor
import data_sdf_h5_queue # as data
import create_file_lst
slim = tf.contrib.slim
lst_dir, cats, all_cats, raw_dirs = create_file_lst.get_all_info()

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
parser.add_argument('--max_epoch', type=int, default=1, help='Epoch to run [default: 201]')
parser.add_argument('--img_h', type=int, default=137, help='Image Height')
parser.add_argument('--img_w', type=int, default=137, help='Image Width')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate [default: 0.001]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.9, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--num_classes', type=int, default=1024, help='vgg dim')
parser.add_argument('--num_points', type=int, default=1, help='Point Number [default: 2048]')
parser.add_argument('--sdf_res', type=int, default=64, help='sdf grid')
parser.add_argument('--mask_tp', type=str, default="neg_two_sides")
parser.add_argument('--mask_rt', type=int, default=40000)
parser.add_argument('--alpha', action='store_true')
parser.add_argument('--rot', action='store_true')
parser.add_argument('--tanh', action='store_true')
parser.add_argument('--cat_limit', type=int, default=168000, help="balance each category, 1500 * 24 = 36000")
parser.add_argument('--multi_view', action='store_true')
parser.add_argument('--num_sample_points', type=int, default=1, help='Sample Point Number [default: 2048]')

parser.add_argument('--log_dir', default='checkpoint/exp_200', help='Log dir [default: log]')
parser.add_argument('--test_lst_dir', default=lst_dir, help='test mesh data list')
parser.add_argument('--iso', type=float, default=0.0, help='iso value')
parser.add_argument('--threedcnn', action='store_true')
parser.add_argument('--img_feat_onestream', action='store_true')
parser.add_argument('--img_feat_twostream', action='store_true')
parser.add_argument('--category', default="all", help='Which single class to train on [default: None]')
parser.add_argument('--binary', action='store_true')
parser.add_argument('--create_obj', action='store_true', help="create_obj or test accuracy on test set")
parser.add_argument('--store', action='store_true')
parser.add_argument('--view_num', type=int, default=24, help="how many views do you want to create for each obj")
parser.add_argument('--cam_est', action='store_true', help="if you are using the estimated camera image h5")

parser.add_argument('--augcolorfore', action='store_true')
parser.add_argument('--augcolorback', action='store_true')
parser.add_argument('--backcolorwhite', action='store_true')

FLAGS = parser.parse_args()
print('pid: %s'%(str(os.getpid())))
print(FLAGS)

EPOCH_CNT = 0
NUM_POINTS = FLAGS.num_points
BATCH_SIZE = FLAGS.batch_size
RESOLUTION = FLAGS.sdf_res+1
TOTAL_POINTS = RESOLUTION * RESOLUTION * RESOLUTION
if FLAGS.img_feat_twostream:
    SPLIT_SIZE = int(np.ceil(TOTAL_POINTS / 214669.0))
elif FLAGS.threedcnn :
    SPLIT_SIZE = 1
else:
    SPLIT_SIZE = int(np.ceil(TOTAL_POINTS / 274625.0))
NUM_SAMPLE_POINTS = int(np.ceil(TOTAL_POINTS / SPLIT_SIZE))
GPU_INDEX = FLAGS.gpu
PRETRAINED_MODEL_PATH = FLAGS.log_dir
LOG_DIR = FLAGS.log_dir
SDF_WEIGHT = 10.

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_INDEX

if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)

RESULT_PATH = os.path.join(LOG_DIR, 'test_results_allpts')
if FLAGS.cam_est:
    RESULT_OBJ_PATH = os.path.join(LOG_DIR, 'test_objs', "camest_"
                                   + str(RESOLUTION) + "_" + str(FLAGS.iso))
    print("RESULT_OBJ_PATH: ",RESULT_OBJ_PATH)
else:
    RESULT_OBJ_PATH = os.path.join(LOG_DIR, 'test_objs', str(RESOLUTION) + "_" + str(FLAGS.iso))
if not os.path.exists(RESULT_PATH): os.mkdir(RESULT_PATH)
if not os.path.exists(RESULT_OBJ_PATH): os.makedirs(RESULT_OBJ_PATH, exist_ok=True)

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_test.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

IMG_SIZE = FLAGS.img_h
HOSTNAME = socket.gethostname()
print("HOSTNAME:", HOSTNAME)
VV = False
VV = VV and (HOSTNAME == "ubuntu")

TEST_LISTINFO = []

cat_ids = []
cats_limit = {}

if FLAGS.category == "all":
    for key, value in cats.items():
        cat_ids.append(value)
        cats_limit[value] = 0
else:
    cat_ids.append(cats[FLAGS.category])
    cats_limit[cats[FLAGS.category]] = 0

for cat_id in cat_ids:
    test_lst = os.path.join(FLAGS.test_lst_dir, cat_id+"_test.lst")
    with open(test_lst, 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            render_list = random.sample(range(24), FLAGS.view_num)
            for render in render_list:
                cats_limit[cat_id] += 1
                TEST_LISTINFO += [(cat_id, line.strip(), render)]


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


if FLAGS.threedcnn:
    info = {'rendered_dir': raw_dirs["renderedh5_dir_v2"],
            'sdf_dir': raw_dirs["3dnnsdf_dir"]}
elif FLAGS.img_feat_onestream or FLAGS.img_feat_twostream:
    info = {'rendered_dir': raw_dirs["renderedh5_dir"],
            'sdf_dir': raw_dirs["sdf_dir"]}
    if FLAGS.cam_est:
        info['rendered_dir']= raw_dirs["renderedh5_dir_est"]
else:
    info = {'rendered_dir': raw_dirs["renderedh5_dir_v2"],
            'sdf_dir': raw_dirs['sdf_dir_v2']}

TEST_DATASET = data_sdf_h5_queue.Pt_sdf_img(FLAGS,
    listinfo=TEST_LISTINFO, info=info, cats_limit=cats_limit, shuffle=False)
print(info)

def create():
    log_string(LOG_DIR)

    input_pls = model.placeholder_inputs(BATCH_SIZE, NUM_POINTS, (IMG_SIZE, IMG_SIZE),
                        num_sample_pc=NUM_SAMPLE_POINTS, scope='inputs_pl', FLAGS=FLAGS)
    is_training_pl = tf.placeholder(tf.bool, shape=())
    print(is_training_pl)
    batch = tf.Variable(0, name='batch')

    print("--- Get model and loss")
    # Get model and loss

    end_points = model.get_model(input_pls, NUM_POINTS, is_training_pl, bn=False,FLAGS=FLAGS)

    loss, end_points = model.get_loss(end_points,
        sdf_weight=SDF_WEIGHT, num_sample_points=NUM_SAMPLE_POINTS, FLAGS=FLAGS)
    # Create a session
    gpu_options = tf.GPUOptions() # per_process_gpu_memory_fraction=0.99
    config = tf.ConfigProto(gpu_options=gpu_options)
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)

    init = tf.global_variables_initializer()
    sess.run(init)

    ######### Loading Checkpoint ###############
    saver = tf.train.Saver([v for v in tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES) if
                            ('lr' not in v.name) and ('batch' not in v.name)])
    ckptstate = tf.train.get_checkpoint_state(PRETRAINED_MODEL_PATH)

    if ckptstate is not None:
        LOAD_MODEL_FILE = os.path.join(PRETRAINED_MODEL_PATH, os.path.basename(ckptstate.model_checkpoint_path))
        try:
            # load_model(sess, PRETRAINED_PN_MODEL_FILE, ['refpc_reconstruction','sdfprediction','vgg_16'], strict=True)
            with NoStdStreams():
                saver.restore(sess, LOAD_MODEL_FILE)
            print("Model loaded in file: %s" % LOAD_MODEL_FILE)
        except:
            print("Fail to load overall modelfile: %s" % PRETRAINED_MODEL_PATH)

    ###########################################

    ops = {'input_pls': input_pls,
           'is_training_pl': is_training_pl,
           'loss': loss,
           'step': batch,
           'end_points': end_points}

    TEST_DATASET.start()
    test_one_epoch(sess, ops)
    TEST_DATASET.shutdown()


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

def test_one_epoch(sess, ops):
    """ ops: dict mapping from string to tf ops """
    is_training = False

    # Shuffle train samples
    num_batches = int(len(TEST_DATASET)) // FLAGS.batch_size

    print('num_batches', num_batches)
    loss_all = 0
    log_string(str(datetime.now()))
    losses = {}
    for lossname in ops['end_points']['losses'].keys():
        losses[lossname] = 0

    with ThreadPoolExecutor(max_workers=4) as executor:
        for batch_idx in range(num_batches):
            batch_data = TEST_DATASET.fetch()
            extra_pts = np.zeros((1, SPLIT_SIZE * NUM_SAMPLE_POINTS - TOTAL_POINTS, 3), dtype=np.float32)
            batch_points = np.zeros((SPLIT_SIZE, 0, NUM_SAMPLE_POINTS, 3), dtype=np.float32)
            if not FLAGS.threedcnn:
                for b in range(BATCH_SIZE):
                    print(batch_data)
                    sdf_params = batch_data['sdf_params'][b]
                    x_ = np.linspace(sdf_params[0], sdf_params[3], num=RESOLUTION)
                    y_ = np.linspace(sdf_params[1], sdf_params[4], num=RESOLUTION)
                    z_ = np.linspace(sdf_params[2], sdf_params[5], num=RESOLUTION)
                    z, y, x = np.meshgrid(z_, y_, x_, indexing='ij')
                    x = np.expand_dims(x, 3)
                    y = np.expand_dims(y, 3)
                    z = np.expand_dims(z, 3)
                    all_pts = np.concatenate((x, y, z), axis=3).astype(np.float32)
                    all_pts = all_pts.reshape(1, -1, 3)
                    all_pts = np.concatenate((all_pts, extra_pts), axis=1).reshape(SPLIT_SIZE, 1, -1, 3)
                    print('all_pts', all_pts.shape)
                    batch_points = np.concatenate((batch_points, all_pts), axis=1)

            pred_sdf_val_all = np.zeros((SPLIT_SIZE, BATCH_SIZE, NUM_SAMPLE_POINTS, 2 if FLAGS.binary else 1))

            for sp in range(SPLIT_SIZE):
                if FLAGS.threedcnn:
                    feed_dict = {ops['is_training_pl']: is_training,
                                 ops['input_pls']['imgs']: batch_data['img']}
                else:
                    feed_dict = {ops['is_training_pl']: is_training,
                                 ops['input_pls']['sample_pc']: batch_points[sp,...].reshape(BATCH_SIZE, -1, 3),
                                 ops['input_pls']['sample_pc_rot']: batch_points[sp,...].reshape(BATCH_SIZE, -1, 3),
                                 ops['input_pls']['imgs']: batch_data['img'],
                                 ops['input_pls']['trans_mat']: batch_data['trans_mat']}

                output_list = [ops['end_points']['pred_sdf'], ops['end_points']['ref_img'],
                               ops['end_points']['sample_img_points']]
                pred_sdf_val, ref_img_val, sample_img_points_val = sess.run(output_list, feed_dict=feed_dict)
                pred_sdf_val_all[sp,:,:,:] = pred_sdf_val
            pred_sdf_val_all = np.swapaxes(pred_sdf_val_all,0,1) # B, S, NUM SAMPLE, 1 or 2
            pred_sdf_val_all = pred_sdf_val_all.reshape((BATCH_SIZE,-1,2 if FLAGS.binary else 1))[:, :TOTAL_POINTS, :]
            if FLAGS.binary:
                expo = np.exp(pred_sdf_val_all)
                prob = expo[:,:,1] / np.sum(expo, axis = 2)
                result = (prob - 0.5) / 10.
                print("result.shape", result.shape)
            else:
                result = pred_sdf_val_all / SDF_WEIGHT
            for b in range(BATCH_SIZE):
                print("{}/{}, submit create_obj {}, {}, {}".format(batch_idx, num_batches, batch_data['cat_id'][b], batch_data['obj_nm'][b], batch_data['view_id'][b]))
                executor.submit(create_obj, result[b], batch_data['sdf_params'][b], RESULT_OBJ_PATH,
                    batch_data['cat_id'][b], batch_data['obj_nm'][b], batch_data['view_id'][b], FLAGS.iso)


def to_binary(res, pos, pred_sdf_val_all, sdf_file):
    f_sdf_bin = open(sdf_file, 'wb')

    f_sdf_bin.write(struct.pack('i', -res))  # write an int
    f_sdf_bin.write(struct.pack('i', res))  # write an int
    f_sdf_bin.write(struct.pack('i', res))  # write an int

    positions = struct.pack('d' * len(pos), *pos)
    f_sdf_bin.write(positions)
    val = struct.pack('=%sf'%pred_sdf_val_all.shape[0], *(pred_sdf_val_all))
    f_sdf_bin.write(val)
    f_sdf_bin.close()

def create_obj(pred_sdf_val, sdf_params, dir, cat_id, obj_nm, view_id, i):
    if not isinstance(view_id, str):
        view_id = "%02d"%view_id
    dir = os.path.join(dir, cat_id)
    os.makedirs(dir, exist_ok=True)
    obj_nm = cat_id +"_" + obj_nm
    cube_obj_file = os.path.join(dir, obj_nm+"_"+view_id+".obj")
    sdf_file = os.path.join(dir, obj_nm+"_"+view_id+".dist")
    to_binary((RESOLUTION-1), sdf_params, pred_sdf_val, sdf_file)
    create_one_cube_obj("./isosurface/computeMarchingCubes", i, sdf_file, cube_obj_file)
    command_str = "rm -rf " + sdf_file
    print("command:", command_str)
    os.system(command_str)

def create_one_cube_obj(marching_cube_command, i, sdf_file, cube_obj_file):
    command_str = marching_cube_command + " " + sdf_file + " " + cube_obj_file + " -i " + str(i)
    print("command:", command_str)
    os.system(command_str)
    return cube_obj_file

def get_sdf_h5(sdf_h5_file, cat_id, obj):
    h5_f = h5py.File(sdf_h5_file, 'r')
    try:
        if ('pc_sdf_original' in h5_f.keys()
                and 'pc_sdf_sample' in h5_f.keys()
                and 'norm_params' in h5_f.keys()):
            ori_sdf = h5_f['pc_sdf_original'][:].astype(np.float32)
            # sample_sdf = np.reshape(h5_f['pc_sdf_sample'][:],(ori_sdf.shape[0], -1 ,4)).astype(np.float32)
            sample_sdf = h5_f['pc_sdf_sample'][:].astype(np.float32)
            ori_pt = ori_sdf[:,:3]#, ori_sdf[:,3]
            ori_sdf_val = None
            if sample_sdf.shape[1] == 4:
                sample_pt, sample_sdf_val = sample_sdf[:,:3], sample_sdf[:,3]
            else:
                sample_pt, sample_sdf_val = None, sample_sdf[:, 0]
            norm_params = h5_f['norm_params'][:]
            sdf_params = h5_f['sdf_params'][:]
        else:
            raise Exception(cat_id, obj, "no sdf and sample")
    finally:
        h5_f.close()
    return ori_pt, ori_sdf_val, sample_pt, sample_sdf_val, norm_params, sdf_params

if __name__ == "__main__":


    # 1. create all categories / some of the categories:
    create()



    # 2. create single obj, just run python -u create_sdf.py

    # ori_pt, ori_sdf_val, sample_pt, sample_sdf_val, norm_params, sdf_params = \
    #     get_sdf_h5("/ssd1/datasets/ShapeNet/SDF_full/64_expr_1.2/03001627/47cd848a5584867b1e8791c225564ae0/ori_sample.h5",
    #                 "03001627", "47cd848a5584867b1e8791c225564ae0")
    # create_obj(sample_sdf_val, sdf_params, "send/",
    #            "03001627", "97cd4ed02e022ce7174150bd56e389a8", "111", 0.00)

