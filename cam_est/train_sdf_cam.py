import argparse
import math
from datetime import datetime
import numpy as np
import random
import tensorflow as tf
import socket
import importlib
import os
import cv2
import sys
import h5py
import time
from tensorflow.contrib.framework.python.framework import checkpoint_utils
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(os.path.join(os.path.dirname(BASE_DIR), 'data'))
sys.path.append(BASE_DIR) # model
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'preprocessing'))
sys.path.append(os.path.join(os.path.dirname(BASE_DIR), 'data'))
print(os.path.join(BASE_DIR, 'data'))
import model_cam as model
import data_sdf_h5_queue_mask_imgh5_cammat as data
import create_file_lst

slim = tf.contrib.slim
lst_dir, cats, all_cats, raw_dirs = create_file_lst.get_all_info()

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='2', help='GPU to use [default: GPU 0]')
parser.add_argument('--category', default="all", help='Which single class to train on [default: None]')
parser.add_argument('--log_dir', default='checkpoint/sdf_2d_twostream_cam_pcrot_all', help='Log dir [default: log]')
parser.add_argument('--num_points', type=int, default=1, help='Point Number [default: 2048]')
parser.add_argument('--num_sample_points', type=int, default=2048, help='Sample Point Number [default: 2048]')
parser.add_argument('--max_epoch', type=int, default=200, help='Epoch to run [default: 201]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--img_h', type=int, default=137, help='Image Height')
parser.add_argument('--img_w', type=int, default=137, help='Image Width')
parser.add_argument('--verbose_freq', type=int, default=100, help='verbose frequency')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--restore_model', default='', help='restore_model')
parser.add_argument('--restore_modelpn', default='', help='restore_model')#checkpoint/sdf_3dencoder_sdfbasic2/latest.ckpt
parser.add_argument('--restore_modelcnn', default='', help='restore_model')#../../models/CNN/pretrained_model/vgg_16.ckpt
parser.add_argument('--rotation', action='store_true', help='Disable random rotation during training.')
parser.add_argument('--sample', action='store_false', help='Disable sample during training.')
parser.add_argument('--img_feat', action='store_false', help='Disable sample during training.')
parser.add_argument('--splitvalid', action='store_true', help='Disable sample during training.')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.9, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--loss_mode', type=str, default="3D", help='loss on 3D points or 2D points')
parser.add_argument('--test', action="store_true")
parser.add_argument('--create', action="store_true")
parser.add_argument('--cat_limit', type=int, default=168000, help="balance each category, 1500 * 24 = 36000")
parser.add_argument('--img_h5_dir', type=str, default=raw_dirs["renderedh5_dir_est"], help="where to save img_h5")
parser.add_argument('--shift', action="store_true")
parser.add_argument('--shift_weight', type=float, default=0.5)

FLAGS = parser.parse_args()

EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINTS = FLAGS.num_points
NUM_SAMPLE_POINTS = FLAGS.num_sample_points
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
PRETRAINED_MODEL_PATH = FLAGS.restore_model
PRETRAINED_CNN_MODEL_FILE = FLAGS.restore_modelcnn
PRETRAINED_PN_MODEL_FILE = FLAGS.restore_modelpn
LOG_DIR = FLAGS.log_dir
VV=False

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_INDEX

if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)

RESULT_PATH = os.path.join(LOG_DIR, 'train_results')
if not os.path.exists(RESULT_PATH): os.mkdir(RESULT_PATH)

VALID_RESULT_PATH = os.path.join(LOG_DIR, 'valid_results_'+str(time.time()))
TEST_RESULT_PATH = os.path.join(LOG_DIR, 'test_results_'+str(time.time()))
if not os.path.exists(VALID_RESULT_PATH): os.mkdir(VALID_RESULT_PATH)
if not os.path.exists(TEST_RESULT_PATH): os.mkdir(TEST_RESULT_PATH)

os.system('cp %s.py %s' % (os.path.splitext(model.__file__)[0], LOG_DIR))
os.system('cp %s.py %s' % (os.path.splitext(__file__)[0], LOG_DIR))
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train_%s.txt' % str(datetime.now())), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99
IMG_SIZE = 137
SDF_WEIGHT = 10.

HOSTNAME = socket.gethostname()
TRAIN_LISTINFO = []
TEST_LISTINFO = []

# "chair": "03001627",
# "bench": "02828884",
# "cabinet": "02933112",
# "car": "02958343",
# "airplane": "02691156",
# "display": "03211117",
# "lamp": "03636649",
# "speaker": "03691459",
# "rifle": "04090263",
# "sofa": "04256520",
# "table": "04379243",
# "phone": "04401088",
# "watercraft": "04530566"
CAT_LIST = ["02691156", "02828884", "02933112", "02958343", "03001627", "03211117", "03636649", "03691459", "04090263",
            "04256520", "04379243", "04401088", "04530566"]
# CAT_LIST = ["03636649","04090263"]
cats_limit_train = {}
cats_limit_test = {}
cat_ids=[]

for value in CAT_LIST:
    cat_ids.append(value)
    cats_limit_train[value] = 0
    cats_limit_test[value] = 0


for cat in CAT_LIST:
    TRAIN_LST = lst_dir + '/%s_train.lst' % cat
    with open(TRAIN_LST, 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            for render in range(24):
                cats_limit_train[cat] += 1
                TRAIN_LISTINFO += [(cat, line.strip(), render)]

for cat in CAT_LIST:
    VALID_LST = lst_dir + '/%s_test.lst' % cat
    with open(VALID_LST, 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            for render in range(24):
                cats_limit_test[cat] += 1
                TEST_LISTINFO += [(cat, line.strip(), render)]


info = {'rendered_dir': raw_dirs['renderedh5_dir'],
        'rendered_dir_v2': raw_dirs['renderedh5_dir_v2'],
        'sdf_dir': raw_dirs['sdf_dir'],
        'iso_value': 0.003}

TRAIN_DATASET = data.Pt_sdf_img(FLAGS, listinfo=TRAIN_LISTINFO, info=info, cats_limit=cats_limit_train)
VALID_DATASET = data.Pt_sdf_img(FLAGS, listinfo=TEST_LISTINFO, info=info, cats_limit=cats_limit_test)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 1e-6, name='lr') # CLIP THE LEARNING RATE!
    return learning_rate

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

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

def load_model(sess, LOAD_MODEL_FILE, prefixs, strict=False):

    vars_in_pretrained_model = dict(checkpoint_utils.list_variables(LOAD_MODEL_FILE))
    # print(vars_in_pretrained_model)
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
    print(vars_in_defined_model)
    saver = tf.train.Saver(vars_in_defined_model)
    try:
        saver.restore(sess, LOAD_MODEL_FILE)
        print( "Model loaded in file: %s" % (LOAD_MODEL_FILE))
    except:
        if strict:
            print( "Fail to load modelfile: %s" % LOAD_MODEL_FILE)
            return False
        else:
            print( "Fail loaded in file: %s" % (LOAD_MODEL_FILE))
            return True

    return True

def train():
    log_string(LOG_DIR)
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            input_pls = model.placeholder_inputs(BATCH_SIZE, NUM_POINTS, (IMG_SIZE, IMG_SIZE), num_pc=NUM_POINTS, num_sample_pc=NUM_SAMPLE_POINTS, scope='inputs_pl')
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0, name='batch')
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            print("--- Get model and loss")
            # Get model and loss

            end_points = model.get_model(input_pls, NUM_POINTS, is_training_pl, img_size = (IMG_SIZE, IMG_SIZE), bn=False, wd=2e-3, FLAGS=FLAGS)
            loss, end_points = model.get_loss(end_points, sdf_weight=SDF_WEIGHT, FLAGS=FLAGS)
            tf.summary.scalar('loss', loss)

            print("--- Get training operator")
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)

            # Create a session
            config = tf.ConfigProto()
            gpu_options = tf.GPUOptions()#per_process_gpu_memory_fraction=0.99)
            config=tf.ConfigProto(gpu_options=gpu_options)
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            sess = tf.Session(config=config)
            # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

            # Add summary writers
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
            test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

            ##### all
            update_variables = [x for x in tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)]

            train_op = optimizer.minimize(loss, global_step=batch, var_list=update_variables)

            # Init variables
            init = tf.global_variables_initializer()
            sess.run(init)

            ######### Loading Checkpoint ###############
            # CNN(Pretrained from ImageNet)
            # if PRETRAINED_CNN_MODEL_FILE is not '':
            #     if not load_model(sess, PRETRAINED_CNN_MODEL_FILE, 'vgg_16', strict=True):
            #         return

            if PRETRAINED_PN_MODEL_FILE is not '':
                if not load_model(sess, PRETRAINED_PN_MODEL_FILE, ['refpc_reconstruction','sdfprediction'], strict=True):
                    return 
            # Overall  
            saver = tf.train.Saver([v for v in tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES) if('lr' not in v.name) and ('batch' not in v.name)])  
            ckptstate = tf.train.get_checkpoint_state(PRETRAINED_MODEL_PATH)

            if ckptstate is not None:
                LOAD_MODEL_FILE = os.path.join(PRETRAINED_MODEL_PATH, os.path.basename(ckptstate.model_checkpoint_path))
                try:
                    load_model(sess, LOAD_MODEL_FILE, ['vgg_16'], strict=True)
                    # load_model(sess, LOAD_MODEL_FILE, ['sdfprediction','vgg_16'], strict=True)
                    with NoStdStreams():
                        saver.restore(sess, LOAD_MODEL_FILE)
                    print( "Model loaded in file: %s" % LOAD_MODEL_FILE)    
                except:
                    print( "Fail to load overall modelfile: %s" % PRETRAINED_MODEL_PATH)

            ###########################################

            ops = {'input_pls': input_pls,
                   'is_training_pl': is_training_pl,
                   'loss': loss,
                   'train_op': train_op,
                   'merged': merged,
                   'step': batch,
                   'end_points': end_points}

            best_loss = 1e20
            if FLAGS.test or FLAGS.create:
                VALID_DATASET.start()
                eval_one_epoch(sess, ops)
                VALID_DATASET.shutdown()
            else:
                TRAIN_DATASET.start()
                for epoch in range(MAX_EPOCH):
                    log_string('**** EPOCH %03d ****' % (epoch))
                    sys.stdout.flush()

                    # eval_one_epoch(sess, ops, test_writer)
                    train_one_epoch(sess, ops, train_writer, saver)
                    # epoch_loss = eval_one_epoch(sess, ops, test_writer)
                    # if epoch_loss < best_loss:
                    #     best_loss = epoch_loss
                    #     save_path = saver.save(sess, os.path.join(LOG_DIR, "best_model_epoch_%03d.ckpt"%(epoch)))
                    #     log_string("Model saved in file: %s" % save_path)

                    # # Save the variables to disk.
                    # if epoch % 1 == 0:
                    #     save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                    #     log_string("Model saved in file: %s" % save_path)
                TRAIN_DATASET.shutdown()


def pc_normalize(pc, centroid=None):

    """ pc: NxC, return NxC """
    l = pc.shape[0]

    if centroid is None:
        centroid = np.mean(pc, axis=0)

    pc = pc - centroid
    # m = np.max(pc, axis=0)
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))

    pc = pc / m

    return pc

def train_one_epoch(sess, ops, train_writer, saver):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    # Shuffle train samples
    num_batches = int(len(TRAIN_DATASET) / BATCH_SIZE)


    print('num_batches', num_batches)

    log_string(str(datetime.now()))

    loss_all = 0.
    losses = {}
    for lossname in ops['end_points']['losses'].keys():
        losses[lossname] = 0.

    tic = time.time()

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        batch_data = TRAIN_DATASET.fetch()

        feed_dict = {ops['is_training_pl']: is_training,
                     ops['input_pls']['sample_pc']: batch_data['sdf_pt'],
                     ops['input_pls']['trans_mat']: batch_data['trans_mat'],
                     ops['input_pls']['RT']: batch_data['RT'],
                     ops['input_pls']['imgs']: batch_data['img'][:,:,:,:3],
                     ops['input_pls']['shifts']: batch_data['shifts']}
        if FLAGS.rotation:
            feed_dict[ops['input_pls']['sample_pc_rot']] = batch_data['sdf_pt_rot']
        else:
            feed_dict[ops['input_pls']['sample_pc_rot']] = batch_data['sdf_pt']
        output_list = [ops['train_op'], ops['merged'], ops['step'], ops['loss'],
                       ops['end_points']['sample_img_points'], ops['end_points']['pred_sample_img_points'],
                       ops['end_points']['ref_img'], ops['end_points']['rot_homopc'], ops['end_points']['pred_rot_homopc']]

        loss_list = []
        for il, lossname in enumerate(losses.keys()):
            loss_list += [ops['end_points']['losses'][lossname]]

        outputs = sess.run(output_list + loss_list, feed_dict=feed_dict)

        _, summary, step, loss_val, \
        sample_img_points_val, pred_sample_img_points_val, ref_img_val, rot_homopc_val, pred_rot_homopc_val = outputs[:-len(losses)]

        train_writer.add_summary(summary, step)

        for il, lossname in enumerate(losses.keys()):
            losses[lossname] += outputs[len(output_list)+il]

        loss_all += losses['overall_loss']

        save_freq = 1000
        if (batch_idx + 1) % save_freq == 0:
            save_path = saver.save(sess, os.path.join(LOG_DIR, "latest.ckpt"))
            log_string("Model saved in file: %s" % save_path)

        verbose_freq = FLAGS.verbose_freq
        if (batch_idx) % verbose_freq == 0:
            bid = 0
            np.savetxt(os.path.join(VALID_RESULT_PATH, '%d_rot_homopc.xyz' % batch_idx), rot_homopc_val[bid,:,:])
            np.savetxt(os.path.join(VALID_RESULT_PATH, '%d_pred_rot_homopc.xyz' % batch_idx), pred_rot_homopc_val[bid,:,:])

            saveimg = (batch_data['img'][bid,:,:,:] * 255).astype(np.uint8)
            samplept_img = sample_img_points_val[bid,...]
            choice = np.random.randint(samplept_img.shape[0], size=10)
            samplept_img = samplept_img[choice, ...]

            pred_sample_img = pred_sample_img_points_val[bid, ...]
            pred_sample_img = pred_sample_img[choice, ...]

            for j in range(samplept_img.shape[0]):
                x = int(samplept_img[j, 0])
                y = int(samplept_img[j, 1])
                cv2.circle(saveimg, (x, y), 3, (0, 255, 0, 255), -1)

            for j in range(pred_sample_img.shape[0]):
                x = int(pred_sample_img[j, 0])
                y = int(pred_sample_img[j, 1])
                cv2.circle(saveimg, (x, y), 3, (0, 0, 255, 255), -1)
            cv2.imwrite(os.path.join(VALID_RESULT_PATH, '%s_%s_%s_comp.png' %
                                     (
                                     batch_data['cat_id'][bid], batch_data['obj_nm'][bid], batch_data['view_id'][bid])),
                        saveimg)
            outstr = ' -- %03d / %03d -- ' % (batch_idx+1, num_batches)
            for lossname in losses.keys():
                outstr += '%s: %f, ' % (lossname, losses[lossname] / verbose_freq)
                losses[lossname] = 0
            outstr += 'time: %.02f, ' % (time.time() - tic)
            tic = time.time()
            log_string(outstr)


def eval_one_epoch(sess, ops):
    """ ops: dict mapping from string to tf ops """
    
    is_training = False

    # Shuffle train samples
    num_batches = int(len(VALID_DATASET)/BATCH_SIZE)

    print('num_batches', num_batches)
    print('len(VALID_DATASET)', len(VALID_DATASET))

    pc3d_dist_lst = []
    pc2d_dist_lst = []
    losses = {}
    for lossname in ops['end_points']['losses'].keys():
        losses[lossname] = 0.

    tic = time.time()
    for batch_idx in range(num_batches):

        batch_data = VALID_DATASET.fetch()

        feed_dict = {ops['is_training_pl']: is_training,
                     ops['input_pls']['sample_pc']: batch_data['sdf_pt'],
                     ops['input_pls']['trans_mat']: batch_data['trans_mat'],
                     ops['input_pls']['RT']: batch_data['RT'],
                     ops['input_pls']['imgs']: batch_data['img'][:,:,:,:3],
                     ops['input_pls']['shifts']: batch_data['shifts']}
        if FLAGS.rotation:
            feed_dict[ops['input_pls']['sample_pc_rot']] = batch_data['sdf_pt_rot']
        else:
            feed_dict[ops['input_pls']['sample_pc_rot']] = batch_data['sdf_pt']
        output_list = [ops['loss'],
                       ops['end_points']['pred_trans_mat'],
                       ops['end_points']['pred_sample_img_points'],
                       ops['end_points']['sample_img_points'],
                       ops['end_points']['ref_img'],
                       ops['end_points']['rot_homopc'], ops['end_points']['pred_rot_homopc'],
                       ops['end_points']['results']['rot2d_dist_all'],
                       ops['end_points']['results']['rot3d_dist_all']]

        loss_list = []
        for il, lossname in enumerate(losses.keys()):
            loss_list += [ops['end_points']['losses'][lossname]]

        outputs = sess.run(output_list + loss_list, feed_dict=feed_dict)

        loss_val, pred_trans_mat_val, pred_sample_img_points_val, sample_img_points_val, ref_img_val,\
        rot_homopc_val, pred_rot_homopc_val, rot2d_dist_all_val, rot3d_dist_all_val = outputs[:-len(losses)]

        for il, lossname in enumerate(losses.keys()):
            if lossname == "rot2d_dist":
                pc2d_dist_lst.append(outputs[len(output_list)+il])
            elif lossname == "rot3d_dist":
                pc3d_dist_lst.append(outputs[len(output_list)+il])
            losses[lossname] += outputs[len(output_list)+il]

        verbose_freq = FLAGS.verbose_freq
        if (batch_idx) % verbose_freq == 0:
            log_f_name = os.path.join(TEST_RESULT_PATH, "err_log.txt")
            bids = range(BATCH_SIZE)
            for bid in bids:
                np.savetxt(os.path.join(TEST_RESULT_PATH, '%s_%s_%s_gt.xyz' %
                    (batch_data['cat_id'][bid], batch_data['obj_nm'][bid], batch_data['view_id'][bid])), rot_homopc_val[bid, :, :])
                np.savetxt(os.path.join(TEST_RESULT_PATH, '%s_%s_%s_pred.xyz' %
                    (batch_data['cat_id'][bid], batch_data['obj_nm'][bid], batch_data['view_id'][bid])),
                           pred_rot_homopc_val[bid, :, :])
                with open(log_f_name, "a") as logf:
                    logf.write("rot3d_dist: {}, rot2d_dist: {}, filename: {}_{}_{}_comp.png \n"
                               .format(rot3d_dist_all_val[bid], rot2d_dist_all_val[bid],
                               batch_data['cat_id'][bid], batch_data['obj_nm'][bid], batch_data['view_id'][bid]))
                saveimg = (batch_data['img'][bid, :, :, :] * 255).astype(np.uint8)
                # pred_saveimg = saveimg.copy()
                samplept_img = sample_img_points_val[bid, ...]
                choice = np.random.randint(samplept_img.shape[0], size=10)
                samplept_img = samplept_img[choice, ...]
                pred_sample_img = pred_sample_img_points_val[bid, ...][choice, ...]
                for j in range(samplept_img.shape[0]):
                    x = int(samplept_img[j, 0])
                    y = int(samplept_img[j, 1])
                    cv2.circle(saveimg, (x, y), 3, (0, 255, 0, 255), -1)

                for j in range(pred_sample_img.shape[0]):
                    x = int(pred_sample_img[j, 0])
                    y = int(pred_sample_img[j, 1])
                    cv2.circle(saveimg, (x, y), 3, (0, 0, 255, 255), -1)
                cv2.imwrite(os.path.join(TEST_RESULT_PATH, '%s_%s_%s_comp.png' %
                    (batch_data['cat_id'][bid], batch_data['obj_nm'][bid], batch_data['view_id'][bid])), saveimg)

                outstr = ' -- %03d / %03d -- ' % (batch_idx + 1, num_batches)
            for lossname in losses.keys():
                outstr += '%s: %f, ' % (lossname, losses[lossname] / verbose_freq)
                losses[lossname] = 0
            outstr += 'time: %.02f, ' % (time.time() - tic)
            tic = time.time()
            log_string(outstr)
        if FLAGS.create:
            create_img_h5(batch_data, pred_trans_mat_val)

    pc2d_dist_lst = np.asarray(pc2d_dist_lst)
    pc3d_dist_lst = np.asarray(pc3d_dist_lst)
    print("avg 2d dist {}, max 2d dist {}, min 2d dist {}".
          format(np.mean(pc2d_dist_lst), np.max(pc2d_dist_lst), np.min(pc2d_dist_lst)))
    print("avg 3d dist {}, max 3d dist {}, min 3d dist {}".
          format(np.mean(pc3d_dist_lst), np.max(pc3d_dist_lst), np.min(pc3d_dist_lst)))

    return 1


def create_img_h5(batch_data, transmat):
    # batch_data['pc'] = batch_pc
    # batch_data['sdf_pt'] = batch_sdf_pt
    # batch_data['sdf_pt_rot'] = batch_sdf_pt_rot
    # batch_data['sdf_val'] = batch_sdf_val
    # batch_data['norm_params'] = batch_norm_params
    # batch_data['sdf_params'] = batch_sdf_params
    # batch_data['img'] = batch_img
    # # batch_data['img_mat'] = batch_img_mat
    # # batch_data['img_pos'] = batch_img_pos
    # batch_data['trans_mat'] = batch_trans_mat
    # batch_data['cat_id'] = batch_cat_id
    # batch_data['obj_nm'] = batch_obj_nm
    # batch_data['view_id'] = batch_view_id
    for i in range(BATCH_SIZE):
        src_img_h5 = os.path.join(info['rendered_dir'], batch_data["cat_id"][i],
                               batch_data["obj_nm"][i], '{0:02d}'.format(batch_data['view_id'][i])+".h5")
        print("src:", src_img_h5)
        tar_img_dir = os.path.join(FLAGS.img_h5_dir, batch_data["cat_id"][i],
                               batch_data["obj_nm"][i])
        os.makedirs(tar_img_dir,exist_ok=True)
        tar_img_h5 = os.path.join(tar_img_dir,'{0:02d}'.format(batch_data['view_id'][i]) + ".h5")
        print("tar:", tar_img_h5)
        with h5py.File(src_img_h5, 'r') as h5_f:
            obj_rot_mat = h5_f["obj_rot_mat"][:].astype(np.float32)
            regress_mat = h5_f["regress_mat"][:].astype(np.float32)
            img_arr = h5_f["img_arr"][:].astype(np.float32)
            K = h5_f["K"][:].astype(np.float32)
            RT = h5_f["RT"][:].astype(np.float32)
            trans_mat_right = transmat[i,...]
            print("transmat[i,...].shape", transmat[i,...].shape)
            with h5py.File(tar_img_h5, 'w') as f1:
                    f1.create_dataset('img_arr', data=img_arr, compression='gzip',
                                      dtype='uint8', compression_opts=4)
                    f1.create_dataset('trans_mat', data=trans_mat_right, compression='gzip',
                                      dtype='float32', compression_opts=4)
                    f1.create_dataset('K', data=K, compression='gzip',
                                      dtype='float32', compression_opts=4)
                    f1.create_dataset('RT', data=RT, compression='gzip',
                                      dtype='float32', compression_opts=4)
                    f1.create_dataset('obj_rot_mat', data=obj_rot_mat, compression='gzip',
                                      dtype='float32', compression_opts=4)
                    f1.create_dataset('regress_mat', data=regress_mat, compression='gzip',
                                      dtype='float32', compression_opts=4)
                    print("write:", tar_img_h5)

def check_all_h5():
    # ()
    for info in TEST_LISTINFO:
        cat, obj, view = info
        src = os.path.join("/ssd1/datasets/ShapeNet/ShapeNetRenderingh5_v1",
                           cat, obj, "%02d.h5"%view)
        print(src)
        with h5py.File(src, 'r') as h5_f:
            trans_mat = h5_f["trans_mat"][:].astype(np.float32)

if __name__ == "__main__":
    try:
        log_string('pid: %s'%(str(os.getpid())))
        train()
        LOG_FOUT.close()
    except KeyboardInterrupt:
        TRAIN_DATASET.shutdown()
    # check_all_h5()

# nohup python -u cam_est/train_sdf_cam.py --test --restore_model /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/cam/checkpoint/cam_mixloss_all_2.93 --log_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/cam/checkpoint/cam_mixloss_all_2.93 --gpu 0 --loss_mode 3DM --verbose_freq 1 &> log/test_cam_mixloss_all_2.93.log &



# nohup python -u cam_est/train_sdf_cam.py --test --batch_size 1 --restore_model /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/cam/checkpoint/cam_mixloss_all_2.93 --log_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/cam/checkpoint/cam_mixloss_all_2.93 --gpu 0 --loss_mode 3DM --verbose_freq 1 &> log/test_cam_mixloss_3sample.log &
#
# nohup python -u cam_est/train_sdf_cam.py --restore_model checkpoint/cam_3D_shift_0.5 --log_dir checkpoint/cam_3D_shift_0.5 --gpu 0 --loss_mode 3D --learning_rate 1e-4 --shift --shift_weight 0.5 &> log/cam_3D_shift0.5_all.log &
#
# nohup python -u cam_est/train_sdf_cam.py --restore_model checkpoint/cam_3D_shift_0.5 --log_dir checkpoint/cam_3D_shift_1 --gpu 1 --loss_mode 3D --learning_rate 1e-4 --shift --shift_weight 1 &> log/cam_3D_shift1_all.log &
#
# nohup python -u cam_est/train_sdf_cam.py --restore_model checkpoint/cam_3D_shift_0.5 --log_dir checkpoint/cam_3D_shift_2 --gpu 2 --loss_mode 3D --learning_rate 1e-4 --shift --shift_weight 2 &> log/cam_3D_shift2_all.log &
#
# nohup python -u cam_est/train_sdf_cam.py --restore_model checkpoint/cam_3D_shift_0.5 --log_dir checkpoint/cam_3D_shift_5 --gpu 3 --loss_mode 3D --learning_rate 1e-4 --shift --shift_weight 5 &> log/cam_3D_shift5_all.log &
#
#
