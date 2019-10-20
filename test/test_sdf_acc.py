import argparse
from datetime import datetime
import tensorflow as tf
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR) # model
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'data'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'preprocessing'))
import model_normalization as model
import data_sdf_h5_queue # as data
import create_file_lst
slim = tf.contrib.slim

parser = argparse.ArgumentParser()
parser.add_argument('--max_epoch', type=int, default=1, help='Epoch to run [default: 201]')
parser.add_argument('--img_h', type=int, default=137, help='Image Height')
parser.add_argument('--img_w', type=int, default=137, help='Image Width')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate [default: 0.001]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.9, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--num_classes', type=int, default=1024, help='vgg global embedding dimensions')
parser.add_argument('--num_points', type=int, default=1, help='Point Number [default: 2048]')
parser.add_argument('--mask_weight', type=float, default=4.0)
parser.add_argument('--alpha', action='store_true')
parser.add_argument('--rot', action='store_true')
parser.add_argument('--tanh', action='store_true')
parser.add_argument('--cat_limit', type=int, default=168000, help="balance each category, 1500 * 24 = 36000")

parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 32]')
parser.add_argument('--sdf_res', type=int, default=64, help='sdf grid')
parser.add_argument('--log_dir', default='checkpoint/exp_200', help='Log dir [default: log]')
parser.add_argument('--test_lst_dir', default='', help='test mesh data list')
parser.add_argument('--num_sample_points', type=int, default=2048, help='Sample Point Number for each obj to test[default: 2048]')
parser.add_argument('--threedcnn', action='store_true')
parser.add_argument('--img_feat', action='store_true')
parser.add_argument('--img_feat_far', action='store_true')
parser.add_argument('--img_feat_twostream', action='store_true')
parser.add_argument('--category', default="all", help='Which single class to train on [default: None]')
parser.add_argument('--binary', action='store_true')
parser.add_argument('--create_obj', action='store_true', help="create_obj or test accuracy on test set")
parser.add_argument('--store', action='store_true')
parser.add_argument('--multi_view', action='store_true')
parser.add_argument('--cam_est', action='store_true')
parser.add_argument('--backcolorwhite', action='store_true')

FLAGS = parser.parse_args()
print('pid: %s'%(str(os.getpid())))
print(FLAGS)

EPOCH_CNT = 0
NUM_POINTS = FLAGS.num_points
BATCH_SIZE = FLAGS.batch_size
NUM_SAMPLE_POINTS = FLAGS.num_sample_points
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


########### load test lst info
TEST_LISTINFO = []
lst_dir, cats, all_cats, raw_dirs = create_file_lst.get_all_info()

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
            render_list = range(24)
            for render in render_list:
                cats_limit[cat_id]+=1
                TEST_LISTINFO += [(cat_id, line.strip(), render)]

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


if FLAGS.threedcnn:
    info = {'rendered_dir': '/ssd1/datasets/ShapeNet/ShapeNetRenderingh5_v2',
            'sdf_dir': '/ssd1/datasets/ShapeNet/SDF_full/64_expr_1.2'}
elif FLAGS.img_feat or FLAGS.img_feat_far or FLAGS.img_feat_twostream:
    info = {'rendered_dir': '/ssd1/datasets/ShapeNet/ShapeNetRenderingh5_v1',
            'sdf_dir': '/ssd1/datasets/ShapeNet/SDF_v1/256_expr_1.2_bw_0.1'}
    if FLAGS.cam_est:
        info = {'rendered_dir': '/ssd1/datasets/ShapeNet/ShapeNetRenderingh5_v1_pred_3d',
                'sdf_dir': '/ssd1/datasets/ShapeNet/SDF_v1/256_expr_1.2_bw_0.1'}
else:
    info = {'rendered_dir': '/ssd1/datasets/ShapeNet/ShapeNetRenderingh5_v2',
                'sdf_dir': '/ssd1/datasets/ShapeNet/SDF_neg/simp_256_expr_1.2_bw_0.1'}
TEST_DATASET = data_sdf_h5_queue.Pt_sdf_img(FLAGS, listinfo=TEST_LISTINFO, info=info, cats_limit=cats_limit)
print(info)

def test():
    log_string(LOG_DIR)

    input_pls = model.placeholder_inputs(BATCH_SIZE, NUM_POINTS, (IMG_SIZE, IMG_SIZE),
                        num_sample_pc=NUM_SAMPLE_POINTS, scope='inputs_pl', FLAGS=FLAGS)
    is_training_pl = tf.placeholder(tf.bool, shape=())
    print(is_training_pl)
    batch = tf.Variable(0, name='batch')

    print("--- Get model and loss")
    # Get model and loss

    end_points = model.get_model(input_pls, NUM_POINTS, is_training_pl, bn=False,FLAGS=FLAGS)

    loss, end_points = model.get_loss(end_points, sdf_weight=SDF_WEIGHT, mask_weight = FLAGS.mask_weight,
                                                        num_sample_points=NUM_SAMPLE_POINTS, FLAGS=FLAGS)
    # Create a session
    gpu_options = tf.GPUOptions()
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
    is_training = False

    # Shuffle train samples
    num_batches = int(len(TEST_DATASET)) // FLAGS.batch_size
    print()
    print('num_batches', num_batches)
    log_string(str(datetime.now()))
    losses = {}
    for lossname in ops['end_points']['losses'].keys():
        losses[lossname] = 0

    for batch_idx in range(num_batches):
        batch_data = TEST_DATASET.fetch()

        feed_dict = {ops['is_training_pl']: is_training,
                     ops['input_pls']['sample_pc']: batch_data['sdf_pt'],
                     ops['input_pls']['sample_pc_rot']: batch_data['sdf_pt_rot'],
                     ops['input_pls']['sdf']: batch_data['sdf_val'] - 0.003,
                     ops['input_pls']['imgs']: batch_data['img'],
                     ops['input_pls']['trans_mat']: batch_data['trans_mat']}
        output_list = [ops['end_points']['pred_sdf'], ops['end_points']['ref_img'],
                       ops['end_points']['sample_img_points']]
        loss_list = []
        for il, lossname in enumerate(losses.keys()):
            loss_list += [ops['end_points']['losses'][lossname]]
        outputs = sess.run(output_list + loss_list, feed_dict=feed_dict)
        pred_sdf_val, ref_img_val, sample_img_points_val = outputs[:-len(losses)]
        outstr = ' -- %03d / %03d -- ' % (batch_idx + 1, num_batches)
        for il, lossname in enumerate(losses.keys()):
            losses[lossname] += outputs[len(output_list) + il]
            outstr += '%s: %f, ' % (lossname, outputs[len(output_list) + il])
        log_string(outstr)
    # summary
    outstr = "Summary: "
    for lossname in losses.keys():
        outstr += '%s: %f, ' % (lossname, losses[lossname] / num_batches)
    log_string(outstr)

if __name__ == "__main__":

    # test accuracy
    test()

