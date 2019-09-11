import tensorflow as tf
import numpy as np
import os
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.nets import vgg
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR))
sys.path.append(os.path.join(ROOT_DIR,'..'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
import posenet

def placeholder_inputs(batch_size, num_points, img_size, num_pc=2018, num_sample_pc = 2048*8, scope=''):

    with tf.variable_scope(scope) as sc:
        pc_pl = tf.placeholder(tf.float32, shape=(batch_size, num_pc, 3))
        sample_pc_pl = tf.placeholder(tf.float32, shape=(batch_size, num_sample_pc, 3))
        sample_pc_rot_pl = tf.placeholder(tf.float32, shape=(batch_size, num_sample_pc, 3))
        imgs_pl = tf.placeholder(tf.float32, shape=(batch_size, img_size[0], img_size[1], 3))
        sdf_value_pl = tf.placeholder(tf.float32, shape=(batch_size, num_sample_pc, 1))
        sdf_sign_pl = tf.placeholder(tf.int32, shape=(batch_size, num_sample_pc))
        trans_mat_pl = tf.placeholder(tf.float32, shape=(batch_size, 4, 3))
        RT_mat_pl = tf.placeholder(tf.float32, shape=(batch_size, 4, 3))
        shifts_pl = tf.placeholder(tf.float32, shape=(batch_size, 2))

        # camera intrinsic matrix
        K = np.array([[149.84375, 0., 68.5],[0., 149.84375, 68.5],[0., 0., 1.]], dtype=np.float32)#.T
        K_pl = tf.constant(K)
        K_pl = tf.expand_dims(K_pl, 0)    # Convert to a len(yp) x 1 matrix.
        K_pl = tf.tile(K_pl, [batch_size, 1, 1])  # Create multiple columns.

    sdf = {}
    sdf['pc'] = pc_pl
    sdf['sample_pc'] = sample_pc_pl
    sdf['sample_pc_rot'] = sample_pc_rot_pl
    sdf['imgs'] = imgs_pl
    sdf['sdf_value'] = sdf_value_pl
    sdf['sdf_sign'] = sdf_sign_pl
    sdf['trans_mat'] = trans_mat_pl
    sdf['RT'] = RT_mat_pl
    sdf['K'] = K_pl
    sdf['shifts'] = shifts_pl

    return sdf

def get_model(ref_dict, num_point, is_training, bn=False, bn_decay=None, img_size = (137,137), wd=1e-5, FLAGS=None):

    ref_img = ref_dict['imgs']
    ref_pc = ref_dict['pc']
    ref_sample_pc = ref_dict['sample_pc']
    ref_sample_pc_rot = ref_dict['sample_pc_rot']
    trans_mat = ref_dict['trans_mat']
    K = ref_dict['K']
    RT = ref_dict['RT']
    gt_xyshift = ref_dict['shifts']

    batch_size = ref_img.get_shape()[0].value

    # endpoints
    end_points = {}
    end_points['ref_pc'] = ref_pc
    end_points['RT'] = RT
    end_points['K'] = K
    end_points['gt_xyshift'] = gt_xyshift
    end_points['trans_mat'] = trans_mat
    end_points['sample_pc'] = ref_sample_pc #* 10

    # Image extract features
    if ref_img.shape[1] != 224 or ref_img.shape[2] != 224:
        ref_img = tf.image.resize_bilinear(ref_img, [224, 224])
    end_points['ref_img'] = ref_img

    # vgg.vgg_16.default_image_size = (224, 224)
    with slim.arg_scope([slim.conv2d],
                         weights_regularizer=slim.l2_regularizer(wd)):
        ref_feats_embedding, vgg_end_points = vgg.vgg_16(ref_img, num_classes=1024, is_training=False, scope='vgg_16', spatial_squeeze=False)
        ref_feats_embedding_cnn = tf.squeeze(ref_feats_embedding, axis = [1,2]) 
    end_points['embedding'] = ref_feats_embedding_cnn
    print(vgg_end_points.keys())

    with tf.variable_scope("cameraprediction") as scope:
        if FLAGS.shift:
            pred_rotation, pred_translation, pred_RT, pred_xyshift = posenet.get_cam_mat_shft(ref_feats_embedding_cnn, is_training, batch_size, bn, bn_decay, wd)
            end_points['pred_rotation'] = pred_rotation
            end_points['pred_translation'] = pred_translation
            end_points['pred_RT'] = pred_RT
            end_points['pred_xyshift'] = pred_xyshift
        else:
            pred_rotation, pred_translation, pred_RT = posenet.get_cam_mat(ref_feats_embedding_cnn, is_training, batch_size, bn, bn_decay, wd)
            end_points['pred_rotation'] = pred_rotation
            end_points['pred_translation'] = pred_translation
            end_points['pred_RT'] = pred_RT
            end_points['pred_xyshift'] = None
            pred_xyshift = None

    print('trans_mat', trans_mat.shape)
    sample_img_points, gt_xy = get_img_points(ref_sample_pc, trans_mat, gt_xyshift, FLAGS)
    end_points['sample_img_points'] = sample_img_points
    end_points['gt_xy'] = gt_xy

    K_transpose = tf.transpose(K, perm=[0, 2, 1])
    pred_trans_mat = tf.matmul(pred_RT, K_transpose)
    pred_sample_img_points, pred_xy = get_img_points(ref_sample_pc, pred_trans_mat, pred_xyshift, FLAGS)
    end_points['pred_sample_img_points'] = pred_sample_img_points
    end_points['pred_trans_mat'] = pred_trans_mat
    end_points['pred_xy'] = pred_xy
    print("gt_xy, pred_xy", gt_xy.get_shape(), pred_xy.get_shape())
    return end_points

def get_img_points(sample_pc, trans_mat_right, pred_xyshift, FLAGS):
    # sample_pc B*N*3
    size_lst = sample_pc.get_shape().as_list()
    homo_pc = tf.concat((sample_pc, tf.ones((size_lst[0], size_lst[1], 1),dtype=np.float32)),axis= -1)
    print("homo_pc.get_shape()", homo_pc.get_shape())
    pc_xyz = tf.matmul(homo_pc, trans_mat_right)
    print("pc_xyz.get_shape()", pc_xyz.get_shape()) # B * N * 3
    pc_xy = tf.cast(tf.divide(pc_xyz[:,:,:2], tf.expand_dims(pc_xyz[:,:,2], axis = 2)), dtype=tf.float32)
    if FLAGS.shift:
        pc_xy = pc_xy + tf.tile(tf.expand_dims(pred_xyshift / 2 * FLAGS.img_h, axis=1), (1,FLAGS.num_points,1))
    mintensor = tf.constant([0.0,0.0], dtype=tf.float32)
    maxtensor = tf.constant([136.0,136.0], dtype=tf.float32)
    return tf.minimum(maxtensor, tf.maximum(mintensor, pc_xy)), pc_xy

def get_loss(end_points, sdf_weight=10., regularization=True, FLAGS=None):
    """ sigmoid loss+sdf value"""

    sample_pc = end_points['sample_pc']
    RT = end_points['RT']
    pred_RT = end_points['pred_RT']
    sample_img_points = end_points['sample_img_points']
    pred_sample_img_points = end_points['pred_sample_img_points']
    pred_xy = end_points['pred_xy']
    pred_xyshift = end_points['pred_xyshift']
    gt_xyshift = end_points['gt_xyshift']
    gt_xy = end_points['gt_xy']
    pred_trans_mat = end_points['pred_trans_mat']
    trans_mat = end_points['trans_mat']
    loss = 0.
    # K = end_points['K']
    # pred_rotation = end_points['pred_rotation']
    # pred_translation = end_points['pred_translation']
    # trans_cam = end_points['trans_mat']

    ################
    # Compute loss #
    ################
    end_points['losses'] = {}
    end_points['results'] = {}

    ########### camera loss
    ##point cloud rotation error
    size_lst = sample_pc.get_shape().as_list()
    homo_sample_pc = tf.concat((sample_pc, tf.ones((size_lst[0], size_lst[1], 1),dtype=np.float32)),axis= -1)
    sub_3d = tf.matmul(homo_sample_pc, pred_RT) - tf.matmul(homo_sample_pc, RT)
    rotpc_loss = tf.reduce_mean(tf.nn.l2_loss(sub_3d))
    rot2d_loss = tf.reduce_mean(tf.nn.l2_loss(pred_xy - gt_xy)) / 10000.
    rot2d_dist_all = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(sample_img_points - pred_sample_img_points), axis = -1)), axis=1)
    rot2d_dist = tf.reduce_mean(rot2d_dist_all)
    rot3d_dist_all = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(sub_3d), axis = -1)), axis=1)
    rot3d_dist = tf.reduce_mean(rot3d_dist_all)
    rotmatrix_loss = tf.reduce_mean((tf.square((pred_trans_mat-trans_mat))))
    end_points['rot_homopc'] = tf.matmul(homo_sample_pc, RT)
    end_points['pred_rot_homopc'] = tf.matmul(homo_sample_pc, pred_RT)
    rotpc_loss = rotpc_loss #* 100
    end_points['losses']['rotpc_loss'] = rotpc_loss
    end_points['losses']['rot2d_loss'] = rot2d_loss
    end_points['losses']['rot3d_dist'] = rot3d_dist
    end_points['losses']['rot2d_dist'] = rot2d_dist
    end_points['losses']['rotmatrix_loss'] = rotmatrix_loss
    end_points['results']['rot2d_dist_all'] = rot2d_dist_all
    end_points['results']['rot3d_dist_all'] = rot3d_dist_all
    if FLAGS.loss_mode == "3D":
        loss += rotpc_loss
    elif FLAGS.loss_mode == "2D":
        loss += rot2d_loss
    elif FLAGS.loss_mode == "3DM":
        loss += rotpc_loss + rotmatrix_loss * 0.3
    else:
        loss += rot2d_loss + rotpc_loss + rotmatrix_loss
    if FLAGS.shift:
        shiftxy_loss = tf.reduce_mean(tf.nn.l2_loss(gt_xyshift - pred_xyshift))
        end_points['losses']['shiftxy_loss'] = shiftxy_loss
        loss+= shiftxy_loss * FLAGS.shift_weight
    # ## rotation geodesic distance loss
    # geodist = posenet.compute_geodesic_distance_from_two_matrices(pred_rotation, RT[:,:3,:])
    # geodist_loss = tf.reduce_mean(tf.nn.l2_loss(geodist))
    # geodist_loss = geodist_loss
    # end_points['losses']['geodist_loss'] = geodist_loss
    # loss += geodist_loss

    # ## rotation mat loss
    # rot_loss = tf.reduce_mean(tf.nn.l2_loss(pred_rotation - RT[:,:3,:]))
    # rot_loss = 100 * rot_loss
    # end_points['losses']['rot_loss'] = rot_loss
    # loss += rot_loss

    # ## rotation mat differencel loss
    # rot_mat_diff = tf.matmul(pred_rotation, tf.transpose(RT[:,:3,:], perm=[0,2,1]))
    # end_points['rot_mat_diff'] = rot_mat_diff
    # rot_mat_diff -= tf.constant(np.eye(3), dtype=tf.float32)
    # rot_mat_diff_loss = tf.reduce_mean(tf.nn.l2_loss(rot_mat_diff))
    # rot_mat_diff_loss = 100 * rot_mat_diff_loss
    # end_points['losses']['rot_mat_diff'] = rot_mat_diff_loss
    # loss += rot_mat_diff_loss

    # trans_loss = tf.reduce_mean(tf.abs(pred_translation - RT[:,3,:]))
    # trans_loss = 100 * trans_loss
    # end_points['losses']['trans_loss'] = trans_loss
    # loss += trans_loss

    # cam_loss = tf.reduce_mean(tf.abs(pred_RT - RT))
    # cam_loss = 100 * cam_loss

    # end_points['losses']['rot_loss'] = rot_loss
    # end_points['losses']['cam_loss'] = cam_loss
    # cam_loss = rot_mat_diff + trans_loss
    # loss += cam_loss

    # cam_loss = tf.reduce_mean(tf.abs(pred_cam - RT))
    # end_points['losses']['cam_loss'] = cam_loss
    # loss += cam_loss

    # mat_diff = tf.matmul(pred_rotation, tf.transpose(pred_rotation, perm=[0,2,1]))
    # mat_diff -= tf.constant(np.eye(3), dtype=tf.float32)
    # mat_diff_loss = tf.nn.l2_loss(mat_diff) 
    # end_points['losses']['mat_diff_loss'] = mat_diff_loss
    # loss += mat_diff_loss

    ############### weight decay
    if regularization:
        vgg_regularization_loss = tf.add_n(slim.losses.get_regularization_losses())
        # decoder_regularization_loss = tf.add_n(tf.get_collection('regularizer'))
        end_points['losses']['regularization'] = vgg_regularization_loss#(vgg_regularization_loss + decoder_regularization_loss)
        loss += vgg_regularization_loss#(vgg_regularization_loss + decoder_regularization_loss)

    end_points['losses']['overall_loss'] = loss

    return loss, end_points

