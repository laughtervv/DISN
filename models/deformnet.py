import tensorflow as tf
import tf_util

def get_sdf_3dcnn(grid_idx, globalfeats, is_training, batch_size, num_point, bn, bn_decay, wd=None, FLAGS=None):

    globalfeats_expand = tf.reshape(globalfeats, [batch_size, 1, 1, 1, -1])
    print('globalfeats_expand', globalfeats_expand.get_shape())
    net2 = tf_util.conv3d_transpose(globalfeats_expand, 128, [2, 2, 2], stride=[2, 2, 2],
                                    bn_decay=bn_decay, bn=bn,
                                    is_training=is_training, weight_decay=wd, scope='3deconv1') # 2

    net2 = tf_util.conv3d_transpose(net2, 128, [3, 3, 3], stride=[2, 2, 2],bn_decay=bn_decay, bn=bn,
                                    is_training=is_training, weight_decay=wd, scope='3deconv2') # 4

    net2 = tf_util.conv3d_transpose(net2, 128, [3, 3, 3], stride=[2, 2, 2], bn_decay=bn_decay, bn=bn,
                                    is_training=is_training, weight_decay=wd, scope='3deconv3')  # 8

    net2 = tf_util.conv3d_transpose(net2, 64, [3, 3, 3], stride=[2, 2, 2], bn_decay=bn_decay, bn=bn,
                                    is_training=is_training, weight_decay=wd, scope='3deconv4')  # 16

    net2 = tf_util.conv3d_transpose(net2, 64, [3, 3, 3], stride=[2, 2, 2], bn_decay=bn_decay, bn=bn,
                                    is_training=is_training, weight_decay=wd, scope='3deconv5')  # 32

    net2 = tf_util.conv3d_transpose(net2, 32, [3, 3, 3], stride=[2, 2, 2], bn_decay=bn_decay, bn=bn,
                                    is_training=is_training, weight_decay=wd, padding='VALID', scope='3deconv6') # 65

    net2 = tf_util.conv3d(net2, 1, [1, 1, 1], stride=[1, 1, 1], bn_decay=bn_decay, bn=bn, activation_fn=None,
                                is_training=is_training, weight_decay=wd, padding='VALID', scope='3conv7')
    res_plus = FLAGS.sdf_res+1
    full_inter = tf.reshape(net2, (batch_size, res_plus, res_plus, res_plus))

    print("3d cnn net2 shape:", full_inter.get_shape())

    pred = tf.reshape(full_inter, [batch_size, -1, 1])
    return pred

def get_sdf_3dcnn_binary(grid_idx, globalfeats, is_training, batch_size, num_point, bn, bn_decay, wd=None, FLAGS=None):
    globalfeats_expand = tf.reshape(globalfeats, [batch_size, 1, 1, 1, -1])
    print('globalfeats_expand', globalfeats_expand.get_shape())
    net2 = tf_util.conv3d_transpose(globalfeats_expand, 128, [2, 2, 2], stride=[2, 2, 2],
                                    bn_decay=bn_decay, bn=bn,
                                    is_training=is_training, weight_decay=wd, scope='3deconv1') # 2

    net2 = tf_util.conv3d_transpose(net2, 128, [3, 3, 3], stride=[2, 2, 2],bn_decay=bn_decay, bn=bn,
                                    is_training=is_training, weight_decay=wd, scope='3deconv2') # 4

    net2 = tf_util.conv3d_transpose(net2, 128, [3, 3, 3], stride=[2, 2, 2], bn_decay=bn_decay, bn=bn,
                                    is_training=is_training, weight_decay=wd, scope='3deconv3')  # 8

    net2 = tf_util.conv3d_transpose(net2, 64, [3, 3, 3], stride=[2, 2, 2], bn_decay=bn_decay, bn=bn,
                                    is_training=is_training, weight_decay=wd, scope='3deconv4')  # 16

    net2 = tf_util.conv3d_transpose(net2, 64, [3, 3, 3], stride=[2, 2, 2], bn_decay=bn_decay, bn=bn,
                                    is_training=is_training, weight_decay=wd, scope='3deconv5')  # 32

    net2 = tf_util.conv3d_transpose(net2, 32, [3, 3, 3], stride=[2, 2, 2], bn_decay=bn_decay, bn=bn,
                                    is_training=is_training, weight_decay=wd, padding='VALID', scope='3deconv6') # 65

    net2 = tf_util.conv3d(net2, 2, [1, 1, 1], stride=[1, 1, 1], bn_decay=bn_decay, bn=bn, activation_fn=None,
                                is_training=is_training, weight_decay=wd, padding='VALID', scope='3conv7_binary')
    res_plus = FLAGS.sdf_res+1
    full_inter = tf.reshape(net2, (batch_size, res_plus, res_plus, res_plus))

    print("3d cnn net2 shape:", full_inter.get_shape())

    pred = tf.reshape(full_inter, [batch_size, -1, 2])
    return pred

def get_sdf_basic2(src_pc, globalfeats, is_training, batch_size, num_point, bn, bn_decay, wd=None):

    net2 = tf_util.conv2d(tf.expand_dims(src_pc,2), 64, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training, 
                            weight_decay=wd, scope='fold1/conv1')
    net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training, 
        weight_decay=wd, scope='fold1/conv2')
    net2 = tf_util.conv2d(net2, 512, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training, 
        weight_decay=wd, scope='fold1/conv3')

    globalfeats = tf.reshape(globalfeats, [batch_size, 1, 1, -1])
    globalfeats_expand = tf.tile(globalfeats, [1, src_pc.get_shape()[1], 1, 1])
    print( 'net2', net2.shape)
    print( 'globalfeats_expand', globalfeats_expand.shape)
    concat = tf.concat(axis=3, values=[net2, globalfeats_expand])

    net2 = tf_util.conv2d(concat, 512, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training, 
        weight_decay=wd, scope='fold2/conv1')
    net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training, 
        weight_decay=wd, scope='fold2/conv2')
    pred = tf_util.conv2d(net2, 1, [1,1], padding='VALID', stride=[1,1], activation_fn=None, bn=False, weight_decay=wd, scope='fold2/conv5')

    pred = tf.reshape(pred, [batch_size, -1, 1])

    return pred


def get_sdf_basic2_binary(src_pc, globalfeats, is_training, batch_size, num_point, bn, bn_decay, wd=None):

    net2 = tf_util.conv2d(tf.expand_dims(src_pc,2), 64, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
                            weight_decay=wd, scope='fold1/conv1')
    net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold1/conv2')
    net2 = tf_util.conv2d(net2, 512, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold1/conv3')

    globalfeats = tf.reshape(globalfeats, [batch_size, 1, 1, -1])
    globalfeats_expand = tf.tile(globalfeats, [1, src_pc.get_shape()[1], 1, 1])
    print( 'net2', net2.shape)
    print( 'globalfeats_expand', globalfeats_expand.shape)
    concat = tf.concat(axis=3, values=[net2, globalfeats_expand])

    net2 = tf_util.conv2d(concat, 512, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold2/conv1')
    net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold2/conv2')
    pred = tf_util.conv2d(net2, 2, [1,1], padding='VALID', stride=[1,1], activation_fn=None, bn=False, weight_decay=wd, scope='fold2/conv5_bi')

    pred = tf.reshape(pred, [batch_size, -1, 2])

    return pred

#############
def get_sdf_partial(src_pc, partialfeats, is_training, batch_size, num_point, bn, bn_decay, wd=None):

    net2 = tf_util.conv2d(tf.expand_dims(src_pc,2), 64, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
                            weight_decay=wd, scope='fold1/conv1')
    net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold1/conv2')
    net2 = tf_util.conv2d(net2, 512, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold1/conv3')

    print( 'net2', net2.shape)
    partialfeats = tf.expand_dims(partialfeats, axis=2)
    print( 'partialfeats', partialfeats.shape)
    concat = tf.concat(axis=3, values=[net2, partialfeats])

    net2 = tf_util.conv2d(concat, 512, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold2/conv1')
    net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold2/conv2')
    pred = tf_util.conv2d(net2, 1, [1,1], padding='VALID', stride=[1,1], activation_fn=None, bn=False, weight_decay=wd, scope='fold2/conv5')

    pred = tf.reshape(pred, [batch_size, -1, 1])

    return pred

def get_sdf_partial_binary(src_pc, partialfeats, is_training, batch_size, num_point, bn, bn_decay, wd=None):

    net2 = tf_util.conv2d(tf.expand_dims(src_pc,2), 64, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
                            weight_decay=wd, scope='fold1/conv1')
    net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold1/conv2')
    net2 = tf_util.conv2d(net2, 512, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold1/conv3')

    print( 'net2', net2.shape)
    partialfeats = tf.expand_dims(partialfeats, axis=2)
    print( 'partialfeats', partialfeats.shape)
    concat = tf.concat(axis=3, values=[net2, partialfeats])

    net2 = tf_util.conv2d(concat, 512, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold2/conv1')
    net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold2/conv2')
    pred = tf_util.conv2d(net2, 2, [1,1], padding='VALID', stride=[1,1], activation_fn=None, bn=False, weight_decay=wd, scope='fold2/conv5_bi')

    pred = tf.reshape(pred, [batch_size, -1, 2])

    return pred

def get_sdf_basic2_imgfeat(src_pc, globalfeats, point_feat, is_training, batch_size, num_point, bn, bn_decay, wd=None):

    net2 = tf_util.conv2d(tf.expand_dims(src_pc,2), 64, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
                            weight_decay=wd, scope='fold1/conv1')
    net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold1/conv2')
    net2 = tf_util.conv2d(net2, 512, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold1/conv3')

    globalfeats = tf.reshape(globalfeats, [batch_size, 1, 1, -1])
    globalfeats_expand = tf.tile(globalfeats, [1, src_pc.get_shape()[1], 1, 1])
    print('net2', net2.shape)
    print('globalfeats_expand', globalfeats_expand.shape)
    concat = tf.concat(axis=3, values=[net2, globalfeats_expand, point_feat])

    net2 = tf_util.conv2d(concat, 512, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold2/conv1')
    net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold2/conv2')
    pred = tf_util.conv2d(net2, 1, [1,1], padding='VALID', stride=[1,1], activation_fn=None, bn=False, weight_decay=wd, scope='fold2/conv5')

    pred = tf.reshape(pred, [batch_size, -1, 1])

    return pred

def get_sdf_basic2_imgfeat_binary(src_pc, globalfeats, point_feat, is_training, batch_size, num_point, bn, bn_decay, wd=None):

    net2 = tf_util.conv2d(tf.expand_dims(src_pc,2), 64, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
                            weight_decay=wd, scope='fold1/conv1')
    net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold1/conv2')
    net2 = tf_util.conv2d(net2, 512, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold1/conv3')

    globalfeats = tf.reshape(globalfeats, [batch_size, 1, 1, -1])
    globalfeats_expand = tf.tile(globalfeats, [1, src_pc.get_shape()[1], 1, 1])
    print('net2', net2.shape)
    print('globalfeats_expand', globalfeats_expand.shape)
    concat = tf.concat(axis=3, values=[net2, globalfeats_expand, point_feat])

    net2 = tf_util.conv2d(concat, 512, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold2/conv1')
    net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold2/conv2')
    pred = tf_util.conv2d(net2, 2, [1,1], padding='VALID', stride=[1,1], activation_fn=None, bn=False, weight_decay=wd, scope='fold2/conv5_bi')

    pred = tf.reshape(pred, [batch_size, -1, 2])

    return pred

def get_sdf_basic2_imgfeat_twostream(src_pc, point_feat, is_training, batch_size, num_point, bn, bn_decay, wd=None):

    net2 = tf_util.conv2d(tf.expand_dims(src_pc,2), 64, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
                            weight_decay=wd, scope='fold1/conv1')
    net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold1/conv2')
    net2 = tf_util.conv2d(net2, 512, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold1/conv3')

    concat = tf.concat(axis=3, values=[net2, point_feat])

    net2 = tf_util.conv2d(concat, 512, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold2/conv1')
    net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold2/conv2')
    pred = tf_util.conv2d(net2, 1, [1,1], padding='VALID', stride=[1,1], activation_fn=None, bn=False, weight_decay=wd, scope='fold2/conv5')

    pred = tf.reshape(pred, [batch_size, -1, 1])

    return pred

def get_sdf_basic2_imgfeat_twostream_binary(src_pc, point_feat, is_training, batch_size, num_point, bn, bn_decay, wd=None):

    net2 = tf_util.conv2d(tf.expand_dims(src_pc,2), 64, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
                            weight_decay=wd, scope='fold1/conv1')
    net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold1/conv2')
    net2 = tf_util.conv2d(net2, 512, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold1/conv3')

    concat = tf.concat(axis=3, values=[net2, point_feat])

    net2 = tf_util.conv2d(concat, 512, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold2/conv1')
    net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold2/conv2')
    pred = tf_util.conv2d(net2, 2, [1,1], padding='VALID', stride=[1,1], activation_fn=None, bn=False, weight_decay=wd, scope='fold2/conv5_bi')

    pred = tf.reshape(pred, [batch_size, -1, 2])

    return pred


#
#
# #############
# def get_sdf_basic3(src_pc, globalfeats, is_training, batch_size, num_point, bn, bn_decay, wd=None):
#
#     net2 = tf_util.conv2d(tf.expand_dims(src_pc,2), 64, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, weight_decay=wd, is_training=is_training, scope='fold1/conv1')
#     net2 = tf_util.conv2d(net2, 128, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training, weight_decay=wd, scope='fold1/conv2')
#     net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training, weight_decay=wd, scope='fold1/conv3')
#
#     globalfeats = tf_util.fully_connected(globalfeats, 256, bn=bn, is_training=is_training, scope='global_fc', weight_decay=wd, bn_decay=bn_decay)
#     globalfeats = tf.reshape(globalfeats, [batch_size, 1, 1, -1])
#     globalfeats_expand = tf.tile(globalfeats, [1, src_pc.get_shape()[1], 1, 1])
#     print( 'net2', net2.shape)
#     print( 'globalfeats_expand', globalfeats_expand.shape)
#     # concat = tf.concat(axis=3, values=[net2, globalfeats_expand])
#     concat = net2 + globalfeats_expand
#
#     net2 = tf_util.conv2d(concat, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, weight_decay=wd, is_training=is_training, scope='fold2/conv1')
#     net2 = tf_util.conv2d(net2, 128, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, weight_decay=wd, is_training=is_training, scope='fold2/conv2')
#     pred = tf_util.conv2d(net2, 1, [1,1], padding='VALID', stride=[1,1], activation_fn=None, bn=False, weight_decay=wd, scope='fold2/conv5')
#
#     pred = tf.reshape(pred, [batch_size, -1, 1])
#
#     return pred
#
# #############
# def get_sdf_basic4(src_pc, globalfeats, is_training, batch_size, num_point, bn, bn_decay, wd=None):
#
#     net2 = tf_util.conv2d(tf.expand_dims(src_pc,2), 64, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training, weight_decay=wd, scope='fold1/conv1')
#     net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training, weight_decay=wd, scope='fold1/conv2')
#     net2 = tf_util.conv2d(net2, 512, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training, weight_decay=wd, scope='fold1/conv3')
#
#     globalfeats = tf.reshape(globalfeats, [batch_size, 1, 1, -1])
#     globalfeats_expand = tf.tile(globalfeats, [1, src_pc.get_shape()[1], 1, 1])
#     print('net2', net2.shape)
#     print('globalfeats_expand', globalfeats_expand.shape)
#     concat = tf.concat(axis=3, values=[net2, globalfeats_expand])
#
#     net2 = tf_util.conv2d(concat, 512, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
#         weight_decay=wd, scope='fold2/conv1')
#     net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
#         weight_decay=wd, scope='fold2/conv2')
#     pred = tf_util.conv2d(net2, 1, [1,1], padding='VALID', stride=[1,1], activation_fn=None, bn=False, weight_decay=wd, scope='fold2/conv5')
#
#     pred = tf.reshape(pred, [batch_size, -1, 1])
#
#     return pred
#
# def deepsdf_decoder(net, batch_size, is_training, bn=True, bn_decay=None, scope = ''):
#
#     net2 = tf_util.conv2d(net, 512, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training, scope=scope+'decoder/conv1')
#     net2 = tf_util.conv2d(net2, 512, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training, scope=scope+'decoder/conv2')
#     net2 = tf_util.conv2d(net2, 512, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training, scope=scope+'decoder/conv3')
#
#     return net2
#
# def get_sdf_deepsdf(src_pc, globalfeats, is_training, batch_size, num_point, bn, bn_decay):
#     ##TODO: Symmetry
#
#     # globalfeats = tf.concat([src_feats, ref_feats], axis=1)
#
#     globalfeats = tf.reshape(globalfeats, [batch_size, 1, 1, -1])
#     globalfeats_expand = tf.tile(globalfeats, [1, src_pc.get_shape()[1], 1, 1])
#
#     concat = tf.concat(axis=3, values=[tf.expand_dims(src_pc,2), globalfeats_expand])
#     displacement = deepsdf_decoder(concat, batch_size, is_training, bn=bn, bn_decay=bn_decay)
#     displacement = tf.reshape(displacement, [batch_size, -1, 512])
#
#     concat = tf.concat(axis=3, values=[tf.expand_dims(displacement,2), globalfeats_expand])
#     net = deepsdf_decoder(concat, batch_size, is_training, bn=bn, bn_decay=bn_decay, scope='fold2/')
#
#     net = tf_util.conv2d(net, 1, [1,1], padding='VALID', stride=[1,1], activation_fn=None, bn=bn, scope='deepsdf/fc')
#     pred = tf.nn.tanh(net)
#     pred = tf.reshape(pred, [batch_size, -1, 1])
#
#     return pred#, displacement

