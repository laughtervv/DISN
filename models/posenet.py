import tensorflow as tf
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR))
sys.path.append(os.path.join(ROOT_DIR,'..'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import tf_util

# batch*n
def normalize_vector(v):
    batch=v.shape[0]

    v_mag = tf.sqrt(tf.reduce_sum(tf.square(v), axis=1, keepdims=True))
    v_mag = tf.maximum(v_mag, 1e-8)
    v = v / v_mag
    return v
       
######################
def compute_rotation_matrix_from_ortho6d(poses):
    x_raw = poses[:,0:3]#batch*3
    y_raw = poses[:,3:6]#batch*3
    x = normalize_vector(x_raw) #batch*3
    z = tf.linalg.cross(x,y_raw) #batch*3
    z = normalize_vector(z)#batch*3
    y = tf.linalg.cross(z,x) #batch*3
    print('x', x.shape, 'y', y.shape, 'z', z.shape)
    x = tf.reshape(x, [-1, 3, 1])
    y = tf.reshape(y, [-1, 3, 1])
    z = tf.reshape(z, [-1, 3, 1])
    matrix = tf.concat((x,y,z), 2) #batch*3*3
    print('matrix', matrix.shape)

    return matrix

def compute_geodesic_distance_from_two_matrices(m1, m2):
    batch=m1.shape[0]
    m = tf.matmul(m1, tf.transpose(m2, [0,2,1])) #batch*3*3
    cos = (m[:,0,0] + m[:,1,1] + m[:,2,2] - 1) / 2.
    cos = tf.minimum(cos, 1.)
    cos = tf.maximum(cos, -1.)

    theta = tf.acos(cos)

    return theta

#############
def get_cam_mat_shift(globalfeat, is_training, batch_size, bn, bn_decay, wd=None):

    with tf.variable_scope("scale") as scope:   #
        scale = tf_util.fully_connected(globalfeat, 64, bn=bn, is_training=is_training, scope='fc1', bn_decay=bn_decay)
        scale = tf_util.fully_connected(scale, 32, bn=bn, is_training=is_training, scope='fc2', bn_decay=bn_decay)
        scale = tf_util.fully_connected(scale, 1, bn=bn, is_training=is_training, scope='fc3', activation_fn=None, bn_decay=bn_decay)
        pred_scale = tf.reshape(scale, [batch_size, 1, 1]) * tf.tile(tf.expand_dims(tf.eye(3), 0), [batch_size, 1, 1])

    with tf.variable_scope("ortho6d") as scope:   #
        rotation = tf_util.fully_connected(globalfeat, 512, bn=bn, is_training=is_training, scope='fc1', bn_decay=bn_decay)
        rotation = tf_util.fully_connected(rotation, 256, bn=bn, is_training=is_training, scope='fc2', bn_decay=bn_decay)
        rotation = tf_util.fully_connected(rotation, 6, bn=bn, is_training=is_training, scope='fc3', activation_fn=None, bn_decay=bn_decay)
        pred_rotation = tf.reshape(rotation, [batch_size, 6])

    with tf.variable_scope("translation") as scope:  
        translation = tf_util.fully_connected(globalfeat, 128, bn=bn, is_training=is_training, scope='fc1', bn_decay=bn_decay)
        translation = tf_util.fully_connected(translation, 64, bn=bn, is_training=is_training, scope='fc2', bn_decay=bn_decay)
        # w_trans_init = 
        weights = tf.get_variable('fc3/weights', [64, 3],
                                  initializer=tf.truncated_normal_initializer(stddev=0.05, seed=1),
                                  dtype=tf.float32)
        biases = tf.get_variable('fc3/biases', [3],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        translation = tf.matmul(translation, weights)
        translation = tf.nn.bias_add(translation, biases)
        pred_translation = tf.reshape(translation, [batch_size, 3])
        pred_translation += tf.constant([-0.00193892, 0.00169222, 1.3949631], dtype=tf.float32)

    with tf.variable_scope("xyshift") as scope:
        pred_xyshift = tf_util.fully_connected(globalfeat, 128, bn=bn, is_training=is_training, scope='fc1', bn_decay=bn_decay)
        pred_xyshift = tf_util.fully_connected(pred_xyshift, 64, bn=bn, is_training=is_training, scope='fc2', bn_decay=bn_decay)
        pred_xyshift = tf_util.fully_connected(pred_xyshift, 2, bn=bn, is_training=is_training, scope='fc3', activation_fn=None, bn_decay=bn_decay)
    pred_translation = tf.reshape(pred_translation, [batch_size, 1, 3])
    pred_rotation_mat = compute_rotation_matrix_from_ortho6d(pred_rotation)
    pred_rotation_mat = tf.matmul(pred_scale, pred_rotation_mat)
    pred_RT = tf.concat([pred_rotation_mat, pred_translation], axis = 1)
    return pred_rotation_mat, pred_translation, pred_RT, pred_xyshift


#############
def get_cam_mat(globalfeat, is_training, batch_size, bn, bn_decay, wd=None):

    with tf.variable_scope("scale") as scope:   #
        scale = tf_util.fully_connected(globalfeat, 64, bn=bn, is_training=is_training, scope='fc1', bn_decay=bn_decay)
        scale = tf_util.fully_connected(scale, 32, bn=bn, is_training=is_training, scope='fc2', bn_decay=bn_decay)
        scale = tf_util.fully_connected(scale, 1, bn=bn, is_training=is_training, scope='fc3', activation_fn=None, bn_decay=bn_decay)
        pred_scale = tf.reshape(scale, [batch_size, 1, 1]) * tf.tile(tf.expand_dims(tf.eye(3), 0), [batch_size, 1, 1])

    with tf.variable_scope("ortho6d") as scope:   #
        rotation = tf_util.fully_connected(globalfeat, 512, bn=bn, is_training=is_training, scope='fc1', bn_decay=bn_decay)
        rotation = tf_util.fully_connected(rotation, 256, bn=bn, is_training=is_training, scope='fc2', bn_decay=bn_decay)
        rotation = tf_util.fully_connected(rotation, 6, bn=bn, is_training=is_training, scope='fc3', activation_fn=None, bn_decay=bn_decay)
        pred_rotation = tf.reshape(rotation, [batch_size, 6])

    with tf.variable_scope("translation") as scope:
        translation = tf_util.fully_connected(globalfeat, 128, bn=bn, is_training=is_training, scope='fc1', bn_decay=bn_decay)
        translation = tf_util.fully_connected(translation, 64, bn=bn, is_training=is_training, scope='fc2', bn_decay=bn_decay)
        # w_trans_init =
        weights = tf.get_variable('fc3/weights', [64, 3],
                                  initializer=tf.truncated_normal_initializer(stddev=0.05, seed=1),
                                  dtype=tf.float32)
        biases = tf.get_variable('fc3/biases', [3],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        translation = tf.matmul(translation, weights)
        translation = tf.nn.bias_add(translation, biases)
        pred_translation = tf.reshape(translation, [batch_size, 3])
        pred_translation += tf.constant([-0.00193892, 0.00169222, 1.3949631], dtype=tf.float32)

    pred_translation = tf.reshape(pred_translation, [batch_size, 1, 3])
    pred_rotation_mat = compute_rotation_matrix_from_ortho6d(pred_rotation)
    pred_rotation_mat = tf.matmul(pred_scale, pred_rotation_mat)
    pred_RT = tf.concat([pred_rotation_mat, pred_translation], axis = 1)
    return pred_rotation_mat, pred_translation, pred_RT
