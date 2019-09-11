import tensorflow as tf 
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.contrib.framework.python.framework import checkpoint_utils
import cv2

slim = tf.contrib.slim
resnet_v1 = resnet_v1

if __name__ == "__main__":
    #print ("unitest for resnet")
    batch_size = 10
    img_size = 256
    img = cv2.imread('/mnt/ilcompf8d0/user/weiyuewa/sources/pipeline1/tf_neural_renderer/img.png')

    # with tf.Session('') as sess:

    with tf.device('/gpu:0'):
        inputbatch = tf.expand_dims(tf.constant(img, dtype=tf.float32), axis=0)#tf.zeros([batch_size, img_size, img_size, 3])

        logits, endpoints = resnet_v1.resnet_v1_50(inputbatch, 1000, is_training=False)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)
        variables_to_restore = []

        a = [name for name, _ in checkpoint_utils.list_variables('pretrained_model/resnet_v1_50.ckpt')]
        # print a
        for var in slim.get_model_variables():
            if (var.op.name.startswith('resnet_v1_50')) and (var.op.name in a) and ('logits' not in var.op.name):
                variables_to_restore.append(var)
        # print variables_to_restore
        # slim.assign_from_checkpoint_fn('pretrained_model/resnet_v1_50.ckpt', variables_to_restore, ignore_missing_vars=False)
        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver(variables_to_restore)
        saver.restore(sess, 'pretrained_model/resnet_v1_50.ckpt')
        # print a.keys()

        cls_val = sess.run(logits)
        print cls_val

    # print logits
    print endpoints.keys()