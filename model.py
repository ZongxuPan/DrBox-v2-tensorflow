import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl
from rbox_functions import *

class VGG16(object):
    def __init__(self, prior_num, para_num, cls_num, is_fpn, use_third_layer, BATCH_SIZE):
        self.name = 'VGG16'
        self.eps = 1e-10
        self.scale = 20.0
        self.para_num = para_num
        self.cls_num = cls_num
        self.prior_num = prior_num
        self.cls_out = cls_num + 1
        self.loc_output_num = prior_num[-1] * para_num
        self.conf_output_num = prior_num[-1] * (cls_num + 1)
        self.fpn = is_fpn
        self.use_third_layer = use_third_layer
        self.BATCH_SIZE = BATCH_SIZE

    def __call__(self, inputs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            c1_1 = tcl.conv2d(inputs, num_outputs=64, kernel_size=3, stride=1, padding='SAME', scope='conv1_1')
            c1_2 = tcl.conv2d(c1_1, num_outputs=64, kernel_size=3, stride=1, padding='SAME', scope='conv1_2')
            p1 = tcl.max_pool2d(inputs=c1_2, kernel_size=2, stride=2, padding='SAME')
            c2_1 = tcl.conv2d(p1, num_outputs=128, kernel_size=3, stride=1, padding='SAME', scope='conv2_1')
            c2_2 = tcl.conv2d(c2_1, num_outputs=128, kernel_size=3, stride=1, padding='SAME', scope='conv2_2')
            p2 = tcl.max_pool2d(inputs=c2_2, kernel_size=2, stride=2, padding='SAME')
            c3_1 = tcl.conv2d(p2, num_outputs=256, kernel_size=3, stride=1, padding='SAME', scope='conv3_1')
            c3_2 = tcl.conv2d(c3_1, num_outputs=256, kernel_size=3, stride=1, padding='SAME', scope='conv3_2')
            c3_3 = tcl.conv2d(c3_2, num_outputs=256, kernel_size=3, stride=1, padding='SAME', scope='conv3_3')
            p3 = tcl.max_pool2d(inputs=c3_3, kernel_size=2, stride=2, padding='SAME')
            c4_1 = tcl.conv2d(p3, num_outputs=512, kernel_size=3, stride=1, padding='SAME', scope='conv4_1')
            c4_2 = tcl.conv2d(c4_1, num_outputs=512, kernel_size=3, stride=1, padding='SAME', scope='conv4_2')
            c4_3 = tcl.conv2d(c4_2, num_outputs=512, kernel_size=3, stride=1, padding='SAME', scope='conv4_3')
            if self.fpn:
                p4 = tcl.max_pool2d(inputs=c4_3, kernel_size=2, stride=2, padding='SAME')
                c5_1 = tcl.conv2d(p4, num_outputs=512, kernel_size=3, stride=1, padding='SAME', scope='conv5_1')
                c5_2 = tcl.conv2d(c5_1, num_outputs=512, kernel_size=3, stride=1, padding='SAME', scope='conv5_2')
                c5_3 = tcl.conv2d(c5_2, num_outputs=512, kernel_size=3, stride=1, padding='SAME', scope='conv5_3')
                c4_3_conv = tcl.conv2d(c4_3, num_outputs=256, kernel_size=1, stride=1, padding='SAME',weights_initializer=tf.random_normal_initializer(0, 0.1),biases_initializer=tf.constant_initializer(0.1), scope='conv43_conv')
                c5_3_conv = tcl.conv2d(c5_3, num_outputs=256, kernel_size=1, stride=1, padding='SAME',weights_initializer=tf.random_normal_initializer(0, 0.1), biases_initializer=tf.constant_initializer(0.1), scope='conv53_conv')
                # c5_3_up = tcl.conv2d_transpose(c5_3, num_outputs=512, kernel_size=3, stride=2, padding='SAME', weights_initializer = tf.random_normal_initializer(0,0.1), biases_initializer=tf.constant_initializer(0.1), scope='conv53_up')
                c4_3_shape = tf.shape(c4_3_conv)
                c5_3_up = tf.image.resize_images(c5_3_conv, c4_3_shape[1:3], method=0)
                c4_3_new = tf.add(c4_3_conv, c5_3_up)
                o4 = self.normalize_channel(c4_3_new)
                loc4 = tcl.conv2d(o4, num_outputs=self.prior_num[1] * self.para_num, kernel_size=3, stride=1,padding='SAME', activation_fn=None, scope='loc4')
                conf4 = tcl.conv2d(o4, num_outputs=self.prior_num[1] * self.cls_out, kernel_size=3, stride=1,padding='SAME', activation_fn=None, scope='conf4')
                if self.use_third_layer:
                    c3_3_conv = tcl.conv2d(c3_3, num_outputs=256, kernel_size=1, stride=1, padding='SAME',weights_initializer=tf.random_normal_initializer(0, 0.1),biases_initializer=tf.constant_initializer(0.1), scope='conv33_conv')
                    c3_3_shape = tf.shape(c3_3_conv)
                    c4_3_up = tf.image.resize_images(c4_3_new, c3_3_shape[1:3], method=0)
                    c3_3_new = tf.add(c3_3_conv, c4_3_up)
                    o3 = self.normalize_channel(c3_3_new)
                    loc3 = tcl.conv2d(o3, num_outputs=self.prior_num[0] * self.para_num, kernel_size=3, stride=1, padding='SAME', activation_fn=None, scope='loc3')
                    conf3 = tcl.conv2d(o3, num_outputs=self.prior_num[0] * self.cls_out, kernel_size=3, stride=1, padding='SAME', activation_fn=None, scope='conf3')
                    for idx in range(self.BATCH_SIZE):
                        loc3_i = tf.slice(loc3, [idx, 0, 0, 0], [1, -1, -1, -1])
                        loc3_i_reshape = tf.reshape(loc3_i, [-1, self.para_num])
                        loc4_i = tf.slice(loc4, [idx, 0, 0, 0], [1, -1, -1, -1])
                        loc4_i_reshape = tf.reshape(loc4_i, [-1, self.para_num])
                        if idx == 0:
                            loc_reshape = tf.concat([loc3_i_reshape, loc4_i_reshape], axis=0)
                        else:
                            loc34_i_reshape = tf.concat([loc3_i_reshape, loc4_i_reshape], axis=0)
                            loc_reshape = tf.concat([loc_reshape, loc34_i_reshape], axis=0)
                        conf3_i = tf.slice(conf3, [idx, 0, 0, 0], [1, -1, -1, -1])
                        conf3_i_reshape = tf.reshape(conf3_i, [-1, self.cls_out])
                        conf4_i = tf.slice(conf4, [idx, 0, 0, 0], [1, -1, -1, -1])
                        conf4_i_reshape = tf.reshape(conf4_i, [-1, self.cls_out])
                        if idx == 0:
                            conf_reshape = tf.concat([conf3_i_reshape, conf4_i_reshape], axis=0)
                        else:
                            conf34_i_reshape = tf.concat([conf3_i_reshape, conf4_i_reshape], axis=0)
                            conf_reshape = tf.concat([conf_reshape, conf34_i_reshape], axis=0)
                else:
                    loc_reshape = tf.reshape(loc4, [-1, self.para_num])
                    conf_reshape = tf.reshape(conf4, [-1, self.cls_out])
            else:
                if self.use_third_layer:
                    o3 = self.normalize_channel(c3_3)
                    loc3 = tcl.conv2d(o3, num_outputs=self.prior_num[0] * self.para_num, kernel_size=3, stride=1, padding='SAME', activation_fn=None, scope='loc3')
                    conf3 = tcl.conv2d(o3, num_outputs=self.prior_num[0] * self.cls_out, kernel_size=3, stride=1, padding='SAME', activation_fn=None, scope='conf3')
                    o4 = self.normalize_channel(c4_3)
                    loc4 = tcl.conv2d(o4, num_outputs=self.loc_output_num, kernel_size=3, stride=1, padding='SAME', activation_fn=None,                         scope='loc4')
                    conf4 = tcl.conv2d(o4, num_outputs=self.conf_output_num, kernel_size=3, stride=1, padding='SAME', activation_fn=None,                      scope='conf4')
                    
                    for idx in range(self.BATCH_SIZE):
                        loc3_i = tf.slice(loc3, [idx, 0, 0, 0], [1, -1, -1, -1])
                        loc3_i_reshape = tf.reshape(loc3_i, [-1, self.para_num])
                        loc4_i = tf.slice(loc4, [idx, 0, 0, 0], [1, -1, -1, -1])
                        loc4_i_reshape = tf.reshape(loc4_i, [-1, self.para_num])
                        if idx == 0:
                            loc_reshape = tf.concat([loc3_i_reshape, loc4_i_reshape], axis=0)
                        else:
                            loc34_i_reshape = tf.concat([loc3_i_reshape, loc4_i_reshape], axis=0)
                            loc_reshape = tf.concat([loc_reshape, loc34_i_reshape], axis=0)
                        conf3_i = tf.slice(conf3, [idx, 0, 0, 0], [1, -1, -1, -1])
                        conf3_i_reshape = tf.reshape(conf3_i, [-1, self.cls_out])
                        conf4_i = tf.slice(conf4, [idx, 0, 0, 0], [1, -1, -1, -1])
                        conf4_i_reshape = tf.reshape(conf4_i, [-1, self.cls_out])
                        if idx == 0:
                            conf_reshape = tf.concat([conf3_i_reshape, conf4_i_reshape], axis=0)
                        else:
                            conf34_i_reshape = tf.concat([conf3_i_reshape, conf4_i_reshape], axis=0)
                            conf_reshape = tf.concat([conf_reshape, conf34_i_reshape], axis=0)
                            
                else:
                    o4 = self.normalize_channel(c4_3)
                    # p4 = tcl.max_pool2d(inputs=c4_3, kernel_size=2, stride=2, padding='SAME')
                    # c5_1 = tcl.conv2d(p4, num_outputs=512, kernel_size=3, stride=1, padding='SAME', scope='conv5_1')
                    # c5_2 = tcl.conv2d(c5_1, num_outputs=512, kernel_size=3, stride=1, padding='SAME', scope='conv5_2')
                    # c5_3 = tcl.conv2d(c5_2, num_outputs=512, kernel_size=3, stride=1, padding='SAME', scope='conv5_3')
                    # p5 = tcl.max_pool2d(inputs=c5_3, kernel_size=2, stride=2, padding='SAME')
                    # fc = tcl.flatten(p5)
                    # outputs = tcl.fully_connected(fc, num_outputs=21, activation_fn=None, scope='fc')
                    loc = tcl.conv2d(o4, num_outputs=self.loc_output_num, kernel_size=3, stride=1, padding='SAME', activation_fn=None, scope='loc')
                    conf = tcl.conv2d(o4, num_outputs=self.conf_output_num, kernel_size=3, stride=1, padding='SAME', activation_fn=None, scope='conf')
                    loc_reshape = tf.reshape(loc, [-1, self.para_num])
                    conf_reshape = tf.reshape(conf, [-1, self.cls_num + 1])
        return loc_reshape, conf_reshape 


    def normalize_channel(self, inputs):
        norm_data = tf.square(inputs)
        norm_data = tf.reduce_sum(norm_data, axis=3)
        norm_data = tf.sqrt(norm_data + self.eps)
        norm_data = tf.expand_dims(norm_data, axis=3)
        outputs = self.scale * tf.divide(inputs, norm_data)
        return outputs


    def regular_loss(self, lamda):
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        loss = []
        for var in var_list:
            loss += [tcl.l2_regularizer(lamda)(var)]
        return loss
    


    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


    @property
    def vars_train(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='trains')
