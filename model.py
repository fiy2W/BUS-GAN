# -*- coding: utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import tensorlayer as tl


# initialize
w_init = tf.random_normal_initializer(stddev=0.02)
b_init = None # tf.constant_initializer(value=0.0)
g_init = tf.random_normal_initializer(1., 0.02)
lrelu = lambda x: tl.act.lrelu(x, 0.2)


def TransitionLayer(x, n_filters=None, compression=1., is_train=False, dropout=0.8, name='TransitionLayer'):
    x_nc = x.outputs.get_shape().as_list()[3]
    if n_filters == None:
        n_filters = int(x_nc*compression)

    with tf.variable_scope(name):
        x = tl.layers.BatchNormLayer(x, act=lrelu, is_train=is_train, gamma_init=g_init, name='bn')
        x = tl.layers.Conv2dLayer(x, shape=[1,1,x_nc,n_filters], strides=[1,2,2,1], padding='SAME', W_init=w_init, b_init=b_init, name='conv')
        x = tl.layers.DropoutLayer(x, keep=dropout, is_fix=True, is_train=is_train, name='drop')
        #x = MeanPool2d(x, [2,2], strides=[2,2], name='mean_pool')
        return x


def BottleNeckLayer(x, growth_rate=None, is_train=False, dropout=0.8, name='BottleNeckLayer'):
    x_nc = x.outputs.get_shape().as_list()[3]
    if growth_rate == None:
        growth_rate = x_nc
    
    with tf.variable_scope(name):
        x = tl.layers.BatchNormLayer(x, act=lrelu, is_train=is_train, gamma_init=g_init, name='bn1')
        x = tl.layers.Conv2dLayer(x, shape=[1,1,x_nc,growth_rate*4], strides=[1,1,1,1], padding='SAME', W_init=w_init, b_init=b_init, name='conv1')
        x = tl.layers.DropoutLayer(x, keep=dropout, is_fix=True, is_train=is_train, name='drop1')
        
        x = tl.layers.BatchNormLayer(x, act=lrelu, is_train=is_train, gamma_init=g_init, name='bn2')
        x = tl.layers.Conv2dLayer(x, shape=[3,3,growth_rate*4,growth_rate], strides=[1,1,1,1], padding='SAME', W_init=w_init, b_init=b_init, name='conv2')
        x = tl.layers.DropoutLayer(x, keep=dropout, is_fix=True, is_train=is_train, name='drop2')
        return x


def AtrousBottleNeckLayer(x, rate=1, growth_rate=None, is_train=False, dropout=0.8, name='AtrousBottleNeckLayer'):
    x_nc = x.outputs.get_shape().as_list()[3]
    if growth_rate == None:
        growth_rate = x_nc
    
    with tf.variable_scope(name):
        x = tl.layers.BatchNormLayer(x, act=lrelu, is_train=is_train, gamma_init=g_init, name='bn1')
        x = tl.layers.Conv2dLayer(x, shape=[1,1,x_nc,growth_rate*4], strides=[1,1,1,1], padding='SAME', W_init=w_init, b_init=b_init, name='conv')
        x = tl.layers.DropoutLayer(x, keep=dropout, is_fix=True, is_train=is_train, name='drop1')
        
        x = tl.layers.BatchNormLayer(x, act=lrelu, is_train=is_train, gamma_init=g_init, name='bn2')
        x = tl.layers.AtrousConv2dLayer(x, n_filter=growth_rate, filter_size=(3,3), rate=rate, padding='SAME', W_init=w_init, b_init=b_init, name='atrous_conv')
        x = tl.layers.DropoutLayer(x, keep=dropout, is_fix=True, is_train=is_train, name='drop2')
        return x


def DenseBlock(x, nb_layers=1, growth_rate=None, is_train=False, dropout=0.8, name='DenseBlock'):
    layers_concat = x
    with tf.variable_scope(name):
        for i in range(nb_layers):
            x = BottleNeckLayer(layers_concat, growth_rate=growth_rate, is_train=is_train, dropout=dropout, name='bottlen_{}'.format(i))
            layers_concat = tl.layers.ConcatLayer([layers_concat, x], concat_dim=3, name='concat_{}'.format(i))
        return layers_concat


def AtrousDenseBlock(x, nb_layers=1, growth_rate=None, rates=[1], is_train=False, dropout=0.8, name='AtrousDenseBlock'):
    layers_concat = x
    with tf.variable_scope(name):
        for i in range(nb_layers):
            x = AtrousBottleNeckLayer(layers_concat, rate=rates[i], growth_rate=growth_rate, is_train=is_train, dropout=dropout, name='atrous_bottlen_{}'.format(i))
            layers_concat = tl.layers.ConcatLayer([layers_concat, x], concat_dim=3, name='concat_{}'.format(i))
        return layers_concat


def bn_conv_bn(x, filters=256, k_size=3, use_atrous=False, rate=2, is_train=True, name='bn_conv_bn'):
    x_nc = x.outputs.get_shape().as_list()[3]
    with tf.variable_scope(name):
        x = tl.layers.BatchNormLayer(x, act=lrelu, is_train=is_train, gamma_init=g_init, name='bn1')
        if use_atrous == False:
            x = tl.layers.Conv2dLayer(x, shape=[k_size,k_size,x_nc,filters], strides=[1,1,1,1], padding='SAME', W_init=w_init, b_init=b_init, name='conv')
        else:
            x = tl.layers.AtrousConv2dLayer(x, n_filter=filters, filter_size=(k_size,k_size), rate=rate, padding='SAME', W_init=w_init, b_init=b_init, name='atrous_conv')
        #x = tl.layers.BatchNormLayer(x, act=lrelu, is_train=is_train, gamma_init=g_init, name='bn2')
        return x


def ASPP_block(x, rates=[6,12,18], n_filters=256, is_train=False, name='ASPP'):
    """ Atrous spatial pyramid pooling.
    """
    x_nc = x.outputs.get_shape().as_list()[3]
    with tf.variable_scope(name):
        conv_1x1 = bn_conv_bn(x, filters=n_filters, k_size=1, use_atrous=False, is_train=is_train, name='conv_1x1')
        conv_3x3_1 = bn_conv_bn(x, filters=n_filters, k_size=3, use_atrous=True, rate=rates[0], is_train=is_train, name='atrous_conv_3x3_1')
        conv_3x3_2 = bn_conv_bn(x, filters=n_filters, k_size=3, use_atrous=True, rate=rates[1], is_train=is_train, name='atrous_conv_3x3_2')
        conv_3x3_3 = bn_conv_bn(x, filters=n_filters, k_size=3, use_atrous=True, rate=rates[2], is_train=is_train, name='atrous_conv_3x3_3')

        global_feat = tl.layers.Conv2dLayer(x, shape=[1,1,x_nc,n_filters], strides=[1,1,1,1], padding='SAME', W_init=w_init, b_init=b_init, name='gf_conv')
        global_feat = tl.layers.BatchNormLayer(global_feat, act=lrelu, is_train=is_train, gamma_init=g_init, name='gf_bn')

        ASPP_out = tl.layers.ConcatLayer([conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3, global_feat], concat_dim=3, name='concat')
        return ASPP_out


def DenseDeepLab(x, nb_layers, growth_rate, n_classes=21, is_train=False, dropout=0.8, reuse=False):
    with tf.variable_scope("DenseDeepLab", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        inputs = tl.layers.InputLayer(x, name='input_layer')
        _, x_row, x_col, x_c = x.get_shape().as_list()
        box = [inputs]
        
        with tf.variable_scope('stage0'):
            net = tl.layers.Conv2dLayer(inputs, shape=[7,7,x_c,2*growth_rate], strides=[1,2,2,1], padding='SAME', W_init=w_init, b_init=b_init, name='conv')
            #net = MaxPool2d(net, [3,3], strides=[2,2], name='max_pool')
        
        for stage_idx in range(len(nb_layers)):
            net = DenseBlock(x=net, nb_layers=nb_layers[stage_idx], growth_rate=growth_rate, is_train=is_train, dropout=dropout, name='dense_{}'.format(stage_idx+1))
            box.append(net)
            net = TransitionLayer(net, compression=0.8, is_train=is_train, dropout=dropout, name='trans_{}'.format(stage_idx+1))
        
        net = AtrousDenseBlock(x=net, nb_layers=6, growth_rate=growth_rate, rates=[2,2,4,4,8,8], is_train=is_train, dropout=dropout, name='atrous_dense')
        net = ASPP_block(net, rates=[6, 12, 18], n_filters=128, is_train=is_train, name='ASPP')
        
        #print(net.outputs.get_shape().as_list())
        x_c = net.outputs.get_shape().as_list()[3]
        net = tl.layers.Conv2dLayer(net, shape=[1,1,x_c,256], strides=[1,1,1,1], padding='SAME', W_init=w_init, name='imag_pooling_conv')
        net = tl.layers.BatchNormLayer(net, act=lrelu, is_train=is_train, gamma_init=g_init, name='imag_pooling_bn')
        net = tl.layers.Conv2dLayer(net, shape=[1,1,256,n_classes], strides=[1,1,1,1], padding='SAME', W_init=w_init, name='imag_pooling_out')
        logits = tl.layers.UpSampling2dLayer(net, size=[x_row, x_col], is_scale=False, name='output')
        sigm = tf.nn.sigmoid(logits.outputs-0.5)

        return sigm, logits


def AttentionBlock(x1, x2, g, inter_c=None, name='AttentionBlock'):
    x_c = x1.outputs.get_shape().as_list()[-1]
    g_c = g.outputs.get_shape().as_list()[-1]

    if inter_c is None:
        inter_c = x_c // 2
        if inter_c == 0:
            inter_c = 1
    
    with tf.variable_scope(name):
        d_g = tl.layers.DownSampling2dLayer(g, size=x1.outputs.get_shape().as_list()[1:3], is_scale=False, name='downsampling')
        theta_x = tl.layers.Conv2dLayer(x1, shape=[1,1,x_c,inter_c], strides=[1,1,1,1], padding='SAME', W_init=w_init, b_init=b_init, name='theta_x')
        phi_g = tl.layers.Conv2dLayer(d_g, shape=[1,1,g_c,inter_c], strides=[1,1,1,1], act=tf.nn.sigmoid, padding='SAME', W_init=w_init, b_init=b_init, name='phi_g')
        f = tl.layers.ElementwiseLayer([theta_x, phi_g], combine_fn=tf.multiply, name='mul_x_g')
        f.outputs = tl.act.lrelu(f.outputs, 0.2)

        psi_f = tl.layers.Conv2dLayer(f, shape=[1,1,inter_c,x_c], strides=[1,1,1,1], padding='SAME', W_init=w_init, b_init=b_init, name='psi_f')
        y = tl.layers.ElementwiseLayer([x2, psi_f], combine_fn=tf.add, name='add_x_psif')
        return y


def DenseAttenNet(x, s, nb_layers, growth_rate, n_classes=21, is_train=False, dropout=0.8, reuse=False):
    with tf.variable_scope("DenseAttenNet", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        inputs = tl.layers.InputLayer(x, name='input_layer')
        s1_inputs = tl.layers.InputLayer(s, name='segmentation')
        s2_inputs = tl.layers.InputLayer(1-s, name='inverse')
        x_c = x.get_shape().as_list()[3]
        #box = [inputs]
        
        with tf.variable_scope('stage0'):
            net = tl.layers.Conv2dLayer(inputs, shape=[7,7,x_c,2*growth_rate], strides=[1,2,2,1], padding='SAME', W_init=w_init, name='conv')
            #net = MaxPool2d(net, [3,3], strides=[2,2], name='max_pool')

        for stage_idx in range(len(nb_layers)):
            net = DenseBlock(x=net, nb_layers=nb_layers[stage_idx], growth_rate=growth_rate, is_train=is_train, dropout=dropout, name='dense_{}'.format(stage_idx+1))

            if stage_idx == 0:
                net1 = TransitionLayer(net, compression=.25, is_train=is_train, dropout=dropout, name='trans_a1_{}'.format(stage_idx+1))
                net2 = TransitionLayer(net, compression=.25, is_train=is_train, dropout=dropout, name='trans_a2_{}'.format(stage_idx+1))
                net_a1 = AttentionBlock(net1, net1, s1_inputs, name='atten1_{}'.format(stage_idx+1))
                net_a2 = AttentionBlock(net2, net2, s2_inputs, name='atten2_{}'.format(stage_idx+1))
                net = tl.layers.ConcatLayer([net_a1, net_a2], concat_dim=3, name='concat_{}'.format(stage_idx+1))
            else:
                #box.append(net)
                net = TransitionLayer(net, compression=1., is_train=is_train, dropout=dropout, name='trans_{}'.format(stage_idx+1))
        
        #print(net.outputs.get_shape().as_list())
        _, x_w, x_h, x_c = net.outputs.get_shape().as_list()
        net = tl.layers.MeanPool2d(net, [x_w,x_h], strides=[x_w,x_h], name='global_average_pool1')
        net = tl.layers.FlattenLayer(net, name='flatten')
        logits = tl.layers.DenseLayer(net, n_units=1, act=tf.identity, W_init=w_init, name='dense')
        
        return logits


def DenseDeepLab_1(x, n_classes=21, growth_rate=32, is_train=False, dropout=0.8, reuse=None):
    nb_layers = [4, 6]
    sigm, logits = DenseDeepLab(x, nb_layers=nb_layers, growth_rate=growth_rate, n_classes=n_classes, is_train=is_train, dropout=dropout, reuse=reuse)
    return sigm, logits


def DenseAttenNet_1(x, s, growth_rate=32, is_train=False, dropout=0.8, reuse=None):
    nb_layers = [2, 4, 6, 4]
    logits = DenseAttenNet(x, s, nb_layers=nb_layers, growth_rate=growth_rate, is_train=is_train, dropout=dropout, reuse=reuse)
    return logits


def GAN_g(x, n_classes=21, growth_rate=32, is_train=False, dropout=0.8, reuse=None):
    return DenseDeepLab_1(x, n_classes=n_classes, growth_rate=growth_rate, is_train=is_train, dropout=dropout, reuse=reuse)


def GAN_d(x, s, growth_rate=32, is_train=False, dropout=0.8, reuse=None):
    return DenseAttenNet_1(x, s, growth_rate=growth_rate, is_train=is_train, dropout=dropout, reuse=reuse)
