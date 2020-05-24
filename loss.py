# -*- coding: utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorlayer as tl

from model import GAN_g, GAN_d



def dice_coe(output, target, loss_type='jaccard', axis=(1, 2, 3), smooth=1e-5):
    """Soft dice (Sørensen or Jaccard) coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation
    i.e. labels are binary. The coefficient between 0 to 1, 1 means totally match.

    Parameters
    -----------
    output : Tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : Tensor
        The target distribution, format the same with `output`.
    loss_type : str
        ``jaccard`` or ``sorensen``, default is ``jaccard``.
    axis : tuple of int
        All dimensions are reduced, default ``[1,2,3]``.
    smooth : float
        This small value will be added to the numerator and denominator.
            - If both output and target are empty, it makes sure dice is 1.
            - If either output or target are empty (all pixels are background), dice = ```smooth/(small_value + smooth)``, then if smooth is very small, dice close to 0 (even the image values lower than the threshold), so in this case, higher smooth can have a higher dice.

    Examples
    ---------
    >>> outputs = tl.act.pixel_wise_softmax(network.outputs)
    >>> dice_loss = 1 - tl.cost.dice_coe(outputs, y_)

    References
    -----------
    - `Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`__

    """
    inse = tf.reduce_sum(output * target, axis=axis)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(output * output, axis=axis)
        r = tf.reduce_sum(target * target, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknow loss_type")
    ## old axis=[0,1,2,3]
    # dice = 2 * (inse) / (l + r)
    # epsilon = 1e-5
    # dice = tf.clip_by_value(dice, 0, 1.0-epsilon) # if all empty, dice = 1
    ## new haodong
    dice = (2. * inse + smooth) / (l + r + smooth)
    ##
    #dice = tf.reduce_mean(dice)
    return dice


def binary_cross_entropy(output, target, label, eps=1e-8, name='bce_loss'):
    with tf.name_scope(name):
        output_2c = tf.concat([1-output, output], axis=3)
        target_2c = tf.cast(tf.concat([1-target, target], axis=3), tf.float32)
        weighted = 1 - tf.reduce_sum(target_2c, axis=(0,1,2)) / tf.reduce_sum(target_2c)
        return -tf.reduce_mean(label * tf.reduce_mean(weighted * target_2c * tf.log(tf.clip_by_value(output_2c, eps, 1.)), axis=(1,2,3)))


def seg_loss(output, target, miu_bce=1., miu_dice=1., eps=1e-8, smooth=1e-5, name='seg_loss'):
    with tf.name_scope(name):
        sigm = tf.sigmoid(output)
        label = 1
        bce = binary_cross_entropy(sigm, target, label, eps=eps)
        d_coe = tf.reduce_mean(label * dice_coe(sigm, target, smooth=smooth, axis=[1,2,3]))
        return miu_bce * bce + miu_dice * (1. - d_coe)


def infer_g_init_train(x, y_, is_train=True, reuse=None, keep_prob=0.8):
    miu_bce = 1.
    miu_dice = 1.

    # create network
    _, network = GAN_g(x, n_classes=1, is_train=is_train, dropout=keep_prob, reuse=reuse)
    y = network.outputs

    # segmentation loss
    cost = seg_loss(y, y_, miu_bce=miu_bce, miu_dice=miu_dice, eps=1e-8, smooth=1e-5)
    dice = tl.cost.dice_hard_coe(y, y_, threshold=0, axis=[1,2,3])

    return network, cost, dice


def infer_g_valid(x, y_, is_train=False, reuse=True, keep_prob=1.):
    # create network
    _, network = GAN_g(x, n_classes=1, is_train=is_train, dropout=keep_prob, reuse=reuse)
    y = network.outputs
    dice = tl.cost.dice_hard_coe(y, y_, threshold=0, axis=[1,2,3])

    return network, dice


