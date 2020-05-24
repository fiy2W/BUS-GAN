# -*- coding: utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorlayer as tl

import os
import scipy
import numpy as np
from skimage import measure


def load_csv_list(path):
    with open(path, 'r') as f:
        f_str = f.readlines()
    head = f_str[0].split(',')
    if len(head) == 1:
        f_list = [l.split('\n')[0] for l in f_str[1:]]
        return f_list
    elif len(head) == 2:
        f_list = [l.split(',')[0] for l in f_str[1:]]
        label = [np.int(l.split(',')[1]) for l in f_str[1:]]
        return f_list, label


def crop_sub_imgs_fn(x, is_random=True):
    x = tl.prepro.flip_axis(x, axis=1, is_random=is_random)
    return tl.prepro.crop(x, wrg=224, hrg=224, is_random=is_random)


def load_and_assign_npz(sess, model_path, model_name, var_list):
    load_params = tl.files.load_npz(path=model_path, name=model_name)
    load_ops = []
    for idx, param in enumerate(load_params):
        load_ops.append(var_list[idx].assign(param))
    sess.run(load_ops)


def eval_IoU(output, target, threshold=0.5, axis=(1, 2, 3), smooth=1e-5):
    """Non-differentiable Intersection over Union (IoU) for comparing the
    similarity of two batch of data, usually be used for evaluating binary image segmentation.
    The coefficient between 0 to 1, 1 means totally match.
    
    Parameters
    ----------
    output    : a np array       : A segment mask of prediction
    target    : a np array       : A segment mask of ground truth
    threshold : float            : The threshold value to be true
    axis      : tuple of integer : Dimensions are reduced
    smooth    : float            : This small value will be added to the numerator and denominator
                                   If both output and target are empty, it makes sure iou is 1
    
    Examples
    --------
    >>> y = np.zeros((5, 224, 224, 1)); y[:, 100:200, 100:200, :] = 0.9
    >>> seg = np.zeros((5, 224, 224, 1)); seg[:, 110:210, 90:190, :] = 1
    >>> eval_IoU(y, seg)
        array([ 0.68067227,  0.68067227,  0.68067227,  0.68067227,  0.68067227])
        
    >>> y = np.zeros((5, 224, 224)); y[:, 100:200, 100:200] = 0.9
    >>> seg = np.zeros((5, 224, 224)); seg[:, 110:210, 90:190] = 1
    >>> eval_IoU(y, seg, axis=(1,2))
        array([ 0.68067227,  0.68067227,  0.68067227,  0.68067227,  0.68067227])
    
    >>> y = np.zeros((224, 224)); y[100:200, 100:200] = 0.9
    >>> seg = np.zeros((224, 224)); seg[110:210, 90:190] = 1
    >>> eval_IoU(y, seg, axis=(0,1))
        0.68067227
    """
    output = output > threshold
    target = target > 0.5
    inse = np.sum(output*target, axis=axis)
    union = np.sum(output+target, axis=axis) # array(True)+array(True)=array(True)
    IoU = (inse+smooth) / (union+smooth)
    return IoU



def eval_dice(output, target, dice_type='jaccard', axis=(1, 2, 3), smooth=1e-5):
    """Soft dice (Sørensen or Jaccard) coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation
    i.e. labels are binary. The coefficient between 0 to 1, 1 means totally match.
    
    Parameters
    ----------
    output    : a np array       : A segment probability graph of prediction
    target    : a np array       : A segment mask of ground truth
    dice_type : string           : ``jaccard`` or ``sorensen``
    axis      : tuple of integer : Dimensions are reduced
    smooth    : float            : This small value will be added to the numerator and denominator
                                   If both output and target are empty, it makes sure dice is 1
    
    Examples
    --------
    >>> y = np.zeros((5, 224, 224, 1)); y[:, 100:200, 100:200, :] = 0.9
    >>> seg = np.zeros((5, 224, 224, 1)); seg[:, 110:210, 90:190, :] = 1
    >>> eval_dice(y, seg)
        array([ 0.80552486,  0.80552486,  0.80552486,  0.80552486,  0.80552486])
        
    >>> y = np.zeros((5, 224, 224)); y[:, 100:200, 100:200] = 0.9
    >>> seg = np.zeros((5, 224, 224)); seg[:, 110:210, 90:190] = 1
    >>> eval_dice(y, seg, axis=(1,2))
        array([ 0.80552486,  0.80552486,  0.80552486,  0.80552486,  0.80552486])
    
    >>> y = np.zeros((224, 224)); y[100:200, 100:200] = 0.9
    >>> seg = np.zeros((224, 224)); seg[110:210, 90:190] = 1
    >>> eval_dice(y, seg, axis=(0,1))
        0.80552486
        
    References
    -----------
    [1] Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>
    """
    target = target > 0.5
    inse = np.sum(output*target, axis=axis)
    
    if dice_type == 'jaccard':
        l = np.sum(output*output, axis=axis)
        r = np.sum(target*target, axis=axis)
    elif dice_type == 'sorensen':
        l = np.sum(output, axis=axis)
        r = np.sum(target, axis=axis)
    else:
        raise Exception("Unknow loss_type")

    dice = (2. * inse + smooth) / (l + r + smooth)
    return dice



def eval_dice_hard(output, target, threshold=0.5, axis=(1, 2, 3), smooth=1e-5):
    """Non-differentiable Sørensen–Dice coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation i.e. labels are binary.
    The coefficient between 0 to 1, 1 if totally match.
    
    Parameters
    ----------
    output    : a np array       : A segment probability graph of prediction
    target    : a np array       : A segment mask of ground truth
    dice_type : string           : ``jaccard`` or ``sorensen``
    axis      : tuple of integer : Dimensions are reduced
    smooth    : float            : This small value will be added to the numerator and denominator
                                   If both output and target are empty, it makes sure dice is 1
    
    Examples
    --------
    >>> y = np.zeros((5, 224, 224, 1)); y[:, 100:200, 100:200, :] = 0.9
    >>> seg = np.zeros((5, 224, 224, 1)); seg[:, 110:210, 90:190, :] = 1
    >>> eval_dice_hard(y, seg)
        array([ 0.81,  0.81,  0.81,  0.81,  0.81])
        
    >>> y = np.zeros((5, 224, 224)); y[:, 100:200, 100:200] = 0.9
    >>> seg = np.zeros((5, 224, 224)); seg[:, 110:210, 90:190] = 1
    >>> eval_dice_hard(y, seg, axis=(1,2))
        array([ 0.81,  0.81,  0.81,  0.81,  0.81])
    
    >>> y = np.zeros((224, 224)); y[100:200, 100:200] = 0.9
    >>> seg = np.zeros((224, 224)); seg[110:210, 90:190] = 1
    >>> eval_dice_hard(y, seg, axis=(0,1))
        0.81
    
    References
    -----------
    [1] Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>
    """
    dice_hard = eval_dice(output>threshold, target, axis=axis, smooth=smooth)
    return dice_hard



def eval_hausdorff_distance(c_contour, r_contour):
    """The Hausdorff distance is the longest distance you can be forced to
    travel by an adversary who chooses a point in one of the two sets,
    from where you then must travel to the other set.
    
    Parameters
    ----------
    c_contour : a np array : Computed contour
    r_contour : a np array : Radiologist contour
    
    Examples
    --------
    >>> from skimage import measure
    >>> c_contour = measure.find_contours(c_im, 0)[0]
    >>> r_contour = measure.find_contours(r_im, 0)[0]
    >>> hd = eval_hausdorff_distance(c_contour, r_contour)
    
    References
    -----------
    [1] Wiki-Hausdorff distance <https://en.wikipedia.org/wiki/Hausdorff_distance>
    """
    D_cxr = scipy.spatial.distance.cdist(c_contour, r_contour, 'euclidean')
    #none symmetric Hausdorff distances
    H_c = np.max(np.min(D_cxr, axis=1))
    H_r = np.max(np.min(D_cxr, axis=0))
    return np.max([H_c, H_r])



def eval_average_distance(c_contour, r_contour):
    """The average distance between the boundaries C and R.
    
    Parameters
    ----------
    c_contour : a np array : Computed contour
    r_contour : a np array : Radiologist contour
    
    Examples
    --------
    >>> from skimage import measure
    >>> c_contour = measure.find_contours(c_im, 0)[0]
    >>> r_contour = measure.find_contours(r_im, 0)[0]
    >>> hd = eval_average_distance(c_contour, r_contour)
    
    References
    -----------
    [1] Yanhui Guo -<A novel breast ultrasound image segmentation algorithm
        based on neutrosophic similarity score and level set>- 2 September 2015
    """
    D_cxr = scipy.spatial.distance.cdist(c_contour, r_contour, 'euclidean')
    #none symmetric Hausdorff distances
    H_c = np.mean(np.min(D_cxr, axis=1))
    H_r = np.mean(np.min(D_cxr, axis=0))
    return (H_c + H_r) / 2.


def eval_tfpn(output, target, threshold=0.5, axis=(0, 1, 2, 3), smooth=1e-5):
    output = output > threshold
    target = target > 0.5

    tp = np.sum(output*target, axis=axis)
    fp = np.sum(output*(1-target), axis=axis)
    fn = np.sum((1-output)*target, axis=axis)
    tn = np.sum((1-output)*(1-target), axis=axis)

    acc = (tp+tn)/(tp+tn+fp+fn+smooth)
    precision = tp/(tp+fp+smooth)
    recall = tp/(tp+fn+smooth)
    TNR = tn/(tn+fp+smooth)

    return acc, precision, recall, TNR



def select_max(contours):
    con_i = -1
    max_con = 0
    for i, contour in enumerate(contours):
        if max_con < len(contour):
            max_con = len(contour)
            con_i = i
    if con_i == -1:
        return []
    else:
        return [con_i]


def eval_H_dist(output, target, threshold=0.5):
    output = np.pad(output[0,:,:,0], 10, 'constant', constant_values=0)
    target = np.pad(target[0,:,:,0], 10, 'constant', constant_values=0)
    output = output > threshold
    target = target > 0.5
    c_contours = measure.find_contours(output, 0)
    con_is = select_max(c_contours)
    for con_i in con_is:
        c_contour = c_contours[con_i]
        r_contour = measure.find_contours(target, 0)[0]
        hd = eval_hausdorff_distance(c_contour, r_contour)
        ad = eval_average_distance(c_contour, r_contour)
        return hd, ad
    return 408,408
