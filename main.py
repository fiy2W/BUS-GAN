# -*- coding: utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorlayer as tl

import numpy as np
import os
import time
from tqdm import trange

from model import GAN_g, GAN_d
from config import config as conf
from loss import infer_g_init_train, infer_g_valid, seg_loss
from utils import load_csv_list, crop_sub_imgs_fn, load_and_assign_npz, eval_H_dist, eval_tfpn, eval_dice_hard, eval_IoU

#os.environ['CUDA_VISIBLE_DEVICES'] = ''

def train_gan():
    ## create folders to save result images and trained model
    checkpoint_dir = conf.TRAIN.ckpt_dir
    tl.files.exists_or_mkdir(checkpoint_dir)
    samples_dir = conf.TRAIN.gan_samples_dir
    tl.files.exists_or_mkdir(samples_dir)
    logs_dir = conf.TRAIN.gan_log
    tl.files.exists_or_mkdir(logs_dir)

    ## Adam
    lr_init = conf.TRAIN.lr_init * 0.1
    beta1 = conf.TRAIN.beta1
    batch_size = conf.TRAIN.batch_size
    ni = int(np.ceil(np.sqrt(batch_size)))

    # load data
    train_img_list, _ = load_csv_list(conf.TRAIN.img_list_path)
    train_img_list2, _ = load_csv_list(conf.TRAIN.img_list_path2)
    valid_img_list, _ = load_csv_list(conf.VALID.img_list_path)

    train_imgs = np.expand_dims(np.array(tl.vis.read_images(train_img_list, path=conf.TRAIN.img_path, n_threads=32)), axis=3) / 127.5 - 1.
    train_segs = np.expand_dims(np.array(tl.vis.read_images(train_img_list, path=conf.TRAIN.seg_path, n_threads=32))>0.5, axis=3)
    train_imgs2 = np.expand_dims(np.array(tl.vis.read_images(train_img_list2, path=conf.TRAIN.img_path, n_threads=32)), axis=3) / 127.5 - 1.
    train_segs2 = np.expand_dims(np.array(tl.vis.read_images(train_img_list2, path=conf.TRAIN.seg_path, n_threads=32))>0.5, axis=3)
    valid_imgs = np.expand_dims(np.array(tl.vis.read_images(valid_img_list, path=conf.VALID.img_path, n_threads=32)), axis=3) / 127.5 - 1.
    valid_segs = np.expand_dims(np.array(tl.vis.read_images(valid_img_list, path=conf.VALID.seg_path, n_threads=32))>0.5, axis=3)
    train_data = np.concatenate((train_imgs, train_segs), axis=3)

    # vis data
    vidx = 0
    train_vis_img = train_imgs[vidx:vidx+batch_size,:,:,:]
    train_vis_seg = train_segs[vidx:vidx+batch_size,:,:,:]
    valid_vis_img = valid_imgs[vidx:vidx+batch_size,:,:,:]
    valid_vis_seg = valid_segs[vidx:vidx+batch_size,:,:,:]
    tl.vis.save_images(train_vis_img, [ni,ni], os.path.join(samples_dir, '_train_img.png'))
    tl.vis.save_images(train_vis_seg, [ni,ni], os.path.join(samples_dir, '_train_seg.png'))
    tl.vis.save_images(valid_vis_img, [ni,ni], os.path.join(samples_dir, '_valid_img.png'))
    tl.vis.save_images(valid_vis_seg, [ni,ni], os.path.join(samples_dir, '_valid_seg.png'))

    # define network
    x_m = tf.placeholder(tf.float32, shape=[None, 224, 224, 1])
    y_m = tf.placeholder(tf.float32, shape=[None, 224, 224, 1])
    x_n = tf.placeholder(tf.float32, shape=[None, 224, 224, 1])
    y_n = tf.placeholder(tf.float32, shape=[None, 224, 224, 1])
    x_valid = tf.placeholder(tf.float32, shape=[None, 256, 256, 1])
    y_valid = tf.placeholder(tf.float32, shape=[None, 256, 256, 1])

    gm_tanh, gm_logit = GAN_g(x_m, n_classes=1, is_train=True)
    gn_tanh, _ = GAN_g(x_n, n_classes=1, is_train=True, reuse=True)
    g_dice = tl.cost.dice_hard_coe(gm_logit.outputs, y_m, threshold=0, axis=[1,2,3])
    v_g, v_dice = infer_g_valid(x_valid, y_valid)
    #g_vars = g_logit.all_params

    d_logit1_real = GAN_d(x_m, y_m, is_train=True)
    d_logit1_fake0 = GAN_d(x_m, gm_tanh, is_train=True, reuse=True)
    d_logit1_fake = GAN_d(x_n, gn_tanh, is_train=True, reuse=True)
    #d_vars = d_logit1_real.all_params



    lambda_adv = 0.02
    lambda_a = 0.5
    lambda_u = 1 - lambda_a




    d_l1_loss1 = tl.cost.sigmoid_cross_entropy(d_logit1_real.outputs, tf.ones_like(d_logit1_real.outputs), name='d_l1_1')
    d_l1_loss2 = lambda_a * tl.cost.sigmoid_cross_entropy(d_logit1_fake0.outputs, tf.zeros_like(d_logit1_fake0.outputs), name='d_l1_2')
    d_l1_loss3 = lambda_u * tl.cost.sigmoid_cross_entropy(d_logit1_fake.outputs, tf.zeros_like(d_logit1_fake.outputs), name='d_l1_3')
    
    d_loss = d_l1_loss1 + d_l1_loss2 + d_l1_loss3

    g_seg_loss = seg_loss(gm_logit.outputs, y_m)
    g_gan_loss1 = lambda_adv * lambda_a * tl.cost.sigmoid_cross_entropy(d_logit1_fake0.outputs, tf.ones_like(d_logit1_fake0.outputs), name='g_gan1')
    g_gan_loss2 = lambda_adv * lambda_u * tl.cost.sigmoid_cross_entropy(d_logit1_fake.outputs, tf.ones_like(d_logit1_fake.outputs), name='g_gan2')
    g_loss = g_seg_loss + g_gan_loss1 + g_gan_loss2

    # vars
    g_vars = tl.layers.get_variables_with_name('DenseDeepLab', True, True)
    d_vars = tl.layers.get_variables_with_name('DenseAttenNet', True, True)

    #Train Operation
    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)
    
    g_var1 = [v for v in g_vars if 'ASPP' in v.name]
    g_var2 = [v for v in g_vars if 'ASPP' not in v.name]
    
    ## Pretrain
    g_optim_1 = tf.train.AdamOptimizer(lr_v*10, beta1=beta1).minimize(g_loss, var_list=g_var1)
    g_optim_2 = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_var2)
    g_optim = tf.group(g_optim_1, g_optim_2)
    d_optim = tf.train.GradientDescentOptimizer(lr_v*5).minimize(d_loss, var_list=d_vars)

    # train
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        #sess.run(tf.global_variables_initializer())
        tl.layers.initialize_global_variables(sess)
        
        ## summary
        # use tensorboard --logdir="logs/log_gan"
        # http://localhost:6006/
        tb_writer = tf.summary.FileWriter(logs_dir, sess.graph)
        tf.summary.scalar('loss_d/loss_d', d_loss)
        tf.summary.scalar('loss_d/loss_d_l1r', d_l1_loss1)
        tf.summary.scalar('loss_d/loss_d_l1f', d_l1_loss2)
        tf.summary.scalar('loss_d/loss_d_l1f0', d_l1_loss3)
        #tf.summary.scalar('loss_d/loss_d_l2r', d_l2_loss1)
        #tf.summary.scalar('loss_d/loss_d_l2f', d_l2_loss2)
        tf.summary.scalar('loss_g/loss_g', g_loss)
        tf.summary.scalar('loss_g/loss_gan1', g_gan_loss1)
        tf.summary.scalar('loss_g/loss_gan2', g_gan_loss2)
        tf.summary.scalar('loss_g/loss_seg', g_seg_loss)
        tf.summary.scalar('dice', g_dice)
        tf.summary.scalar('learning_rate', lr_v)
        tb_merge = tf.summary.merge_all()

        
        # load model
        #load_and_assign_npz(sess=sess, model_path=checkpoint_dir, model_name=conf.TRAIN.g_model.split('/')[-1], var_list=g_vars)
        tl.files.load_and_assign_npz(sess=sess, name=conf.TRAIN.g_model, network=gm_logit)
        tl.files.load_and_assign_npz(sess=sess, name=conf.TRAIN.d_model1, network=d_logit1_real)
        #load_and_assign_npz(sess=sess, model_path=checkpoint_dir, model_name=conf.TRAIN.d_model.split('/')[-1], var_list=d_vars)

        # datasets information
        n_epoch = conf.TRAIN.n_epoch
        lr_decay = conf.TRAIN.lr_decay
        decay_every = conf.TRAIN.decay_every
        n_step_epoch = np.int(len(train_imgs)/batch_size)
        n_step = n_epoch * n_step_epoch
        #val_step_epoch = np.int(val_fX.shape[0]/FLAGS.batch_size)
    
        print('\nInput Data Info:')
        print('   train_file_num:', len(train_imgs), '\tval_file_num:', len(valid_imgs))
        print('\nTrain Params Info:')
        print('   learning_rate:', lr_init)
        print('   batch_size:', batch_size)
        print('   n_epoch:', n_epoch, '\tstep in an epoch:', n_step_epoch, '\ttotal n_step:', n_step)
        print('\nBegin Training ...')
    
        #coord = tf.train.Coordinator()
        #threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        max_dice = 0
        tb_train_idx = 0
        for epoch in range(n_epoch):
            ## update learning rate
            if epoch != 0 and (epoch % decay_every == 0):
                new_lr_decay = lr_decay**(epoch // decay_every)
                sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
                log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
                print(log)
            elif epoch == 0:
                sess.run(tf.assign(lr_v, lr_init))
                log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f (for GAN)" % (lr_init, decay_every, lr_decay)
                print(log)

            #time_start = time.time()
            t_batch = [x for x in tl.iterate.minibatches(inputs=train_data, targets=train_data, batch_size=batch_size, shuffle=True)]
            t_batch2 = [x for x in tl.iterate.minibatches(inputs=train_imgs2, targets=train_segs2, batch_size=batch_size, shuffle=True)]
            
            tbar = trange(min(len(t_batch), len(t_batch2)), unit='batch', ncols=100)
            train_err_d, train_err_g, train_dice, n_batch = 0, 0, 0, 0
            for i in tbar:
                ## You can also use placeholder to feed_dict in data after using
                #img_seg = np.concatenate((batch[i][0], batch[i][1]), axis=3)
                img_seg = tl.prepro.threading_data(t_batch[i][0], fn=crop_sub_imgs_fn, is_random=True)
                img_feed = np.expand_dims(img_seg[:,:,:,0], axis=3)
                seg_feed = np.expand_dims(img_seg[:,:,:,1], axis=3)
                xn_img_feed = tl.prepro.threading_data(t_batch2[i][0], fn=crop_sub_imgs_fn, is_random=False)
                yn_img_feed = tl.prepro.threading_data(t_batch2[i][1], fn=crop_sub_imgs_fn, is_random=False)

                feed_dict = {x_m: img_feed, y_m: seg_feed, x_n: xn_img_feed, y_n: yn_img_feed}
                

                # update D
                #sess.run(d_optim, feed_dict=feed_dict)
                _errD, _errDl11, _errDl12, _errDl13, _ = sess.run([d_loss, d_l1_loss1, d_l1_loss2, d_l1_loss3, d_optim], feed_dict=feed_dict)

                # update G
                _tbres, _dice, _errG, _errSeg, _errGAN1, _errGAN2, _ = sess.run([tb_merge, g_dice, g_loss, g_seg_loss, g_gan_loss1, g_gan_loss2, g_optim], feed_dict=feed_dict)

                train_err_g += _errG; train_err_d += _errD; train_dice += _dice; n_batch += 1
                tbar.set_description('Epoch %d/%d ### step %i' % (epoch+1, n_epoch, i))
                tbar.set_postfix(dice=train_dice/n_batch, g=train_err_g/n_batch, d=train_err_d/n_batch, g_seg=_errSeg, g_gan=_errGAN1+_errGAN2, d_11=_errDl11, d_12=_errDl12+_errDl13)

                tb_writer.add_summary(_tbres, tb_train_idx)
                tb_train_idx += 1
            
            if np.mod(epoch, conf.VALID.per_print) == 0:
                # vis image
                feed_dict = {x_valid: train_vis_img, y_valid: train_vis_seg}
                feed_dict.update(v_g.all_drop)
                _output = sess.run(v_g.outputs, feed_dict=feed_dict)
                tl.vis.save_images(_output, [ni,ni], os.path.join(samples_dir, 'train_pred_{}.png'.format(epoch)))
                feed_dict = {x_valid: valid_vis_img, y_valid: valid_vis_seg}
                feed_dict.update(v_g.all_drop)
                _output = sess.run(v_g.outputs, feed_dict=feed_dict)
                tl.vis.save_images(_output, [ni,ni], os.path.join(samples_dir, 'valid_pred_{}.png'.format(epoch)))
                print('Validation ...')
                time_start = time.time()
                val_acc, n_batch = 0, 0
                for batch in tl.iterate.minibatches(inputs=valid_imgs, targets=valid_segs, batch_size=1, shuffle=True):
                    img_feed, seg_feed = batch
                    feed_dict = {x_valid: img_feed, y_valid: seg_feed}
                    feed_dict.update(v_g.all_drop)
                    _dice = sess.run(v_dice, feed_dict=feed_dict)
                    val_acc += _dice; n_batch += 1
                print('   Time:{}\tDice:{}'.format(time.time()-time_start, val_acc/n_batch))

                if val_acc/n_batch > max_dice:
                    max_dice = val_acc/n_batch
                    
                print('[!] Max dice:', max_dice)
        tl.files.save_npz(gm_logit.all_params, name=conf.TRAIN.g_model)
        tl.files.save_npz(d_logit1_real.all_params, name=conf.TRAIN.d_model1)



def evaluate():
    ## create folders to save result images and trained model
    checkpoint_dir = conf.TRAIN.ckpt_dir
    tl.files.exists_or_mkdir(checkpoint_dir)
    result_dir = conf.VALID.result_dir
    tl.files.exists_or_mkdir(result_dir)

    # load data
    valid_img_list, _ = load_csv_list(conf.VALID.img_list_path)
    valid_imgs = np.expand_dims(np.array(tl.vis.read_images(valid_img_list, path=conf.VALID.img_path, n_threads=32)), axis=3) / 127.5 - 1.
    valid_segs = np.expand_dims(np.array(tl.vis.read_images(valid_img_list, path=conf.VALID.seg_path, n_threads=32))>0.5, axis=3)
    
    # define model
    x_valid = tf.placeholder(tf.float32, shape=[None, 288, 288, 1])
    y_valid = tf.placeholder(tf.float32, shape=[None, 288, 288, 1])

    gm_tanh, gm_logit = GAN_g(x_valid, n_classes=1, is_train=False, dropout=1.)
    oris = []
    segs = []
    pred_maps = []

    # valid
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        #sess.run(tf.global_variables_initializer())
        #tl.layers.initialize_global_variables(sess)

        # load model
        tl.files.load_and_assign_npz(sess=sess, name=conf.TRAIN.g_model, network=gm_logit)

        results = {'dice': [], 'FScore': [], 'HDist': [], 'Obj': [], 'Time': []}

        for batch in tl.iterate.minibatches(inputs=valid_imgs, targets=valid_segs, batch_size=1, shuffle=False):
            img_feed, seg_feed = batch
            feed_dict = {x_valid: img_feed, y_valid: seg_feed}
            t_start = time.time()
            _out = sess.run(gm_logit.outputs, feed_dict=feed_dict)
            pred_maps.append(_out)
            oris.append(img_feed)
            segs.append(seg_feed)

            _, precision, recall, _ = eval_tfpn(_out>0, seg_feed)
            HD, AD = eval_H_dist(_out>0, seg_feed)
            IoU = np.mean(eval_IoU(_out>0, seg_feed))
            results['dice'].append(np.mean(eval_dice_hard(_out>0, seg_feed)))
            results['IoU'].append(IoU)
            results['precision'].append(np.mean(precision))
            results['recall'].append(np.mean(recall))
            results['HDist'].append(HD)
            results['AvgDist'].append(AD)
            results['Obj'].append(0 if np.sum(_out>0) == 0 else 1)
            results['Time'].append(time.time()-t_start)

        np.save(os.path.join(conf.VALID.result_dir, 'pred.npy'), np.array(pred_maps))
        np.save(os.path.join(conf.VALID.result_dir, 'ori.npy'), np.array(oris))
        np.save(os.path.join(conf.VALID.result_dir, 'seg.npy'), np.array(segs))
        with open(os.path.join(conf.VALID.result_dir, 'results.csv'), 'w') as f:
            f.write('name,dice,IoU,precision,recall,HDist,AvgDist,Obj,Time\n')
            for i, valid_name in enumerate(valid_img_list):
                f.write('{},{},{},{},{},{},{},{},{}\n'.format(valid_name, results['dice'][i], results['IoU'][i], results['precision'][i], results['recall'][i], results['HDist'][i], results['AvgDist'][i], results['Obj'][i], results['Time'][i]))
        print(len(results['dice']))
        print(np.mean(results['dice']), np.std(results['dice']))
        print(np.mean(results['HDist']), np.std(results['HDist']))




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train_gan', help='train_gan, evaluate')
    args = parser.parse_args()
    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'train_gan':
        train_gan()
    elif tl.global_flag['mode'] == 'evaluate':
        evaluate()
    else:
        raise Exception("Unknow --mode")
