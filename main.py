import os
import csv
import cv2
import json
import utils
import random
import argparse
import data_loader
import numpy as np
import tensorflow as tf
from scipy.misc.pilutil import imsave
import models.losses, models.dsa_gam, models.params, models.dsa_edm, models.params


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.set_random_seed(1)
np.random.seed(0)
slim = tf.contrib.slim


def main():
    
    args = parse_args()

    if args is None:
        exit()

    to_train = args.to_train
    log_dir = args.log_dir
    config_filename = args.config_filename
    checkpoint_dir = args.checkpoint_dir
    skip = args.skip
    switch = args.switch
    threshold_fg = args.threshold

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    with open(config_filename) as config_file:
        config = json.load(config_file)

    lambda_a = float(config['DSAGAM_LAMBDA_A']) if 'DSAGAM_LAMBDA_A' in config else 10.0
    lambda_b = float(config['DSAGAM_LAMBDA_B']) if 'DSAGAM_LAMBDA_B' in config else 10.0
    pool_size = int(config['DSAGAM_pool_size']) if 'DSAGAM_pool_size' in config else 50

    to_restore = (to_train == 2)
    base_lr = float(config['DSAGAM_base_lr']) if 'DSAGAM_base_lr' in config else 0.0002
    max_step = int(config['DSAGAM_max_step']) if 'DSAGAM_max_step' in config else 200
    network_version = str(config['DSAGAM_network_version'])if 'DSAGAM_network_version' in config else "py"
    dataset_name = str(config['DSAGAM_dataset_name'])
    do_flipping = bool(config['DSAGAM_do_flipping'])
    thresh_bbox = float(config['DSAGAM_threshold'])if 'DSAGAM_threshold' in config else 130

    cyclegan_model = models.dsa_edm.CycleGAN(pool_size, lambda_a, lambda_b, log_dir,
                              to_restore, base_lr, max_step, network_version,
                              dataset_name, checkpoint_dir, do_flipping, skip,
                              switch, threshold_fg)

    if to_train > 0:
        cyclegan_model.train()
    else:
        cyclegan_model.test()

    BBox_area = int(config['DSAGAM_area']) if 'DSAGAM_area' in config else 1000

    path_denoise = utils.Bbox(thresh_bbox, BBox_area)
    path = './output/exp_01/DSA-EDM/'
    file_list = os.listdir(path)

    for file_name in file_list:
        if not os.path.isdir(path + file_name):
            train(N_STEP,path + file_name, 0.3, sigma, is_realnoisy = True,)

    os.remove(path_denoise)


config_filename = './configs/exp_01.json'

with open(config_filename) as config_file:
        config = json.load(config_file)

N_STEP = int(config['DSAEDM_N_STEP']) if 'DSAEDM_N_STEP' in config else 3000
sigma = int(config['DSAEDM_sigma']) if 'DSAEDM_sigma' in config else -1
LEARNING_RATE = float(config['DSAEDM_lr']) if 'DSAEDM_lr' in config else 1e-4
N_PREDICTION = int(config['DSAEDM_N_PREDICTION']) if 'DSAEDM_N_PREDICTION' in config else 100
N_SAVE =int(config['DSAEDM_N_save']) if 'DSAEDM_N_save' in config else 1000
TF_DATA_TYPE = tf.float32


def train(N_STEP, file_path, dropout_rate, sigma=25, is_realnoisy=False):

    tf.reset_default_graph()
    ground_truth = utils.load_np_image(file_path) # load image
    _, w, h, c = np.shape(ground_truth)
    model_path = file_path[0:file_path.rfind(".")] + "/" + str(sigma) + "/model/"
    os.makedirs(model_path, exist_ok=True)
    noisy = utils.add_gaussian_noise(ground_truth, model_path, sigma)  # If values as -1 then perform denoising immediately
    model = models.dsa_gam.build_denoising_unet(noisy, 1 - dropout_rate, is_realnoisy)

    loss = model['training_error']

    # summarise and save the training information
    summay = model['summary']

    # save model
    saver = model['saver']

    our_image = model['our_image']

    # slip operation: left/right or up/down
    is_flip_lr = model['is_flip_lr']
    is_flip_ud = model['is_flip_ud']
    
    # initialisation
    avg_op = model['avg_op']
    slice_avg = model['slice_avg']
    optimiser = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    avg_loss = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(model_path, sess.graph)
        for step in range(N_STEP):
            feet_dict = {is_flip_lr: np.random.randint(0, 2), is_flip_ud: np.random.randint(0, 2)}
            _, _op, loss_value, merged, o_image = sess.run([optimiser, avg_op, loss, summay, our_image],
                                                           feed_dict=feet_dict)
            avg_loss += loss_value
            if (step + 1) % N_SAVE == 0:

                print("After %d training dsa_edm step(s)" % (step + 1),
                      "loss  is {:.9f}".format(avg_loss / N_SAVE))
                avg_loss = 0
                sum = np.float32(np.zeros(our_image.shape.as_list()))
                for j in range(N_PREDICTION):
                    feet_dict = {is_flip_lr: np.random.randint(0, 2), is_flip_ud: np.random.randint(0, 2)}
                    o_avg, o_image = sess.run([slice_avg, our_image], feed_dict=feet_dict)
                    sum += o_image
                o_image = np.squeeze(np.uint8(np.clip(sum / N_PREDICTION, 0, 1) * 255))
                o_avg = np.squeeze(np.uint8(np.clip(o_avg, 0, 1) * 255))
                if is_realnoisy:
                    cv2.imwrite(model_path + 'idea-' + str(step + 1) + '.png', o_avg)
                else:
                    cv2.imwrite(model_path + 'idea-' + str(step + 1) + '.png', o_image)
                
            summary_writer.add_summary(merged, step)


def parse_args():

    desc = "IDEA-NET"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--to_train', type=int, default=True, help='Whether it is train or false.')
    parser.add_argument('--log_dir', type=str, default='./output/exp_01', help='Where the data is logged to.')
    parser.add_argument('--config_filename', type=str, default='./configs/exp_01.json', help='configuration file name')
    parser.add_argument('--checkpoint_dir', type=str, default='./output/exp_01/CHECK', help='train/test split name')
    parser.add_argument('--skip', type=bool, default=False, help='Use skip-connection between encoder/decoder or not')
    parser.add_argument('--switch', type=int, default=30, help='Number of epoch that foreground fed to discriminator')
    parser.add_argument('--threshold', type=float, default=0.1, help='Threshold for foreground selection ')

    return parser.parse_args()


if __name__ == '__main__':
    main()
