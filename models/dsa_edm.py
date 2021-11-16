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
import models.losses, models.dsa_gam, models.params


slim = tf.contrib.slim


class CycleGAN:

    def __init__(self, pool_size, lambda_a, lambda_b,
                 output_root_dir, to_restore, base_lr, max_step, network_version,
                 dataset_name, checkpoint_dir, do_flipping, skip, switch, threshold_fg):

        self._pool_size = pool_size
        self._size_before_crop = 256
        self._switch = switch                  
        self._threshold_fg = threshold_fg           # threshold used to justify the foreground, this value is set to 0 as we do not use the result of the generator in DSA-GAM
        self._lambda_a = lambda_a                   # lambda_a: is the upsampling factor used in the loss function of DSA-GAM
        self._lambda_b = lambda_b
        self._output_dir = os.path.join(output_root_dir, 'DSA-GAM')
        self._output_dir1 = os.path.join(output_root_dir, 'CHECK')
        self._images_dir = os.path.join(self._output_dir, 'I_overline')
        self._num_imgs_to_save = 1
        self._to_restore = to_restore               # to_restore: path for saving result
        self._base_lr = base_lr
        self._max_step = max_step                   # max_step: maximum number of iterations
        self._network_version = network_version     # print out the architecture
        self._dataset_name = dataset_name
        self._checkpoint_dir = checkpoint_dir
        self._do_flipping = do_flipping
        self._skip = skip
        self.fake_images_A = []
        self.fake_images_B = []

    def model_setup(self):
        """
        The function builds a model for training.
        Self.input_I_tilde/self.input_MASK0: training image set
        Self.fake_a/self.Fake_b: generate the image through the corresponding generator
        Enter input_I_tilde and input_MASK0
        Self.LR: learning rate
        Self.Cyc_A/self.Cyc_b: images generated after feeding
        Self.fake_a/self.Fake_b corresponding generator.
        This is used to calculate the cycle losses
        """

        self.input_I_tilde = tf.placeholder(
            tf.float32, [
                1,
                models.params.IMG_WIDTH,
                models.params.IMG_HEIGHT,
                models.params.IMG_CHANNELS
            ], name="input_I_tilde")

        self.input_MASK0 = tf.placeholder(
            tf.float32, [
                1,
                models.params.IMG_WIDTH,
                models.params.IMG_HEIGHT,
                models.params.IMG_CHANNELS
            ], name="input_MASK0")

        self.fake_pool_A = tf.placeholder(
            tf.float32, [
                None,
                models.params.IMG_WIDTH,
                models.params.IMG_HEIGHT,
                models.params.IMG_CHANNELS
            ], name="fake_pool_A")

        self.fake_pool_B = tf.placeholder(
            tf.float32, [
                None,
                models.params.IMG_WIDTH,
                models.params.IMG_HEIGHT,
                models.params.IMG_CHANNELS
            ], name="fake_pool_B")

        self.fake_pool_A_mask = tf.placeholder(
            tf.float32, [
                None,
                models.params.IMG_WIDTH,
                models.params.IMG_HEIGHT,
                models.params.IMG_CHANNELS
            ], name="fake_pool_A_mask")

        self.fake_pool_B_mask = tf.placeholder(
            tf.float32, [
                None,
                models.params.IMG_WIDTH,
                models.params.IMG_HEIGHT,
                models.params.IMG_CHANNELS
            ], name="fake_pool_B_mask")

        self.global_step = slim.get_or_create_global_step()
        self.num_fake_inputs = 0
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name="lr")
        self.transition_rate = tf.placeholder(tf.float32, shape=[], name="tr")
        self.donorm = tf.placeholder(tf.bool, shape=[], name="donorm")

        inputs = {
            # the input noisy image
            'images_a': self.input_I_tilde,
            # MASK0
            'images_b': self.input_MASK0,
            'fake_pool_a': self.fake_pool_A,
            'fake_pool_b': self.fake_pool_B,
            'fake_pool_a_mask': self.fake_pool_A_mask,
            'fake_pool_b_mask': self.fake_pool_B_mask,
            # transition_rate == 0.1
            'transition_rate': self.transition_rate,
            # donorm: a param used to control program flow, and is usually valued as True or False
            'donorm': self.donorm,
        }

        outputs = models.dsa_gam.get_outputs(
            inputs, skip=self._skip)

        self.prob_real_a_is_real = outputs['prob_real_a_is_real']
        self.prob_real_b_is_real = outputs['prob_real_b_is_real']
        self.fake_images_a = outputs['fake_images_a']
        self.fake_images_b = outputs['fake_images_b']
        self.prob_fake_a_is_real = outputs['prob_fake_a_is_real']
        self.prob_fake_b_is_real = outputs['prob_fake_b_is_real']

        self.cycle_images_a = outputs['cycle_images_a']
        self.cycle_images_b = outputs['cycle_images_b']

        self.prob_fake_pool_a_is_real = outputs['prob_fake_pool_a_is_real']
        self.prob_fake_pool_b_is_real = outputs['prob_fake_pool_b_is_real']
        self.masks = outputs['masks']
        self.masked_gen_ims = outputs['masked_gen_ims']
        self.masked_ims = outputs['masked_ims']
        self.masks_ = outputs['mask_tmp']

    def compute_losses(self):
        """
        In this function, we define the variables for the loss calculation and training models.
        D_loss_A/d_loss_B: loss for discriminator A/B
        G_LOS_A/G_LOS_B: loss for generator A/B
        trainer: loss function of various trainers
        Summer: summarise the variables of the above loss function
        """
        cycle_consistency_loss_a = \
            self._lambda_a * models.losses.cycle_consistency_loss(
                real_images=self.input_I_tilde, generated_images=self.cycle_images_a,
            )

        cycle_consistency_loss_b = \
            self._lambda_b * models.losses.cycle_consistency_loss(
                real_images=self.input_MASK0, generated_images=self.cycle_images_b,
            )

        lsgan_loss_a = models.losses.lsgan_loss_generator(self.prob_fake_a_is_real)
        lsgan_loss_b = models.losses.lsgan_loss_generator(self.prob_fake_b_is_real)

        g_loss_A = \
            cycle_consistency_loss_a + cycle_consistency_loss_b + lsgan_loss_b
        g_loss_B = \
            cycle_consistency_loss_b + cycle_consistency_loss_a + lsgan_loss_a

        d_loss_A = models.losses.lsgan_loss_discriminator(
            prob_real_is_real=self.prob_real_a_is_real,
            prob_fake_is_real=self.prob_fake_pool_a_is_real,
        )

        d_loss_B = models.losses.lsgan_loss_discriminator(
            prob_real_is_real=self.prob_real_b_is_real,
            prob_fake_is_real=self.prob_fake_pool_b_is_real,
        )

        optimiser = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5)
        self.model_vars = tf.trainable_variables()

        d_A_vars = [var for var in self.model_vars if 'd_A' in var.name]
        g_A_vars = [var for var in self.model_vars if 'g_A/' in var.name]
        d_B_vars = [var for var in self.model_vars if 'd_B' in var.name]
        g_B_vars = [var for var in self.model_vars if 'g_B/' in var.name]
        g_Ae_vars = [var for var in self.model_vars if 'g_A_ae' in var.name]
        g_Be_vars = [var for var in self.model_vars if 'g_B_ae' in var.name]

        self.g_A_trainer = optimiser.minimize(g_loss_A, var_list=g_A_vars+g_Ae_vars)
        self.g_B_trainer = optimiser.minimize(g_loss_B, var_list=g_B_vars+g_Be_vars)
        self.g_A_trainer_bis = optimiser.minimize(g_loss_A, var_list=g_A_vars)
        self.g_B_trainer_bis = optimiser.minimize(g_loss_B, var_list=g_B_vars)
        self.d_A_trainer = optimiser.minimize(d_loss_A, var_list=d_A_vars)
        self.d_B_trainer = optimiser.minimize(d_loss_B, var_list=d_B_vars)
        self.params_ae_c1 = g_A_vars[0]
        self.params_ae_c1_B = g_B_vars[0]

        for var in self.model_vars:
            print(var.name)

        # Summarise variables for tensorboard
        self.g_A_loss_summ = tf.summary.scalar("g_A_loss", g_loss_A)
        self.g_B_loss_summ = tf.summary.scalar("g_B_loss", g_loss_B)
        self.d_A_loss_summ = tf.summary.scalar("d_A_loss", d_loss_A)
        self.d_B_loss_summ = tf.summary.scalar("d_B_loss", d_loss_B)

    def save_images(self, sess, epoch, curr_tr):
        """
        Saves output mask images.
        """
        if not os.path.exists(self._images_dir):
            os.makedirs(self._images_dir)

        if curr_tr > 0:
            donorm = False
        else:
            donorm = True

        names = ['mask_a', 'mask_b']

        with open(os.path.join(
                self._output_dir1, 'epoch_'  + '.html'), 'w') as v_html:
            
            for i in range(0, self._num_imgs_to_save):
                print("=========== DSA-GAM ===============")
                print("Saving image {}/{}".format(i, self._num_imgs_to_save))
                inputs = sess.run(self.inputs)
                fake_A_temp, fake_B_temp, cyc_A_temp, cyc_B_temp, masks = sess.run([self.fake_images_a,self.fake_images_b,self.cycle_images_a,self.cycle_images_b,self.masks,], feed_dict={
                    self.input_I_tilde: inputs['images_i'],
                    self.input_MASK0: inputs['images_j'],
                    self.transition_rate: curr_tr,
                    self.donorm: donorm,
                })

                tensors = [masks[0], masks[1],inputs['images_i'], inputs['images_j'],
                           fake_B_temp, fake_A_temp, cyc_A_temp, cyc_B_temp, masks[0], masks[1]]
                #tensors = [inputs['images_i'], inputs['images_j'],
                            #masks[0], masks[1]]

                for name, tensor in zip(names, tensors):
                    image_name =  str(i) + ".jpg"
                    if 'mask_a' in name:
                        imsave(os.path.join(self._images_dir, image_name),
                               (np.squeeze(tensor[0]))
                               )
                    else:
                        v_html.write("<br>")

    def save_images_bis(self, sess, epoch):
        """
        output images (this could be deleted)
        """
        if not os.path.exists(self._images_dir):
            os.makedirs(self._images_dir)

        names = ['input_I_tilde_', 'mask_A_', 'masked_inputA_', 'fakeB_',
                 'input_MASK0_', 'mask_B_', 'masked_inputB_', 'fakeA_']
        # set up spacing for saving
        space = '&nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp ' \
                '&nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp ' \
                '&nbsp &nbsp &nbsp &nbsp &nbsp'
        with open(os.path.join(self._output_dir1, 'results_'  + '.html'), 'w') as v_html:

            v_html.write("<b>INPUT" + space + "MASK" + space + "MASKED_IMAGE" + space + "GENERATED_IMAGE</b>")
            v_html.write("<br>")

            for i in range(0, self._num_imgs_to_save):
                print("Saving image {}/{}".format(i, self._num_imgs_to_save))
                inputs = sess.run(self.inputs)
                fake_A_temp, fake_B_temp, masks, masked_ims = sess.run([
                    self.fake_images_a,
                    self.fake_images_b,
                    self.masks,
                    self.masked_ims
                ], feed_dict={
                    self.input_I_tilde: inputs['images_i'],
                    self.input_MASK0: inputs['images_j'],
                    self.transition_rate: 0.1
                })

                tensors = [inputs['images_i'], masks[0], masked_ims[0], fake_B_temp,
                           inputs['images_j'], masks[1], masked_ims[1], fake_A_temp]

                for name, tensor in zip(names, tensors):
                    image_name = name + str(i) + ".jpg"

                    if 'mask_' in name:
                        imsave(os.path.join(self._images_dir, image_name),
                               (np.squeeze(tensor[0]))
                               )
                    else:

                        imsave(os.path.join(self._images_dir, image_name),
                               ((np.squeeze(tensor[0]) + 1) * 127.5).astype(np.uint8)
                               )

                    v_html.write(
                        "<img src=\"" +
                        os.path.join('imgs', image_name) + "\">"
                    )

                    if 'fakeB_' in name:
                        v_html.write("<br>")
                v_html.write("<br>")

    def fake_image_pool(self, num_fakes, fake, mask, fake_pool):
        """
        This function saves the generated image to the corresponding location
        Image of the pool.If the number of images saved exceeds the set number
        It's going to feel the pool until it's full, and then at random
        Select a stored image and replace it with a new one.
        """
        tmp = {}
        tmp['im'] = fake
        tmp['mask'] = mask
        if num_fakes < self._pool_size:
            fake_pool.append(tmp)
            return tmp
        else:
            p = random.random()
            if p > 0.5:
                random_id = random.randint(0, self._pool_size - 1)
                temp = fake_pool[random_id]
                fake_pool[random_id] = tmp
                return temp
            else:
                return tmp

    def train(self):
        """ Training """
        # Load Dataset from the dataset folder
        self.inputs = data_loader.load_data(
            self._dataset_name, self._size_before_crop,
            False, self._do_flipping)

        # Build network
        self.model_setup()

        # Loss calculations
        self.compute_losses()

        # Initializing the global variables
        init = (tf.global_variables_initializer(),
                tf.local_variables_initializer())
        
        #saver = tf.train.Saver(max_to_keep=None)

        max_images = models.params.DATASET_TO_SIZES[self._dataset_name]
        half_training = int(self._max_step / 2)
        with tf.Session() as sess:
            sess.run(init)
            if self._to_restore:
                chkpt_fname = tf.train.latest_checkpoint(self._checkpoint_dir)
                #saver.restore(sess, chkpt_fname)

            writer = tf.summary.FileWriter(self._output_dir1)

            if not os.path.exists(self._output_dir):
                os.makedirs(self._output_dir)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            # Training Loop
            for epoch in range(sess.run(self.global_step), self._max_step):
                print("=========== DSA-EDM ===============")
                print("In the epoch ", epoch)
                #saver.save(sess, os.path.join(self._output_dir1, "IDEA"), global_step=1)

                # Dealing with the learning rate as per the epoch number
                if epoch < half_training:
                    curr_lr = self._base_lr
                else:
                    curr_lr = self._base_lr - \
                        self._base_lr * (epoch - half_training) / half_training

                if epoch < self._switch:
                    curr_tr = 0.
                    donorm = True
                    to_train_A = self.g_A_trainer
                    to_train_B = self.g_B_trainer
                else:
                    curr_tr = self._threshold_fg
                    donorm = False
                    to_train_A = self.g_A_trainer_bis
                    to_train_B = self.g_B_trainer_bis

                self.save_images(sess, epoch, curr_tr)

                for i in range(0, max_images):
                    print("Processing batch {}/{}".format(i, max_images))

                    inputs = sess.run(self.inputs)
                    # Optimising the G_A network
                    _, fake_B_temp, smask_a,summary_str = sess.run(
                        [to_train_A,
                         self.fake_images_b,
                         self.masks[0],
                         self.g_A_loss_summ],
                        feed_dict={
                            self.input_I_tilde:
                                inputs['images_i'],
                            self.input_MASK0:
                                inputs['images_j'],
                            self.learning_rate: curr_lr,
                            self.transition_rate: curr_tr,
                            self.donorm: donorm,
                        }
                    )
                    writer.add_summary(summary_str, epoch * max_images + i)

                    fake_B_temp1 = self.fake_image_pool(
                        self.num_fake_inputs, fake_B_temp, smask_a, self.fake_images_B)

                    # Optimising the D_B network
                    _,summary_str = sess.run(
                        [self.d_B_trainer, self.d_B_loss_summ],
                        feed_dict={
                            self.input_I_tilde:
                                inputs['images_i'],
                            self.input_MASK0:
                                inputs['images_j'],
                            self.learning_rate: curr_lr,
                            self.fake_pool_B: fake_B_temp1['im'],
                            self.fake_pool_B_mask: fake_B_temp1['mask'],
                            self.transition_rate: curr_tr,
                            self.donorm: donorm,
                        }
                    )
                    writer.add_summary(summary_str, epoch * max_images + i)

                    # Optimizing the G_B network
                    _, fake_A_temp, smask_b, summary_str = sess.run(
                        [to_train_B,
                         self.fake_images_a,
                         self.masks[1],
                         self.g_B_loss_summ],
                        feed_dict={
                            self.input_I_tilde:
                                inputs['images_i'],
                            self.input_MASK0:
                                inputs['images_j'],
                            self.learning_rate: curr_lr,
                            self.transition_rate: curr_tr,
                            self.donorm: donorm,
                        }
                    )
                    writer.add_summary(summary_str, epoch * max_images + i)

                    fake_A_temp1 = self.fake_image_pool(
                        self.num_fake_inputs, fake_A_temp, smask_b ,self.fake_images_A)

                    # Optimizing the D_A network
                    _, mask_tmp__, summary_str = sess.run(
                        [self.d_A_trainer,self.masks_, self.d_A_loss_summ],
                        feed_dict={
                            self.input_I_tilde:
                                inputs['images_i'],
                            self.input_MASK0:
                                inputs['images_j'],
                            self.learning_rate: curr_lr,
                            self.fake_pool_A: fake_A_temp1['im'],
                            self.fake_pool_A_mask: fake_A_temp1['mask'],
                            self.transition_rate: curr_tr,
                            self.donorm: donorm,
                        }
                    )
                    writer.add_summary(summary_str, epoch * max_images + i)

                    writer.flush()
                    self.num_fake_inputs += 1

                sess.run(tf.assign(self.global_step, epoch + 1))

            coord.request_stop()
            coord.join(threads)
            writer.add_graph(sess.graph)