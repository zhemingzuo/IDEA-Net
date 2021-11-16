import cv2
import numpy as np
import tensorflow as tf
from models.layers import PConv2D
import models.layers, models.params, models.losses


def get_outputs(inputs, skip=False):

    images_a = inputs['images_a']
    images_b = inputs['images_b']
    fake_pool_a = inputs['fake_pool_a']
    fake_pool_b = inputs['fake_pool_b']
    fake_pool_a_mask = inputs['fake_pool_a_mask']
    fake_pool_b_mask = inputs['fake_pool_b_mask']
    transition_rate = inputs['transition_rate']
    donorm = inputs['donorm']

    with tf.variable_scope("Model") as scope:
        current_autoenc = autoenc_upsample
        current_discriminator = discriminator
        current_generator = build_generator_resnet_9blocks

        mask_a = current_autoenc(images_a, "g_A_ae")
        mask_b = current_autoenc(images_b, "g_B_ae")
        mask_a = tf.concat([mask_a] * 3, axis=3)
        mask_b = tf.concat([mask_b] * 3, axis=3)

        mask_a_on_a = tf.multiply(images_a, mask_a)
        mask_b_on_b = tf.multiply(images_b, mask_b)

        prob_real_a_is_real = current_discriminator(images_a, mask_a, transition_rate, donorm, "d_A")
        prob_real_b_is_real = current_discriminator(images_b, mask_b, transition_rate, donorm, "d_B")

        fake_images_b_from_g = current_generator(images_a, name="g_A", skip=skip)
        fake_images_b = tf.multiply(fake_images_b_from_g, mask_a) + tf.multiply(images_a, 1 - mask_a)

        fake_images_a_from_g = current_generator(images_b, name="g_B", skip=skip)
        fake_images_a = tf.multiply(fake_images_a_from_g, mask_b) + tf.multiply(images_b, 1 - mask_b)
        scope.reuse_variables()

        prob_fake_a_is_real = current_discriminator(fake_images_a, mask_b, transition_rate, donorm, "d_A")
        prob_fake_b_is_real = current_discriminator(fake_images_b, mask_a, transition_rate, donorm, "d_B")

        mask_acycle = current_autoenc(fake_images_a, "g_A_ae")
        mask_bcycle = current_autoenc(fake_images_b, "g_B_ae")
        mask_bcycle = tf.concat([mask_bcycle] * 3, axis=3)
        mask_acycle = tf.concat([mask_acycle] * 3, axis=3)

        mask_acycle_on_fakeA = tf.multiply(fake_images_a, mask_acycle)
        mask_bcycle_on_fakeB = tf.multiply(fake_images_b, mask_bcycle)

        cycle_images_a_from_g = current_generator(fake_images_b, name="g_B", skip=skip)
        cycle_images_b_from_g = current_generator(fake_images_a, name="g_A", skip=skip)

        cycle_images_a = tf.multiply(cycle_images_a_from_g,
                                     mask_bcycle) + tf.multiply(fake_images_b, 1 - mask_bcycle)

        cycle_images_b = tf.multiply(cycle_images_b_from_g,
                                     mask_acycle) + tf.multiply(fake_images_a, 1 - mask_acycle)

        scope.reuse_variables()

        prob_fake_pool_a_is_real = current_discriminator(fake_pool_a, fake_pool_a_mask, transition_rate, donorm, "d_A")
        prob_fake_pool_b_is_real = current_discriminator(fake_pool_b, fake_pool_b_mask, transition_rate, donorm, "d_B")

    return {
        'prob_real_a_is_real': prob_real_a_is_real,
        'prob_real_b_is_real': prob_real_b_is_real,
        'prob_fake_a_is_real': prob_fake_a_is_real,
        'prob_fake_b_is_real': prob_fake_b_is_real,
        'prob_fake_pool_a_is_real': prob_fake_pool_a_is_real,
        'prob_fake_pool_b_is_real': prob_fake_pool_b_is_real,
        'cycle_images_a': cycle_images_a,
        'cycle_images_b': cycle_images_b,
        'fake_images_a': fake_images_a,
        'fake_images_b': fake_images_b,
        'masked_ims': [mask_a_on_a, mask_b_on_b, mask_acycle_on_fakeA, mask_bcycle_on_fakeB],
        'masks': [mask_a, mask_b, mask_acycle, mask_bcycle],
        'masked_gen_ims': [fake_images_b_from_g, fake_images_a_from_g, cycle_images_a_from_g, cycle_images_b_from_g],
        'mask_tmp': mask_a,
    }


def autoenc_upsample(inputae, name):

    with tf.variable_scope(name):
        f = 7
        ks = 3
        padding = "REFLECT"

        pad_input = tf.pad(inputae, [[0, 0], [ks, ks], [ks, ks], [0, 0]], padding)
        o_c1 = models.layers.general_conv2d(pad_input, tf.constant(True, dtype=bool), models.params.ngf, f, f, 2, 2, 0.02,
                                           name="c1")
        o_c2 = models.layers.general_conv2d(o_c1, tf.constant(True, dtype=bool), models.params.ngf * 2, ks, ks, 2, 2,
                                           0.02, "SAME", "c2")
        o_r1 = build_resnet_block_Att(o_c2, models.params.ngf * 2, "r1", padding)
        size_d1 = o_r1.get_shape().as_list()
        o_c4 = models.layers.upsamplingDeconv(o_r1, size=[size_d1[1] * 2, size_d1[2] * 2], is_scale=False, method=1,
                                             align_corners=False, name='up1')

        o_c4_end = models.layers.general_conv2d(o_c4, tf.constant(True, dtype=bool), models.params.ngf * 2, (3, 3),
                                               (1, 1), padding='VALID', name='c4')
        size_d2 = o_c4_end.get_shape().as_list()
        o_c5 = models.layers.upsamplingDeconv(o_c4_end, size=[size_d2[1] * 2, size_d2[2] * 2], is_scale=False, method=1,
                                             align_corners=False, name='up2')
        oc5_end = models.layers.general_conv2d(o_c5, tf.constant(True, dtype=bool), models.params.ngf, (3, 3), (1, 1),
                                              padding='VALID', name='c5')
        o_c6_end = models.layers.general_conv2d(oc5_end, tf.constant(False, dtype=bool), 1, (f, f), (1, 1),
                                               padding='VALID', name='c6', do_relu=False)
        return tf.nn.sigmoid(o_c6_end, 'sigmoid')


def build_resnet_block(inputres, dim, name="resnet", padding="REFLECT"):
    """
    build a single block of resnet.
    return: a single block of resnet.
    """
    with tf.variable_scope(name):
        out_res = tf.pad(inputres, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)
        out_res = models.layers.general_conv2d(out_res, tf.constant(True, dtype=bool), dim, 3, 3, 1, 1, 0.02, "VALID",
                                              "c1")
        out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)
        out_res = models.layers.general_conv2d(out_res, tf.constant(True, dtype=bool), dim, 3, 3, 1, 1, 0.02, "VALID",
                                              "c2", do_relu=False)
        return tf.nn.relu(out_res + inputres)


def build_resnet_block_Att(inputres, dim, name="resnet", padding="REFLECT"):
    """
    build a single block of resnet.
    return: a single block of resnet.
    """
    with tf.variable_scope(name):
        out_res = tf.pad(inputres, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)
        out_res = models.layers.general_conv2d(out_res, tf.constant(True, dtype=bool), dim, 3, 3, 1, 1, 0.02, "VALID",
                                              "c1")
        out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)
        out_res = models.layers.general_conv2d(out_res, tf.constant(True, dtype=bool), dim, 3, 3, 1, 1, 0.02, "VALID",
                                              "c2", do_relu=False)

        return tf.nn.relu(out_res + inputres)


def build_generator_resnet_9blocks(inputgen, name="generator", skip=False):

    with tf.variable_scope(name):
        f = 7
        ks = 3
        padding = "CONSTANT"
        inputgen = tf.pad(inputgen, [[0, 0], [ks, ks], [
            ks, ks], [0, 0]], padding)

        o_c1 = models.layers.general_conv2d(inputgen, tf.constant(True, dtype=bool), models.params.ngf, f, f, 1, 1, 0.02,
                                           name="c1")

        o_c2 = models.layers.general_conv2d(o_c1, tf.constant(True, dtype=bool), models.params.ngf * 2, ks, ks, 2, 2,
                                           0.02, padding='same', name="c2")

        o_c3 = models.layers.general_conv2d(o_c2, tf.constant(True, dtype=bool), models.params.ngf * 4, ks, ks, 2, 2,
                                           0.02, padding='same', name="c3")

        o_r1 = build_resnet_block(o_c3, models.params.ngf * 4, "r1", padding)
        o_r2 = build_resnet_block(o_r1, models.params.ngf * 4, "r2", padding)
        o_r3 = build_resnet_block(o_r2, models.params.ngf * 4, "r3", padding)
        o_r4 = build_resnet_block(o_r3, models.params.ngf * 4, "r4", padding)
        o_r5 = build_resnet_block(o_r4, models.params.ngf * 4, "r5", padding)
        o_r6 = build_resnet_block(o_r5, models.params.ngf * 4, "r6", padding)
        o_r7 = build_resnet_block(o_r6, models.params.ngf * 4, "r7", padding)
        o_r8 = build_resnet_block(o_r7, models.params.ngf * 4, "r8", padding)
        o_r9 = build_resnet_block(o_r8, models.params.ngf * 4, "r9", padding)

        o_c4 = models.layers.general_deconv2d(
            o_r9, [models.params.BATCH_SIZE, 128, 128, models.params.ngf * 2], models.params.ngf * 2, ks, ks, 2, 2, 0.02,
            "SAME", "c4")

        o_c5 = models.layers.general_deconv2d(o_c4, [models.params.BATCH_SIZE, 256, 256, models.params.ngf],
                                             models.params.ngf, ks, ks, 2, 2, 0.02, "SAME", "c5")

        o_c6 = models.layers.general_conv2d(o_c5, tf.constant(False, dtype=bool), models.params.IMG_CHANNELS, f, f, 1, 1,
                                           0.02, "SAME", "c6", do_relu=False)

        if skip is True:
            out_gen = tf.nn.tanh(inputgen + o_c6, "t1")
        else:
            out_gen = tf.nn.tanh(o_c6, "t1")

        return out_gen


def BBOX(thresh):

    img, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return img, contours, hierarchy


def discriminator(inputdisc, mask, transition_rate, donorm, name="discriminator"):

    with tf.variable_scope(name):
        mask = tf.cast(tf.greater_equal(mask, transition_rate), tf.float32)
        inputdisc = tf.multiply(inputdisc, mask)
        f = 4
        padw = 2
        pad_input = tf.pad(inputdisc, [[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT")

        o_c1 = models.layers.general_conv2d(pad_input, donorm, models.params.ndf, f, f, 2, 2, 0.02, "VALID", "c1",
                                           relufactor=0.2)

        pad_o_c1 = tf.pad(o_c1, [[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT")

        o_c2 = models.layers.general_conv2d(pad_o_c1, donorm, models.params.ndf * 2, f, f, 2, 2, 0.02, "VALID", "c2",
                                           relufactor=0.2)

        pad_o_c2 = tf.pad(o_c2, [[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT")

        o_c3 = models.layers.general_conv2d(pad_o_c2, donorm, models.params.ndf * 4, f, f, 2, 2, 0.02, "VALID", "c3",
                                           relufactor=0.2)

        pad_o_c3 = tf.pad(o_c3, [[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT")

        o_c4 = models.layers.general_conv2d(pad_o_c3, donorm, models.params.ndf * 8, f, f, 1, 1, 0.02, "VALID", "c4",
                                           relufactor=0.2)

        pad_o_c4 = tf.pad(o_c4, [[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT")

        o_c5 = models.layers.general_conv2d(pad_o_c4, tf.constant(False, dtype=bool), 1, f, f, 1, 1, 0.02, "VALID", "c5",
                                           do_relu=False)
        return o_c5


def get_weight(shape, gain=np.sqrt(2)):

    fan_in = np.prod(shape[:-1])
    std = gain / np.sqrt(fan_in)
    w = tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal(0, std))

    return w


def apply_bias(x):

    b = tf.get_variable('bias', shape=[x.shape[1]], initializer=tf.initializers.zeros())
    b = tf.cast(b, x.dtype)
    if len(x.shape) == 2:
        return x + b

    return x + tf.reshape(b, [1, -1, 1, 1])


def Pconv2d_bias(x, fmaps, kernel, mask_in=None):

    assert kernel >= 1 and kernel % 2 == 1
    x = tf.pad(x, [[0, 0], [0, 0], [1, 1], [1, 1]], "SYMMETRIC")
    mask_in = tf.pad(mask_in, [[0, 0], [0, 0], [1, 1], [1, 1]], "CONSTANT", constant_values=1)
    conv, mask = PConv2D(fmaps, kernel, strides=1, padding='valid',
                         data_format='channels_first')([x, mask_in])
    return conv, mask


def conv2d_bias(x, fmaps, kernel, gain=np.sqrt(2)):

    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain)
    w = tf.cast(w, x.dtype)
    x = tf.pad(x, [[0, 0], [0, 0], [1, 1], [1, 1]], "SYMMETRIC")

    return apply_bias(tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='VALID', data_format='NCHW'))


def Pmaxpool2d(x, k=2, mask_in=None):

    ksize = [1, 1, k, k]
    x = tf.nn.max_pool(x, ksize=ksize, strides=ksize, padding='SAME', data_format='NCHW')
    mask_out = tf.nn.max_pool(mask_in, ksize=ksize, strides=ksize, padding='SAME', data_format='NCHW')

    return x, mask_out


def maxpool2d(x, k=2):

    ksize = [1, 1, k, k]

    return tf.nn.max_pool(x, ksize=ksize, strides=ksize, padding='SAME', data_format='NCHW')


def upscale2d(x, factor=2):

    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Upscale2D'):
        s = x.shape
        x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
        x = tf.tile(x, [1, 1, 1, factor, 1, factor])
        x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])

        return x


def conv_lr(name, x, fmaps, p=0.7):

    with tf.variable_scope(name):
        x = tf.nn.dropout(x, p)

        return tf.nn.leaky_relu(conv2d_bias(x, fmaps, 3), alpha=0.1)


def conv(name, x, fmaps, p):

    with tf.variable_scope(name):
        x = tf.nn.dropout(x, p)

        return tf.nn.sigmoid(conv2d_bias(x, fmaps, 3, gain=1.0))


def Pconv_lr(name, x, fmaps, mask_in):

    with tf.variable_scope(name):
        x_out, mask_out = Pconv2d_bias(x, fmaps, 3, mask_in=mask_in)

        return tf.nn.leaky_relu(x_out, alpha=0.1), mask_out


def autoencoder(x, mask, channel=3, width=256, height=256, p=0.7, **_kwargs):

    x.set_shape([None, channel, height, width])
    mask.set_shape([None, channel, height, width])
    skips = [x]

    # For encoder
    n = x
    n, mask = Pconv_lr('enc_conv0', n, 48, mask_in=mask)
    n, mask = Pmaxpool2d(n, mask_in=mask)
    skips.append(n)

    n, mask = Pconv_lr('enc_conv2', n, 48, mask_in=mask)
    n, mask = Pmaxpool2d(n, mask_in=mask)
    skips.append(n)

    n, mask = Pconv_lr('enc_conv3', n, 48, mask_in=mask)
    n, mask = Pmaxpool2d(n, mask_in=mask)
    skips.append(n)

    n, mask = Pconv_lr('enc_conv4', n, 48, mask_in=mask)
    n, mask = Pmaxpool2d(n, mask_in=mask)
    skips.append(n)

    n, mask = Pconv_lr('enc_conv6', n, 48, mask_in=mask)

    # For decoder
    n = upscale2d(n)
    n = concat(n, skips.pop())
    n = conv_lr('dec_conv5', n, 96, p=p)

    n = upscale2d(n)
    n = concat(n, skips.pop())
    n = conv_lr('dec_conv4', n, 96, p=p)

    n = upscale2d(n)
    n = concat(n, skips.pop())
    n = conv_lr('dec_conv3', n, 96, p=p)

    n = upscale2d(n)
    n = concat(n, skips.pop())
    n = conv_lr('dec_conv2', n, 96, p=p)

    n = upscale2d(n)
    n = concat(n, skips.pop())
    n = conv_lr('dec_conv1a', n, 64, p=p)
    n = conv_lr('dec_conv1b', n, 32, p=p)
    n = conv('dec_conv1', n, channel, p=p)

    return n


def concat(x, y):

    bs1, c1, h1, w1 = x.shape.as_list()
    bs2, c2, h2, w2 = y.shape.as_list()
    x = tf.image.crop_to_bounding_box(tf.transpose(x, [0, 2, 3, 1]), 0, 0, min(h1, h2), min(w1, w2))
    y = tf.image.crop_to_bounding_box(tf.transpose(y, [0, 2, 3, 1]), 0, 0, min(h1, h2), min(w1, w2))

    return tf.transpose(tf.concat([x, y], axis=3), [0, 3, 1, 2])


def build_denoising_unet(noisy, p=0.7, is_realnoisy=False):

    _, h, w, c = np.shape(noisy)
    noisy_tensor = tf.identity(noisy)
    is_flip_lr = tf.placeholder(tf.int16)
    is_flip_ud = tf.placeholder(tf.int16)
    noisy_tensor = data_arg(noisy_tensor, is_flip_lr, is_flip_ud)
    response = tf.transpose(noisy_tensor, [0, 3, 1, 2])
    mask_tensor = tf.ones_like(response)
    mask_tensor = tf.nn.dropout(mask_tensor, 0.7) * 0.7
    response = tf.multiply(mask_tensor, response)
    slice_avg = tf.get_variable('slice_avg', shape=[_, h, w, c], initializer=tf.initializers.zeros())

    if is_realnoisy:
        response = tf.squeeze(tf.random_poisson(25 * response, [1]) / 25, 0)

    response = autoencoder(response, mask_tensor, channel=c, width=w, height=h, p=p)
    response = tf.transpose(response, [0, 2, 3, 1])
    mask_tensor = tf.transpose(mask_tensor, [0, 2, 3, 1])
    data_loss = models.losses.mask_loss(response, noisy_tensor, 1. - mask_tensor)
    response = data_arg(response, is_flip_lr, is_flip_ud)
    avg_op = slice_avg.assign(slice_avg * 0.99 + response * 0.01)
    our_image = response

    training_error = data_loss
    tf.summary.scalar('data loss', data_loss)

    merged = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=3)
    model = {
        'training_error': training_error,
        'data_loss': data_loss,
        'saver': saver,
        'summary': merged,
        'our_image': our_image,
        'is_flip_lr': is_flip_lr,
        'is_flip_ud': is_flip_ud,
        'avg_op': avg_op,
        'slice_avg': slice_avg,
    }

    return model


def data_arg(x, is_flip_lr, is_flip_ud):
    
    x = tf.cond(is_flip_lr > 0, lambda: tf.image.flip_left_right(x), lambda: x)
    x = tf.cond(is_flip_ud > 0, lambda: tf.image.flip_up_down(x), lambda: x)

    return x
