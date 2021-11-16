import tensorflow as tf


def cycle_consistency_loss(real_images, generated_images):
    
    return tf.reduce_mean(tf.abs(real_images - generated_images))


def mask_loss(gen_image, mask):

    return tf.reduce_mean(tf.abs(tf.multiply(gen_image,1-mask)))


def lsgan_loss_generator(prob_fake_is_real):
   
    return tf.reduce_mean(tf.squared_difference(prob_fake_is_real, 1))


def lsgan_loss_discriminator(prob_real_is_real, prob_fake_is_real):
    
    return (tf.reduce_mean(tf.squared_difference(prob_real_is_real, 1)) +tf.reduce_mean(tf.squared_difference(prob_fake_is_real, 0))) * 0.5

def mask_loss(x, labels, masks):
    
    cnt_nonzero = tf.to_float(tf.count_nonzero(masks))
    loss = tf.reduce_sum(tf.multiply(tf.math.pow(x - labels, 2), masks)) / cnt_nonzero

    return loss