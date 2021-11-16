import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D
from keras.engine import InputSpec
from keras.utils import conv_utils


def lrelu(x, leak = 0.2, name  ="lrelu", alt_relu_impl  =False):

    with tf.variable_scope(name):
        if alt_relu_impl:
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            return f1 * x + f2 * abs(x)
        else:
            return tf.maximum(x, leak * x)


def instance_norm(x):

    with tf.variable_scope("instance_norm"):
        epsilon = 1e-5
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        scale = tf.get_variable('scale', [x.get_shape()[-1]],
                                initializer=tf.truncated_normal_initializer(
                                    mean=1.0, stddev=0.02
        ))
        offset = tf.get_variable(
            'offset', [x.get_shape()[-1]],
            initializer=tf.constant_initializer(0.0)
        )
        out = scale * tf.div(x - mean, tf.sqrt(var + epsilon)) + offset

        return out


def instance_norm_bis(x,mask):

    with tf.variable_scope("instance_norm"):
        epsilon = 1e-5
        for i in range(x.shape[-1]):
            slice = tf.gather(x, i, axis=3)
            slice_mask = tf.gather(mask, i, axis=3)
            tmp = tf.boolean_mask(slice,slice_mask)
            mean, var = tf.nn.moments_bis(x, [1, 2], keep_dims=False)

        mean, var = tf.nn.moments_bis(x, [1, 2], keep_dims=True)
        scale = tf.get_variable('scale', [x.get_shape()[-1]],
                                initializer=tf.truncated_normal_initializer(
                                    mean=1.0, stddev=0.02
        ))
        offset = tf.get_variable(
            'offset', [x.get_shape()[-1]],
            initializer=tf.constant_initializer(0.0)
        )
        out = scale * tf.div(x - mean, tf.sqrt(var + epsilon)) + offset

        return out


def general_conv2d_(inputconv, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1, stddev=0.02,
                   padding="VALID", name="conv2d", do_norm=True, do_relu=True,
                   relufactor=0):
    with tf.variable_scope(name):

        conv = tf.contrib.layers.conv2d(
            inputconv, o_d, f_w, s_w, padding,
            activation_fn=None,
            weights_initializer=tf.truncated_normal_initializer(
                stddev=stddev
            ),
            biases_initializer=tf.constant_initializer(0.0)
        )
        if do_norm:
            conv = instance_norm(conv)

        if do_relu:
            if(relufactor == 0):
                conv = tf.nn.relu(conv, "relu")
            else:
                conv = lrelu(conv, relufactor, "lrelu")

        return conv


def general_conv2d(inputconv, do_norm, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1, stddev=0.02,
                   padding="VALID", name="conv2d", do_relu=True,
                   relufactor=0):
    with tf.variable_scope(name):
        conv = tf.contrib.layers.conv2d(
            inputconv, o_d, f_w, s_w, padding,
            activation_fn=None,
            weights_initializer=tf.truncated_normal_initializer(
                stddev=stddev
            ),
            biases_initializer=tf.constant_initializer(0.0))

        conv = tf.cond(do_norm, lambda: instance_norm(conv), lambda: conv)

        if do_relu:
            if(relufactor == 0):
                conv = tf.nn.relu(conv, "relu")
            else:
                conv = lrelu(conv, relufactor, "lrelu")

        return conv


def general_deconv2d(inputconv, outshape, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1,
                     stddev=0.02, padding="VALID", name="deconv2d",
                     do_norm=True, do_relu=True, relufactor=0):
    with tf.variable_scope(name):

        conv = tf.contrib.layers.conv2d_transpose(
            inputconv, o_d, [f_h, f_w],
            [s_h, s_w], padding,
            activation_fn=None,
            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
            biases_initializer=tf.constant_initializer(0.0)
        )

        if do_norm:
            conv = instance_norm(conv)

        if do_relu:
            if(relufactor == 0):
                conv = tf.nn.relu(conv, "relu")
            else:
                conv = lrelu(conv, relufactor, "lrelu")

        return conv


def upsamplingDeconv(inputconv, size, is_scale, method,align_corners, name):

    if len(inputconv.get_shape()) == 3:
        if is_scale:
            size_h = size[0] * int(inputconv.get_shape()[0])
            size_w = size[1] * int(inputconv.get_shape()[1])
            size = [int(size_h), int(size_w)]
    elif len(inputconv.get_shape()) == 4:
        if is_scale:
            size_h = size[0] * int(inputconv.get_shape()[1])
            size_w = size[1] * int(inputconv.get_shape()[2])
            size = [int(size_h), int(size_w)]
    else:
        raise Exception("Donot support shape %s" % inputconv.get_shape())
    print("  [TL] UpSampling2dLayer %s: is_scale:%s size:%s method:%d align_corners:%s" %
          (name, is_scale, size, method, align_corners))
    with tf.variable_scope(name) as vs:
        try:
            out = tf.image.resize_images(inputconv, size=size, method=method, align_corners=align_corners)
        except:  
            out = tf.image.resize_images(inputconv, new_height=size[0], new_width=size[1], method=method,
                                                  align_corners=align_corners)

    return out


def general_fc_layers(inpfc, outshape, name):
    with tf.variable_scope(name):

        fcw = tf.Variable(tf.truncated_normal(outshape,
                                               dtype=tf.float32,
                                               stddev=1e-1), name='weights')
        fcb = tf.Variable(tf.constant(1.0, shape=[outshape[-1]], dtype=tf.float32),
                           trainable=True, name='biases')

        fcl = tf.nn.bias_add(tf.matmul(inpfc, fcw), fcb)
        fc_out = tf.nn.relu(fcl)

        return fc_out


class PConv2D(Conv2D):

    def __init__(self, *args, n_channels=3, mono=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_spec = [InputSpec(ndim=4), InputSpec(ndim=4)]

    def build(self, input_shape):
        """
        Adapted from original _Conv() layer of Keras param input_shape: list of dimensions for [img, mask]
        """

        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1

        if input_shape[0][channel_axis] is None:
            raise ValueError('The channel dimension of the inputs should be defined. Found `None`.')

        self.input_dim = input_shape[0][channel_axis]

        # Image kernel
        kernel_shape = self.kernel_size + (self.input_dim, self.filters)
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='img_kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        # Mask kernel
        self.kernel_mask = K.ones(shape=self.kernel_size + (self.input_dim, self.filters))

        # Calculate padding size to achieve zero-padding
        self.pconv_padding = (
            (int((self.kernel_size[0] - 1) / 2), int((self.kernel_size[0] - 1) / 2)),
            (int((self.kernel_size[0] - 1) / 2), int((self.kernel_size[0] - 1) / 2)),
        )

        # Window size - used for normalization
        self.window_size = self.kernel_size[0] * self.kernel_size[1]

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, mask=None):
        """"
        We're going to multiply the input times the mask, before we apply the
        convolutions. 
        """

        # Both image and mask must be supplied
        if type(inputs) is not list or len(inputs) != 2:
            raise Exception(
                'PartialConvolution2D must be called on a list of two tensors [img, mask]. Instead got: ' + str(inputs))

        # Padding done explicitly so that padding becomes part of the masked partial convolution
        images = K.spatial_2d_padding(inputs[0], self.pconv_padding, self.data_format)
        masks = K.spatial_2d_padding(inputs[1], self.pconv_padding, self.data_format)

        # Apply convolutions to mask
        mask_output = K.conv2d(
            masks, self.kernel_mask,
            strides=self.strides,
            padding='valid',
            data_format=self.data_format,
            dilation_rate=self.dilation_rate
        )

        # Apply convolutions to image
        img_output = K.conv2d(
            (images * masks), self.kernel,
            strides=self.strides,
            padding='valid',
            data_format=self.data_format,
            dilation_rate=self.dilation_rate
        )

        # Calculate the mask ratio on each pixel in the output mask
        mask_ratio = self.window_size / (mask_output + 1e-8)

        # Clip output to be between 0 and 1
        mask_output = K.clip(mask_output, 0, 1)

        # Remove ratio values where there are holes
        mask_ratio = mask_ratio * mask_output

        # Normalize iamge output
        img_output = img_output * mask_ratio

        # Apply bias only to the image (if chosen to do so)
        if self.use_bias:
            img_output = K.bias_add(
                img_output,
                self.bias,
                data_format=self.data_format)

        # Apply activations on the image
        if self.activation is not None:
            img_output = self.activation(img_output)

        return [img_output, mask_output]

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[0][1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding='same',
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            new_shape = (input_shape[0][0],) + tuple(new_space) + (self.filters,)
            return [new_shape, new_shape]
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding='same',
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            new_shape = (input_shape[0], self.filters) + tuple(new_space)

            return [new_shape, new_shape]
