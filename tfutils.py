import tensorflow as tf
import numpy as np


def w(shape, name):
    """
    @return A weight layer with the given shape and standard deviation. Initialized with a
            truncated normal distribution.
    """
    return tf.get_variable(name + '_w', shape, initializer=tf.contrib.layers.xavier_initializer())
    # return tf.Variable(tf.truncated_normal(shape, stddev=stddev), name='w')


def b(shape, const=0.1):
    """
    @return A bias layer with the given shape.
    """
    return tf.Variable(tf.constant(const, shape=shape), name='b')


def conv_out_size(i, p, k, s):
    """
    Gets the output size for a 2D convolution. (Assumes square input and kernel).

    @param i: The side length of the input.
    @param p: The padding type (either 'SAME' or 'VALID').
    @param k: The side length of the kernel.
    @param s: The stride.

    @type i: int
    @type p: string
    @type k: int
    @type s: int

    @return The side length of the output.
    """
    # convert p to a number
    if p == 'SAME':
        p = k // 2
    elif p == 'VALID':
        p = 0
    else:
        raise ValueError('p must be "SAME" or "VALID".')

    return int(((i + (2 * p) - k) / s) + 1)


def log10(t):
    """
    Calculates the base-10 log of each element in t.

    @param t: The tensor from which to calculate the base-10 log.

    @return: A tensor with the base-10 log of each element in t.
    """

    numerator = tf.log(t)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def unpool(value, name='unpool'):
    """N-dimensional version of the unpooling operation from
    https://www.robots.ox.ac.uk/~vgg/rg/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf

    :param name: Name for tensorflow graph
    :param value: A Tensor of shape [b, d0, d1, ..., dn, ch]
    :return: A Tensor of shape [b, 2*d0, 2*d1, ..., 2*dn, ch]
    """
    with tf.name_scope(name) as scope:
        sh = value.get_shape().as_list()
        dim = len(sh[1:-1])
        out = (tf.reshape(value, [-1] + sh[-dim:]))
        for i in range(dim, 0, -1):
            out = tf.concat([out, tf.zeros_like(out)], i)
        out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
        out = tf.reshape(out, out_size, name=scope)
    return out


def batch_pad_to_bounding_box(images, offset_height, offset_width, target_height, target_width):
    """
    Zero-pads a batch of images with the given dimensions.

    @param images: 4-D tensor with shape [batch_size, height, width, channels]
    @param offset_height: Number of rows of zeros to add on top.
    @param offset_width: Number of columns of zeros to add on the left.
    @param target_height: Height of output images.
    @param target_width: Width of output images.

    @return: The batch of images, all zero-padded with the specified dimensions.
    """
    batch_size, height, width, channels = tf.Session().run(tf.shape(images))

    if not offset_height >= 0:
        raise ValueError('offset_height must be >= 0')
    if not offset_width >= 0:
        raise ValueError('offset_width must be >= 0')
    if not target_height >= height + offset_height:
        raise ValueError('target_height must be >= height + offset_height')
    if not target_width >= width + offset_width:
        raise ValueError('target_width must be >= width + offset_width')

    num_tpad = offset_height
    num_lpad = offset_width
    num_bpad = target_height - (height + offset_height)
    num_rpad = target_width - (width + offset_width)

    tpad = np.zeros([batch_size, num_tpad, width, channels])
    bpad = np.zeros([batch_size, num_bpad, width, channels])
    lpad = np.zeros([batch_size, target_height, num_lpad, channels])
    rpad = np.zeros([batch_size, target_height, num_rpad, channels])

    padded = images
    if num_tpad > 0 and num_bpad > 0:
        padded = tf.concat([tpad, padded, bpad], 1)
    elif num_tpad > 0:
        padded = tf.concat([tpad, padded], 1)
    elif num_bpad > 0:
        padded = tf.concat([padded, bpad], 1)
    if num_lpad > 0 and num_rpad > 0:
        padded = tf.concat([lpad, padded, rpad], 2)
    elif num_lpad > 0:
        padded = tf.concat([lpad, padded], 2)
    elif num_rpad > 0:
        padded = tf.concat([padded, rpad], 2)

    return padded


def batch_crop_to_bounding_box(images, offset_height, offset_width, target_height, target_width):
    """
    Crops a batch of images to the given dimensions.

    @param images: 4-D tensor with shape [batch, height, width, channels]
    @param offset_height: Vertical coordinate of the top-left corner of the result in the input.
    @param offset_width: Horizontal coordinate of the top-left corner of the result in the input.
    @param target_height: Height of output images.
    @param target_width: Width of output images.

    @return: The batch of images, all cropped the specified dimensions.
    """
    batch_size, height, width, channels = tf.Session().run(tf.shape(images))

    if not offset_height >= 0:
        raise ValueError('offset_height must be >= 0')
    if not offset_width >= 0:
        raise ValueError('offset_width must be >= 0')
    if not target_height + offset_height <= height:
        raise ValueError('target_height + offset_height must be <= height')
    if not target_width <= width - offset_width:
        raise ValueError('target_width + offset_width must be <= width')

    top = offset_height
    bottom = target_height + offset_height
    left = offset_width
    right = target_width + offset_width

    return images[:, top:bottom, left:right, :]
