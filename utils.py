from __future__ import print_function

import os
import math
import json
import logging
import numpy as np
import tensorflow as tf
from PIL import Image
from datetime import datetime
from scipy import linalg
import moviepy.editor as mpe


def prepare_dirs_and_logger(config):
    formatter = logging.Formatter("%(asctime)s:%(levelname)s::%(message)s")
    logger = logging.getLogger()

    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    if config.load_path:
        if config.load_path.startswith(config.log_dir):
            config.model_dir = config.load_path
        else:
            if config.load_path.startswith(config.dataset):
                config.model_name = config.load_path
            else:
                config.model_name = "{}_{}".format(config.dataset, config.load_path)
    else:
        config.model_name = "{}_{}".format(config.dataset, get_time())

    if not hasattr(config, 'model_dir'):
        config.ckpt_dir = os.path.join(config.check_dir, config.model_name)
    config.data_path = os.path.join(config.data_dir[0], config.dataset)
    config.ckpt_dir = os.path.join(config.check_dir, config.model_name)
    config.model_dir = os.path.join(config.log_dir, config.model_name)

    for path in [config.log_dir, config.data_dir[0], config.model_dir, config.ckpt_dir]:
        if not os.path.exists(path):
            os.makedirs(path)


def get_time():
    return datetime.now().strftime("%m%d_%H%M%S")


def save_config(config):
    param_path = os.path.join(config.model_dir, "params.json")

    print("[*] MODEL dir: %s" % config.model_dir)
    print("[*] PARAM path: %s" % param_path)

    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)


def rank(array):
    return len(array.shape)


def make_grid(tensor, nrow=6, padding=2,
              normalize=False, scale_each=False):
    """Code based on https://github.com/pytorch/vision/blob/master/torchvision/utils.py"""
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[1] + padding), int(tensor.shape[2] + padding)
    grid = np.zeros([height * ymaps + 1 + padding // 2, width * xmaps + 1 + padding // 2, 3], dtype=np.uint8)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            h, h_width = y * height + 1 + padding // 2, height - padding
            w, w_width = x * width + 1 + padding // 2, width - padding

            grid[h:h+h_width, w:w+w_width] = tensor[k]
            k = k + 1
    return grid


def save_image(tensor, filename, nrow=8, padding=0,
               normalize=False, scale_each=False):
    ndarr = make_grid(tensor, nrow=nrow, padding=padding,
                            normalize=normalize, scale_each=scale_each)
    im = Image.fromarray(ndarr)
    im.save(filename)


def instance_norm(x, i, init_gamma_zero=False, noise=False):
    with tf.variable_scope('ins_norm_{}'.format(i)):
        if noise is True:
            add_noise(x, i)
        temp_mean, temp_std = tf.nn.moments(x, axes=[2, 3], keep_dims=True)
        temp_std = tf.sqrt(temp_std + 1.e-6)
        x = (x - temp_mean) * temp_std
        if init_gamma_zero is True:
            gamma = tf.get_variable('gamma_{}'.format(i), shape=[x.shape[1].value], initializer=tf.initializers.zeros())
        else:
            gamma = tf.get_variable('gamma_{}'.format(i), shape=[x.shape[1].value], initializer=tf.initializers.ones())
        # gamma = sp.spectral_norm(gamma, 'gamma__')
        beta = tf.get_variable('beta_{}'.format(i), shape=[x.shape[1].value], initializer=tf.initializers.zeros())
        # beta = sp.spectral_norm(beta, 'bata__')
        x = x * tf.reshape(tf.cast(gamma, x.dtype), [1, -1, 1, 1]) + tf.reshape(tf.cast(beta, x.dtype), [1, -1, 1, 1])
    return x


def int_shape(tensor):
    shape = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in shape]


def get_conv_shape(tensor, data_format):
    shape = int_shape(tensor)
    # always return [N, H, W, C]
    if data_format == 'NCHW':
        return [shape[0], shape[2], shape[3], shape[1]]
    elif data_format == 'NHWC':
        return shape


def nchw_to_nhwc(x):
    return tf.transpose(x, [0, 2, 3, 1])


def nhwc_to_nchw(x):
    return tf.transpose(x, [0, 3, 1, 2])


def reshape(x, h, w, c, data_format):
    if data_format == 'NCHW':
        x = tf.reshape(x, [-1, c, h, w])
    else:
        x = tf.reshape(x, [-1, h, w, c])
    return x


def resize_nearest_neighbor(x, new_size, data_format):
    if data_format == 'NCHW':
        x = nchw_to_nhwc(x)
        x = tf.image.resize_nearest_neighbor(x, new_size)
        x = nhwc_to_nchw(x)
    else:
        x = tf.image.resize_nearest_neighbor(x, new_size)
    return x

def resize_bil(x, new_size, data_format):
    if data_format == 'NCHW':
        x = nchw_to_nhwc(x)
        x = tf.image.resize_bilinear(x, new_size)
        x = nhwc_to_nchw(x)
    else:
        x = tf.image.resize_bilinear(x, new_size)
    return x


def upscale(x, scale, data_format):
    _, h, w, _ = get_conv_shape(x, data_format)
    return resize_nearest_neighbor(x, tf.cast((h * scale, w * scale), dtype=tf.int32), data_format)


def downscale(x, scale, data_format):
    _, h, w, _ = get_conv_shape(x, data_format)
    return resize_nearest_neighbor(x, tf.cast((tf.round(h / scale), tf.round(w / scale)), dtype=tf.int32), data_format)


def upscale_to(x, new_size, data_format):
    _, h, w, _ = get_conv_shape(x, data_format)
    return resize_nearest_neighbor(x, tf.cast((new_size[0], new_size[1]), dtype=tf.int32), data_format)


def upscale_to_bil(x, new_size, data_format):
    _, h, w, _ = get_conv_shape(x, data_format)
    return resize_bil(x, tf.cast((new_size[0], new_size[1]), dtype=tf.int32), data_format)


def add_noise(x, i, is_zero=False):
    with tf.variable_scope("add_noise_{}".format(i)):
        noise = tf.random_normal([tf.shape(x)[0], 1, tf.shape(x)[2], tf.shape(x)[3]], mean=0., stddev=1., dtype=x.dtype)
        if is_zero:
            weight = tf.get_variable('weight_{}'.format(i), shape=[x.shape[1].value], initializer=tf.initializers.zeros())
        else:
            weight = tf.get_variable('weight_{}'.format(i), shape=[x.shape[1].value], initializer=tf.initializers.ones())
        # weight = sp.spectral_norm(weight, name='noise_w')
    return x + noise * tf.reshape(tf.cast(weight, x.dtype), [1, -1, 1, 1])


def add_given_noise(x, z, i):
    with tf.variable_scope("add_noise_{}".format(i)):
        weight = tf.get_variable('weight_{}'.format(i), shape=[x.shape[1].value], initializer=tf.initializers.ones())
        # weight = sp.spectral_norm(weight, name='noise_w')
    return x + z * tf.reshape(tf.cast(weight, x.dtype), [1, -1, 1, 1])


def add_noise_given_mask(x, i, mask):
    with tf.variable_scope("mul_noise_{}".format(i)):
        noise = tf.random_normal([1, 1, tf.shape(x)[2], tf.shape(x)[3]], mean=0., stddev=1., dtype=x.dtype)
        weight = tf.get_variable('weight_{}'.format(i), shape=[x.shape[1].value], initializer=tf.initializers.ones())
        mask = upscale_to(mask, [x.shape[2].value, x.shape[3].value], 'NCHW')
    return x + noise * mask * tf.reshape(tf.cast(weight, x.dtype), [1, -1, 1, 1])


def concat_noise(x, i):
    with tf.variable_scope("concat_noise{}".format(i)):
        noise = tf.random_normal([tf.shape(x)[0], 1, tf.shape(x)[2], tf.shape(x)[3]], mean=0.5, stddev=0.5, dtype=x.dtype)
        weight = tf.get_variable('weight_{}'.format(i), shape=[], initializer=tf.initializers.zeros())
    return tf.concat([x, noise * tf.reshape(tf.cast(weight, x.dtype), [1, -1, 1, 1])], 1)


def standardization(x):
    return x/(tf.reduce_mean(x) + 1.e-6)


def noise_modulation(x, i, is_zero=False):
    with tf.variable_scope("add_noise_{}".format(i)):
        noise = tf.random_normal([2, tf.shape(x)[0], 1, tf.shape(x)[2], tf.shape(x)[3]], mean=0., stddev=1., dtype=x.dtype)
        if is_zero:
            weight = tf.get_variable('weight_{}'.format(i), shape=[x.shape[1].value], initializer=tf.initializers.zeros())
        else:
            weight = tf.get_variable('weight_{}'.format(i), shape=[x.shape[1].value], initializer=tf.initializers.ones())
        # weight = sp.spectral_norm(weight, name='noise_w')
        gamma_weight = tf.get_variable('gamma_{}'.format(i), shape=[x.shape[1].value], initializer=tf.initializers.ones())
    return x + noise[0, :, :, :, :] * tf.reshape(tf.cast(weight, x.dtype), [1, -1, 1, 1]) + x * noise[1, :, :, :, :] * tf.reshape(tf.cast(gamma_weight, x.dtype), [1, -1, 1, 1])


def save_video(x, fixed, path):
    np.save(x, path + ".mp4")
    my_clip = mpe.VideoFileClip(path + ".mp4")
    audio_background = mpe.AudioFileClip(fixed)
    final_audio = mpe.CompositeAudioClip([my_clip.audio, audio_background])
    final_clip = my_clip.set_audio(final_audio)