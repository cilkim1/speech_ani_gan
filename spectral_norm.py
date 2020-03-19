import tensorflow as tf
import numpy as np
import models
slim = tf.contrib.slim


weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
# weight_init = xavier_initializer()
# weight_init = variance_scaling_initializer()
# weight_regularizer = tf_contrib.layers.l2_regularizer(scale=0.0001)
weight_regularizer = None


def spectral_norm(w, name=None, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])
    if name is not None:
        u = tf.get_variable(name+"u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)
    else:
        u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)
    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """

        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


def conv(x, channels, kernel, stride, rate=1, padding=None, sn=True, activation_fn=None, data_format='NCHW', scope='conv', bias=False, weight_init_ones=False):
    with tf.variable_scope(scope):
        if sn:
            if padding is None:
                padding_l = np.asarray(np.rint((kernel - 1) / 2), dtype=np.int32).item() + (rate - 1)
                padding_r = np.asarray(np.rint((kernel - 1.5) / 2), dtype=np.int32).item() + (rate - 1)
            else:
                padding_l = padding
                padding_r = padding

            x = tf.pad(x, [[0, 0], [0, 0], [padding_l, padding_r], [padding_l, padding_r]])
            if weight_init_ones:
                w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[1], channels], initializer=tf.initializers.ones(),
                                    regularizer=weight_regularizer)
            else:
                w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[1], channels],
                                    initializer=weight_init,
                                    regularizer=weight_regularizer)
            x = tf.nn.conv2d(input=x, filter=spectral_norm(w), strides=[1, 1, stride, stride],
                             dilations=[1, 1, rate, rate], padding='VALID', data_format='NCHW')
            if bias is True:
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias, data_format=data_format)
        else:
            x = slim.conv2d(x, channels, kernel, stride, rate=rate, activation_fn=activation_fn, data_format='NCHW')
    return x


def t_conv(x, channels, kernel, stride, rate=1, out_hw=None, sn=True, activation_fn=None, data_format='NCHW', scope='conv', bias=False):
    with tf.variable_scope(scope):
        if sn:
            pre_batch = tf.shape(x)[0]
            pre_ch = x.get_shape()[1].value
            pre_h, pre_w = x.get_shape()[2].value, x.get_shape()[3].value
            w = tf.get_variable("kernel", shape=[kernel, kernel, channels, pre_ch], initializer=weight_init,
                                regularizer=weight_regularizer)
            if out_hw is None:
                out_shape = [pre_batch, channels, pre_h * stride, pre_w * stride]
            else:
                out_shape = [pre_batch, channels, out_hw[0], out_hw[1]]
                stride = int(out_hw[0]/pre_h)
            x = tf.nn.conv2d_transpose(value=x, filter=spectral_norm(w), output_shape=out_shape, strides=[1, 1, stride, stride], padding='SAME', data_format='NCHW')
            if bias is True:
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias, data_format=data_format)
        else:
            x = slim.conv2d(x, channels, kernel, stride, rate=rate, activation_fn=activation_fn, data_format='NCHW')
    return x


def conv_ns(x, channels, kernel, stride, rate=1, sn=False, activation_fn=None, data_format='NCHW', scope='conv'):
    with tf.variable_scope(scope):
        if sn:
            padding_x = kernel[0] - 1 + rate-1
            padding_y = kernel[1] - 1 + rate-1
            x = tf.pad(x, [[0, 0], [0, 0], [padding_x, padding_y], [padding_x, padding_y]])
            w = tf.get_variable("kernel", shape=[kernel[0], kernel[1], x.get_shape()[1], channels], initializer=weight_init, regularizer=weight_regularizer)
            bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
            x = tf.nn.conv2d(input=x, filter=spectral_norm(w), strides=[1, 1, stride, stride], dilations=[1, 1, rate, rate], padding='VALID', data_format='NCHW')
            x = tf.nn.bias_add(x, bias, data_format=data_format)
        else:
            x = slim.conv2d(x, channels, kernel, stride, rate=rate, activation_fn=activation_fn, data_format=data_format)
    return x


def conv_rate(x, channels, kernel, stride, rate=[3, 2], sn=False, activation_fn=None, data_format='NCHW', scope='conv'):
    with tf.variable_scope(scope):
        if sn:
            padding_x = np.asarray(np.rint((kernel[0]-1)/2), dtype=np.int32).item() + (rate[0] - 1)
            padding_y = np.asarray(np.rint((kernel[1] - 1) / 2), dtype=np.int32) + (rate[1] - 1)
            x = tf.pad(x, [[0, 0], [0, 0], [padding_x, padding_y], [padding_x, padding_y]])
            w = tf.get_variable("kernel", shape=[kernel[0], kernel[1], x.get_shape()[1], channels], initializer=weight_init, regularizer=weight_regularizer)
            bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
            x = tf.nn.conv2d(input=x, filter=spectral_norm(w), strides=[1, 1, stride, stride], dilations=[1, 1, rate[0], rate[1]], padding='VALID', data_format='NCHW')
            x = tf.nn.bias_add(x, bias, data_format=data_format)
        else:
            x = slim.conv2d(x, channels, kernel, stride, rate=rate, activation_fn=activation_fn, data_format=data_format)
    return x


def gated_conv(x, cnum, ksize, stride=1, rate=1, name='gated_conv',
     padding='SAME', activation=tf.nn.relu, training=True):
    """Define gated conv for generator. Add a gating filter
    Args:
        x: Input.
        cnum: Channel number.
        ksize: Kernel size.
        Stride: Convolution stride.
        Rate: Rate for or dilated conv.
        name: Name of layers.
        padding: Default to SYMMETRIC.
        activation: Activation function after convolution.
        training: If current graph is for training or inference, used for bn.
    Returns:
        tf.Tensor: output
    """
    assert padding in ['SYMMETRIC', 'SAME', 'REFELECT']
    if padding == 'SYMMETRIC' or padding == 'REFELECT':
        p = int(rate*(ksize-1)/2)
        x = tf.pad(x, [[0,0], [0,0], [p, p], [p, p]], mode=padding)
        padding = 'VALID'
    xin = x
    x = tf.layers.conv2d(
        xin, cnum, ksize, stride, dilation_rate=rate,
        activation=activation, padding=padding, name=name, data_format='NCHW')

    gated_mask = tf.layers.conv2d(
        xin, cnum, ksize, stride, dilation_rate=rate,
        activation=tf.nn.sigmoid, padding=padding, name=name+"_mask", data_format='NCHW')

    return x * gated_mask


def fully_connected(x, channels, sn=True, activation_fn=None, scope='fully', bias=False, weight_init_ones=False):
    with tf.variable_scope(scope):
        if sn:
            if weight_init_ones:
                w = tf.get_variable("kernel", shape=[x.get_shape()[1], channels], initializer=tf.initializers.ones(),
                                    regularizer=weight_regularizer)
            else:
                w = tf.get_variable("kernel", shape=[x.get_shape()[1], channels], initializer=weight_init,
                                    regularizer=weight_regularizer)
            x = tf.matmul(x, w)
            if bias is True:
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)
        else:
            x = slim.fully_connected(x, channels, activation_fn=activation_fn)
    return x


def contextual_attention(f, b, mask=None, ksize=3, stride=1, rate=1,
                         fuse_k=3, softmax_scale=10., training=True, fuse=True):
    """ Contextual attention layer implementation.
    Contextual attention is first introduced in publication:
        Generative Image Inpainting with Contextual Attention, Yu et al.
    Args:
        x: Input feature to match (foreground).
        t: Input feature for match (background).
        mask: Input mask for t, indicating patches not available.
        ksize: Kernel size for contextual attention.
        stride: Stride for extracting patches from t.
        rate: Dilation for matching.
        softmax_scale: Scaled softmax for attention.
        training: Indicating if current graph is training or inference.
    Returns:
        tf.Tensor: output
    """
    # get shapes
    raw_fs = tf.shape(f)
    #batch_size = raw_fs[0]
    raw_int_fs = f.get_shape().as_list()
    raw_int_bs = b.get_shape().as_list()
    # extract patches from background with stride and rate
    kernel = 2*rate
    raw_w = tf.extract_image_patches(
        b, [1,kernel,kernel,1], [1,rate*stride,rate*stride,1], [1,1,1,1], padding='SAME')
    raw_w = tf.reshape(raw_w, [raw_int_bs[0], -1, kernel, kernel, raw_int_bs[3]])
    raw_w = tf.transpose(raw_w, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    # downscaling foreground option: downscaling both foreground and
    # background for matching and use original background for reconstruction.
    f = resize(f, scale=1./rate, func=tf.compat.v1.image.resize_nearest_neighbor)
    b = resize(b, to_shape=[int(raw_int_bs[1]/rate), int(raw_int_bs[2]/rate)], func=tf.compat.v1.image.resize_nearest_neighbor)  # https://github.com/tensorflow/tensorflow/issues/11651
    if mask is not None:
        mask = resize(mask, scale=1./rate, func=tf.compat.v1.image.resize_nearest_neighbor)
    fs = tf.shape(f)
    int_fs = f.get_shape().as_list()
    f_groups = tf.split(f, int_fs[0], axis=0)
    # from t(H*W*C) to w(b*k*k*c*h*w)
    bs = tf.shape(b)
    int_bs = b.get_shape().as_list()
    w = tf.extract_image_patches(
        b, [1,ksize,ksize,1], [1,stride,stride,1], [1,1,1,1], padding='SAME')
    w = tf.reshape(w, [int_fs[0], -1, ksize, ksize, int_fs[3]])
    w = tf.transpose(w, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    # process mask
    if mask is None:
        mask = tf.zeros([1, bs[1], bs[2], 1])
    ms = tf.shape(mask)
    batch_size = ms[0]
    m = tf.extract_image_patches(
        mask, [1, ksize,ksize,1], [1,stride,stride,1], [1,1,1,1], padding='SAME')
    m = tf.reshape(m, [batch_size, -1, ksize, ksize, 1])
    m = tf.transpose(m, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    m = m[0]
    mm = tf.cast(tf.equal(tf.reduce_mean(m, axis=[0,1,2], keep_dims=True), 0.), tf.float32)
    w_groups = tf.split(w, int_bs[0], axis=0)
    raw_w_groups = tf.split(raw_w, int_bs[0], axis=0)
    y = []
    offsets = []
    k = fuse_k
    scale = softmax_scale
    fuse_weight = tf.reshape(tf.eye(k), [k, k, 1, 1])
    for xi, wi, raw_wi in zip(f_groups, w_groups, raw_w_groups):
        # conv for compare
        wi = wi[0]
        wi_normed = wi / tf.maximum(tf.sqrt(tf.reduce_sum(tf.square(wi), axis=[0,1,2])), 1e-4)
        yi = tf.nn.conv2d(xi, wi_normed, strides=[1,1,1,1], padding="SAME")

        # conv implementation for fuse scores to encourage large patches
        if fuse:
            yi = tf.reshape(yi, [1, fs[1]*fs[2], bs[1]*bs[2], 1])
            yi = tf.nn.conv2d(yi, fuse_weight, strides=[1,1,1,1], padding='SAME')
            yi = tf.reshape(yi, [1, fs[1], fs[2], bs[1], bs[2]])
            yi = tf.transpose(yi, [0, 2, 1, 4, 3])
            yi = tf.reshape(yi, [1, fs[1]*fs[2], bs[1]*bs[2], 1])
            yi = tf.nn.conv2d(yi, fuse_weight, strides=[1,1,1,1], padding='SAME')
            yi = tf.reshape(yi, [1, fs[2], fs[1], bs[2], bs[1]])
            yi = tf.transpose(yi, [0, 2, 1, 4, 3])
        yi = tf.reshape(yi, [1, fs[1], fs[2], bs[1]*bs[2]])

        # softmax to match
        yi *=  mm  # mask
        yi = tf.nn.softmax(yi*scale, 3)
        yi *=  mm  # mask

        offset = tf.argmax(yi, axis=3, output_type=tf.int32)
        offset = tf.stack([offset // fs[2], offset % fs[2]], axis=-1)
        # deconv for patch pasting
        # 3.1 paste center
        wi_center = raw_wi[0]
        yi = tf.nn.conv2d_transpose(yi, wi_center, tf.concat([[1], raw_fs[1:]], axis=0), strides=[1,rate,rate,1]) / 4.
        y.append(yi)
        offsets.append(offset)
    y = tf.concat(y, axis=0)
    y.set_shape(raw_int_fs)
    offsets = tf.concat(offsets, axis=0)
    offsets.set_shape(int_bs[:3] + [2])
    # case1: visualize optical flow: minus current position
    h_add = tf.tile(tf.reshape(tf.range(bs[1]), [1, bs[1], 1, 1]), [bs[0], 1, bs[2], 1])
    w_add = tf.tile(tf.reshape(tf.range(bs[2]), [1, 1, bs[2], 1]), [bs[0], bs[1], 1, 1])
    offsets = offsets - tf.concat([h_add, w_add], axis=3)
    # to flow image
    flow = flow_to_image_tf(offsets)
    # # case2: visualize which pixels are attended
    # flow = highlight_flow_tf(offsets * tf.cast(mask, tf.int32))
    if rate != 1:
        flow = resize(flow, scale=rate, func=tf.compat.v1.image.resize_nearest_neighbor)
    return y, flow


def flow_to_image(flow):
    """Transfer flow map to image.
    Part of code forked from flownet.
    """
    out = []
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    maxrad = -1
    for i in range(flow.shape[0]):
        u = flow[i, :, :, 0]
        v = flow[i, :, :, 1]
        idxunknow = (abs(u) > 1e7) | (abs(v) > 1e7)
        u[idxunknow] = 0
        v[idxunknow] = 0
        maxu = max(maxu, np.max(u))
        minu = min(minu, np.min(u))
        maxv = max(maxv, np.max(v))
        minv = min(minv, np.min(v))
        rad = np.sqrt(u ** 2 + v ** 2)
        maxrad = max(maxrad, np.max(rad))
        u = u/(maxrad + np.finfo(float).eps)
        v = v/(maxrad + np.finfo(float).eps)
        img = compute_color(u, v)
        out.append(img)
    return np.float32(np.uint8(out))


def flow_to_image_tf(flow, name='flow_to_image'):
    """Tensorflow ops for computing flow to image.
    """
    with tf.variable_scope(name), tf.device('/cpu:0'):
        img = tf.py_func(flow_to_image, [flow], tf.float32, stateful=False)
        img.set_shape(flow.get_shape().as_list()[0:-1]+[3])
        img = img / 127.5 - 1.
        return img


def make_color_wheel():
    RY, YG, GC, CB, BM, MR = (15, 6, 4, 11, 13, 6)
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros([ncols, 3])
    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


COLORWHEEL = make_color_wheel()


def compute_color(u,v):
    h, w = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0
    # colorwheel = COLORWHEEL
    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)
    rad = np.sqrt(u**2+v**2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a+1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(int)
    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0
    for i in range(np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)
        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))
    return img


def resize(x, scale=2, to_shape=None, align_corners=True, dynamic=False,
           func=tf.compat.v1.image.resize_bilinear, name='resize'):
    if dynamic:
        xs = tf.cast(tf.shape(x), tf.float32)
        new_xs = [tf.cast(xs[1]*scale, tf.int32),
                  tf.cast(xs[2]*scale, tf.int32)]
    else:
        xs = x.get_shape().as_list()
        new_xs = [int(xs[1]*scale), int(xs[2]*scale)]
    with tf.variable_scope(name):
        if to_shape is None:
            x = func(x, new_xs, align_corners=align_corners)
        else:
            x = func(x, [to_shape[0], to_shape[1]],
                     align_corners=align_corners)
    return x