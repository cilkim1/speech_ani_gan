from __future__ import print_function

import os

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
from tqdm import trange
from models import *
import config as conf
import librosa
import skvideo.io


class Trainer(object):
    def __init__(self, config, get_loader):

        self.config = config
        self.get_loader = get_loader
        self.dataset = config.dataset

        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.optimizer = config.optimizer
        self.batch_size = config.batch_size

        self.step = tf.Variable(0, name='step', trainable=False)

        self.g_lr = tf.Variable(config.g_lr, name='g_lr')
        self.d_lr = tf.Variable(config.d_lr, name='d_lr')

        self.model_dir = config.model_dir
        self.load_path = config.load_path
        self.summary_path = os.path.join(config.ckpt_dir)
        self.pre_dir = config.data_dir[0] + '/' + config.pre_dir

        self.use_gpu = config.use_gpu
        self.data_format = config.data_format

        self.start_step = 0
        self.log_step = config.log_step
        self.max_step = config.max_step
        self.save_step = config.save_step
        self.k_t = tf.Variable(0., trainable=False)

        data = self.get_loader
        self.iterator = data.make_initializable_iterator()
        self.data = self.iterator.get_next()

        self.is_train = config.is_train
        self.build_model()

        self.saver = tf.compat.v1.train.Saver()
        self.summary_writer = tf.compat.v1.summary.FileWriter(self.summary_path)

        sv = tf.compat.v1.train.Supervisor(logdir=self.model_dir,
                                 is_chief=True,
                                 saver=self.saver,
                                 summary_op=None,
                                 summary_writer=self.summary_writer,
                                 save_model_secs=1200,
                                 global_step=self.step,
                                 ready_for_local_init_op=None)

        gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
        sess_config = tf.compat.v1.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options,
                                               intra_op_parallelism_threads=6,
                                               inter_op_parallelism_threads=12)

        self.sess = sv.prepare_or_wait_for_session(config=sess_config)

        if not self.is_train:
            g = tf.get_default_graph()
            g._finalized = False

    def train(self):
        self.initial_iterator()
        fixed = 'D:/Private Studies/GRID dataset/s1.mpg_vcd/s1_mpg/bbaf2n.mpg'
        face = skvideo.io.vread(fixed)
        wav, _ = librosa.load(fixed, sr=100)

        for step in trange(self.start_step, self.max_step):
            fetch_dict = {
                'fd': self.fd_optim,
                'sd': self.sd_optim,
                'g': self.g_optim,
            }
            if step % self.log_step == 0:
                fetch_dict.update({
                    "summary": self.summary_op,
                    "fd_loss": self.FD_loss,
                    "sd_loss": self.SD_loss,
                    "g_loss": self.G_loss,
                })
            result = self.sess.run(fetch_dict)

            if step % self.log_step == 0:
                self.summary_writer.add_summary(result['summary'], step)
                self.summary_writer.flush()
                fd_loss = result['fd_loss']
                sd_loss = result['sd_loss']
                g_loss = result['g_loss']
                print(
                    "[{}/{}] Loss_FD: [{:.6f}] Loss_SD: [{:.6f}] || Loss_G: [{:.6f}]" \
                        .format(step, self.max_step, fd_loss, sd_loss, g_loss))

            if step % self.save_step == 0:
                self.generate(face, fixed, self.model_dir, idx=step)

    def build_model(self):
        # with tf.compat.v1.device('/gpu:0'):
        with tf.compat.v1.variable_scope('preprocessing'):
            wav, avi = self.preprocessing()

        with tf.compat.v1.variable_scope('model'):
            G, self.G_var = generator(wav, avi)  # [B, 64, 64, 3, 75]
            FD, self.FD_var = frame_discriminator(avi, G)  # [B, 50]
            SD, self.SD_var = sequence_discriminator(wav, avi, G)  # [B, 75]
            print(G, FD, SD)

        with tf.compat.v1.variable_scope("loss_design"):
            if self.optimizer == 'adam':
                optimizer = tf.compat.v1.train.AdamOptimizer
            else:
                raise Exception("[!] Caution! Paper didn't use {} opimizer other than Adam".format(conf.optimizer))

            with tf.compat.v1.variable_scope("optimizer"):
                g_optimizer = optimizer(self.g_lr)
                fd_optimizer = optimizer(self.d_lr)
                sd_optimizer = optimizer(self.d_lr)

            with tf.compat.v1.variable_scope("loss_function"):
                FD_real, FD_fake = tf.split(FD, 2, 0)
                FD_loss_real = - tf.reduce_mean(tf.math.log(tf.maximum(FD_real, 1.e-6)))
                FD_loss_fake = - tf.reduce_mean(tf.math.log(tf.maximum(1.-FD_fake, 1.e-6)))
                self.FD_loss = FD_loss_real + FD_loss_fake
                SD_real, SD_fake = tf.split(SD, 2, 0)
                SD_loss_real = - tf.reduce_mean(tf.math.log(tf.maximum(SD_real, 1.e-6)))
                SD_loss_fake = - tf.reduce_mean(tf.math.log(tf.maximum(1. - SD_fake, 1.e-6)))
                self.SD_loss = SD_loss_real + SD_loss_fake
                self.G_loss = -tf.reduce_mean(tf.math.log(tf.maximum(FD_fake, 1.e-6)))\
                              - tf.reduce_mean(tf.math.log(tf.maximum(SD_fake, 1.e-6)))
                abs_lower_face, _ = tf.split(tf.abs(G-avi), 2, 1)
                self.G_loss = 400. * tf.reduce_mean(abs_lower_face) + self.G_loss

            with tf.compat.v1.variable_scope("minimizer"):
                self.fd_optim = fd_optimizer.minimize(self.FD_loss, var_list=[self.FD_var])
                self.sd_optim = sd_optimizer.minimize(self.SD_loss, var_list=[self.SD_var])
                self.g_optim = g_optimizer.minimize(self.G_loss, var_list=[self.G_var])

        with tf.compat.v1.variable_scope("de_normal"):
            self.G = tf.clip_by_value((G + 1.) * 127.5, 0., 255.)
            self.avi = tf.clip_by_value((avi + 1.) * 127.5, 0., 255.)

        self.summary_op = tf.compat.v1.summary.merge([
            tf.compat.v1.summary.scalar("loss/fd_loss", self.FD_loss),
            tf.compat.v1.summary.scalar("loss/sd_loss", self.SD_loss),
            tf.compat.v1.summary.scalar("loss/g_loss", self.G_loss),
            # tf.compat.v1.summary.image("X", self.avi),
            # tf.compat.v1.summary.image("G", self.G),
        ])

    def build_test_model(self):
        print("test_model")

    def test(self):
        self.build_test_model()

    def initial_iterator(self):
        self.sess.run(self.iterator.initializer)

    def preprocessing(self):
        with tf.compat.v1.variable_scope("Pre"):
            wav, avi = self.data  # [B, 300]  [B, 288, 288, 3, 75]
            avi = self.norm(avi)
        return wav, avi

    def norm(self, x):
        return tf.clip_by_value((x/127.5)-1., -1., 1.)

    def load_data(self):
        self.initial_iterator()
        x = self.data
        return x

    def generate(self, image, fixed, root_path=None, path=None, idx=None, save=True):
        x = self.sess.run(self.Gz, {self.z: image})
        if path is None and save:
            path = os.path.join(root_path, '{}_G.mp4'.format(idx))
            utils.save_video(x, fixed, path)
            print("[*] Samples saved: {}".format(path))
        return x
