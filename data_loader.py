import tensorflow as tf
import os
import glob
import cv2
import tensorflow_datasets as tfds
import librosa
import numpy as np
import moviepy.editor as mpe


def get_loader(path, batch_size):
    with tf.compat.v1.variable_scope("tfData"):
        # print(path)
        audio = glob.glob("{}/*.{}".format(path[0], "npy"))
        video = glob.glob("{}/*.{}".format(path[1], "npy"))
        check_file_shape([audio, video])
        whole_queue_0 = tf.data.Dataset.from_tensor_slices((audio, video))
        whole_queue = tf.data.Dataset.zip(whole_queue_0)
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        whole_queue = whole_queue.shuffle(buffer_size=1001)
        whole_queue = whole_queue.repeat()
        whole_queue = whole_queue.map(load_and_preprocess_video_audio, num_parallel_calls=AUTOTUNE)
        whole_queue = whole_queue.batch(batch_size)
        whole_queue = whole_queue.prefetch(buffer_size=AUTOTUNE)
    return whole_queue


def load_and_preprocess_video_audio(audio, video):
    # audio = preprocess_audio(audio)  # -> wav
    audio = tf.numpy_function(read_npy_file, [audio], tf.float32)
    video = tf.numpy_function(read_npy_file, [video], tf.float32)
    audio = tf.reshape(audio, [298])
    audio = tf.concat([audio, [0., 0.]], 0)
    video = tf.reshape(video, [32, 40, 75])
    video, _ = tf.split(video, [32, -1], axis=1)
    video = tf.expand_dims(video, -2)
    return audio, video


def preprocess_audio(audio):
    audio = tf.io.read_file(audio)
    audio, _ = tf.audio.decode_wav(audio, desired_samples=300)
    return audio


def read_npy_file(video):
    file = np.load(video)
    return file.astype(np.float32)


def read_liborsa_file(audio):
    y, sr = librosa.load(audio, sr=100)
    # y = mpe.AudioFileClip(audio)
    return y


def check_file_shape(x):
    print('================')
    print('total_audio_length :' + str(len(x[0])))
    print('total_video_length :' + str(len(x[1])))
    print(x[0][0], x[1][0])
    # y, sr = librosa.load(x[0][0], sr=100)
    # print('audio shape: ' + str(y.shape[0]) + ", sample rate: " + str(sr))  # 298, 100 not this # 65664, 22050
    file = np.load(x[0][0])
    print('video shape: ' + str(file.shape))  # 298
    file = np.load(x[1][0])
    print('video shape: ' + str(file.shape))  # 288, 360, 3, 75  # 64, 80, 3, 75  # 64, 80, 1, 75
    print('================')