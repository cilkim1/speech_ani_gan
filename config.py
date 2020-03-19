#-*- coding: utf-8 -*-
import argparse


def str2bool(v):
    return v.lower() in ('true', '1')


arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# Network


net_arg = add_argument_group('Network')

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--dataset', type=str, default='')
data_arg.add_argument('--data_dir', type=str, default=['D:/Private Studies/GRID dataset/s1.mpg_vcd/s1_audio_npy', 'D:/Private Studies/GRID dataset/s1.mpg_vcd/s1_video_npy'])
data_arg.add_argument('--batch_size', type=int, default=1)
data_arg.add_argument('--grayscale', type=str2bool, default=False)
data_arg.add_argument('--num_worker', type=int, default=4)

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--is_train', type=str2bool, default=True)
train_arg.add_argument('--model_dir', type=str, default='')
train_arg.add_argument('--optimizer', type=str, default='adam')
train_arg.add_argument('--max_step', type=int, default=6000000)
train_arg.add_argument('--d_lr', type=float, default=0.0001)
train_arg.add_argument('--g_lr', type=float, default=0.0001)
train_arg.add_argument('--beta1', type=float, default=0.9)
train_arg.add_argument('--beta2', type=float, default=0.999)
train_arg.add_argument('--use_gpu', type=str2bool, default=True)

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--load_path', type=str, default='')
misc_arg.add_argument('--log_step', type=int, default=1000)
misc_arg.add_argument('--save_step', type=int, default=2000)
misc_arg.add_argument('--num_log_samples', type=int, default=3)
misc_arg.add_argument('--log_level', type=str, default='INFO', choices=['INFO', 'DEBUG', 'WARN'])
misc_arg.add_argument('--log_dir', type=str, default='logs')
misc_arg.add_argument('--check_dir', type=str, default='check')
misc_arg.add_argument('--pre_dir', type=str, default='pretrained')
misc_arg.add_argument('--test_data_path', type=str, default=None,
                      help='directory with images which will be used in test sample generation')
misc_arg.add_argument('--sample_per_image', type=int, default=16,
                      help='# of sample per image during test sample generation')
misc_arg.add_argument('--random_seed', type=int, default=123)


def get_config():
    config, unparsed = parser.parse_known_args()
    if config.use_gpu:
        data_format = 'NCHW'
    else:
        data_format = 'NHWC'
    setattr(config, 'data_format', data_format)
    return config, unparsed
