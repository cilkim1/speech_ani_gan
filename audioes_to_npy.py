import os
import cv2
import glob
import numpy as np
import librosa

path = "D:/Private Studies/GRID dataset/s1.mpg_vcd/s1"
new_path = "D:/Private Studies/GRID dataset/s1.mpg_vcd/s1_audio_npy"
root = glob.glob("{}/*.{}".format(path, "mpg"))
if os.path.isdir(new_path) is False:
    os.mkdir(new_path)
for i in range(len(root)):
    current_name, _ = os.path.splitext(os.path.basename(root[i]))
    y, sr = librosa.load(root[i], sr=100)
    # print(total_image.shape, type(total_image))
    np.save("{}/{}.npy".format(new_path, current_name), y)
