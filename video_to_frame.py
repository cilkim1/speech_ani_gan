import os
import cv2
import glob

path = "D:/Private Studies/GRID dataset/s1.mpg_vcd/s1"
root = glob.glob("{}/*.{}".format(path, "mpg"))
for i in range(len(root)):
    vidcap = cv2.VideoCapture(root[i])
    success,image = vidcap.read()
    count = 0
    current_name, _ = os.path.splitext(os.path.basename(root[i]))
    file_path = "{}/{}".format(path, current_name)
    # print(file_path)
    if os.path.isdir(file_path) is False:
        os.mkdir(file_path)
    while success:
      cv2.imwrite("{}/{}/{number:05}.jpg".format(path, current_name, number=count),image)     # save frame as JPEG file
      success,image = vidcap.read()
      count += 1