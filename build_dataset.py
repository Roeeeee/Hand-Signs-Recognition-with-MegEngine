# *_*coding:utf-8 *_*# *_*coding:utf-8 *_*
import os
import os.path as osp
import sys
sys.path.append(".")
import cv2
import time
import argparse
import numpy as np

from find_homography import run_global_homo, make_gif
from utils import homo2cropRatio


# 读取不稳定视频，计算F --> motion， warp稳定视频的一帧


def get_args():
    parser = argparse.ArgumentParser(
        description='Offline Video Stabilization',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--root_dir", type=str, default="./data")
    parser.add_argument("--video_type", type=int, default=0)
    parser.add_argument("--number", type=int,default=0)
    parser.add_argument("--radius", type=int, default=10)
    parser.add_argument("--smooth", action='store_true', default=False)

    return parser.parse_args()


def read_video_frame(video):
    ret, frame = video.read()
    if ret:
        frame = frame.transpose(2, 0, 1)
        # any transformation on a single frame
    else:
        frame = None
    return ret, frame


def read_video_data(video_path):
    video = cv2.VideoCapture(video_path)
    n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f''' Video Information
          Frames: {n_frames}
          FPS:    {fps}
          Width:  {width}
          Height: {height}
    ''')
    
    frame_data = np.zeros((n_frames*3, height, width), dtype=np.uint8)
    for i in range(n_frames):
        _, frame = read_video_frame(video)
        if frame is not None:
            frame_data[3*i:3*i+3, :, :] = frame
    video.release()

    return frame_data


def get_homo_array(frame_data):
    """
    Calculate inverse homography of adjacent frames.
    """
    n_frame = int(frame_data.shape[0] / 3)
    homo_array = np.zeros((n_frame-3, 3, 3), dtype=np.float32)
    
    for i in range(n_frame - 3): # regular 0.avi有两个空白帧
        tmp_path_1 = "./tmp/frame_{}.jpg".format(i)
        tmp_path_2 = "./tmp/frame_{}.jpg".format(i+1)
        cv2.imwrite(tmp_path_1, frame_data[3*i:3*i+3, :, :].transpose(1, 2, 0))
        cv2.imwrite(tmp_path_2, frame_data[3*(i+1):3*(i+1)+3, :, :].transpose(1, 2, 0))
        homo = run_global_homo(tmp_path_2, tmp_path_1, i)
        homo_array[i, :, :] = homo
    return homo_array


def calculate_window_trans(homo_array, index, left, right):
    """Calculate transformation among a fixed-size window.
    Params:
        homo_array: homography array of all frames.
        index: the current frame to be transformed.
        left: the left boundary of the window.
        right: the right boundary of the window.
    Return: 
        The mean value of all transformations.
    """
    buffer = np.zeros((right - left + 1, 3, 3))
    cur = left
    i = 0
    while cur <= right:
        buffer[i, :, :] = calculate_single_trans(homo_array, cur, index)
        i += 1
        cur += 1
    return np.mean(buffer, axis=0)


def calculate_single_trans(homo_array, index, target):
    """Calculate the single transformation of the 'index' frame to the 'target' frame.
    Params: 
        homo_array: homography array of all frames.
        index: the current frame.
        target: the target frame.
    """
    if index < target:
        single_trans = homo_array[index, :, :]
        for i in range(index+1, target):
            single_trans = np.dot(single_trans, homo_array[i, :, :])
            single_trans /= single_trans[-1, -1]
        single_trans = np.linalg.inv(single_trans)
    elif index > target:
        single_trans = homo_array[target, :, :] # homo of frame target->target+1
        for i in range(target+1, index):
            single_trans = np.dot(single_trans, homo_array[i, :, :])
        # single_trans = np.linalg.inv(single_trans)
        single_trans /= single_trans[-1, -1]
    else: 
        single_trans = np.eye(3, 3, dtype=np.float32)

    return single_trans



def get_motion_trans(homo_array, radius):
    """ Calculate windows-based camera motion transformation of all video frames.
    Params:
        homo_array: homography array of all frames. (len(homo_array) = len(video_frames) - 1)
        radius: the radius of the window, the windows size is [-radius, radius]
    """
    n_frames = int(homo_array.shape[0])
    trans = np.zeros((n_frames+1, 3, 3))
    for i in range(n_frames+1):
        # calculate bi for the i-th frame
        l = max(0, i - radius)
        r = min(n_frames, i + radius)
        b_i = calculate_window_trans(homo_array, i, l ,r)
        b_i = np.linalg.inv(b_i)
        trans[i, :, :] = b_i
    return trans


def calculate_camera_path_trans(trans_array):
    """Calculate camera path transformation according to the iterative transformation array.
    """
    n_frames, iter_times, _, _ = trans_array.shape
    camera_path_trans = trans_array[:, 0, :, :].copy()
    for i in range(n_frames):
        for j in range(1, iter_times):
            camera_path_trans[i, :, :] = np.dot(camera_path_trans[i, :, :], trans_array[i, j, :, :])

            camera_path_trans[i, :, :] /= camera_path_trans[i, -1, -1]
    return camera_path_trans



def perform_generate_data(src_path, dst_path, trans):

    # video information
    src_video = cv2.VideoCapture(src_path)
    n_frames = int(src_video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = src_video.get(cv2.CAP_PROP_FPS)
    fourcc = src_video.get(cv2.CAP_PROP_FOURCC)
    width = int(src_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(src_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # video size, if need crop
    video_size = (int(width), int(height))
    stab_video = cv2.VideoWriter(dst_path, int(fourcc), fps, video_size)

    # perform stabilization
    # for i in range(n_frames-1):
    for i in range(300):
        _, src_frame = src_video.read()
        stab_frame = cv2.warpPerspective(src_frame, trans[i, :, :], video_size, flags=cv2.INTER_CUBIC)
        # crop border
        border = 80
        crop_stab_frame = stab_frame[border:height-border, border:width-border, :]
        crop_stab_frame = cv2.resize(crop_stab_frame, (width, height), interpolation=cv2.INTER_CUBIC)

        stab_video.write(crop_stab_frame)

    src_video.release()
    src_video.release()
    




def main():
    args = get_args()
    video_types = {
        0: "Regular",
        1: "QuickRotation",
        2: "Zooming",
        3: "Parallax",
        4: "Crowd",
        5: "Running",
        6: "Others"
    }

    unstab_video_path = "./data/QuickRotation/1.avi"
    stab_video_path = "./data/QuickRotation/2stb.avi"
    generated_unstab_video_path = "./data/QuickRotation/2unstb_from1.avi"
    print("==> Calculating homography...")
    unstab_homo_dir = osp.join('./data/QuickRotation', '1_homo.npy')
    stab_homo_dir = osp.join('./data/QuickRotation', '2stab_homo.npy')
    if not osp.exists(unstab_homo_dir):
        unstab_frame_data = read_video_data(unstab_video_path)
        unstab_homo_array = get_homo_array(unstab_frame_data)
        np.save(unstab_homo_dir, unstab_homo_array)
    else:
        unstab_homo_array = np.load(unstab_homo_dir)
    if not osp.exists(stab_homo_dir):
        stab_frame_data = read_video_data(stab_video_path)
        stab_homo_array = get_homo_array(stab_frame_data)
        np.save(stab_homo_dir, stab_homo_array)
    else:
        stab_homo_array = np.load(stab_homo_dir)

    unstab_trans = get_motion_trans(unstab_homo_array, args.radius)
    perform_generate_data(stab_video_path, generated_unstab_video_path, unstab_trans)



if __name__ == "__main__":
    main()
