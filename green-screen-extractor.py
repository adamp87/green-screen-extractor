"""
This script takes as input a video and extracts detected person and head from frames.
It expects a green-screen background, which will be masked out in output frames.
Script uses yolov5 trained on crowdhuman from https://github.com/deepakcrk/yolov5-crowdhuman.
Script is licensed under GPL (yolov5) for non-commercial (crowdhuman) use.
"""

import os
import argparse
from typing import Any

import cv2
import torch
import numpy as np
from tqdm import tqdm


def green_screen_mask(img: np.ndarray) -> np.ndarray:
    """Perform green-screen subtraction in HSV space and perform closing morph operation on mask"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # mask = cv2.inRange(img, (30, 30, 0), (104, 153, 70))  # rgb
    mask = cv2.inRange(hsv, (36, 55, 55), (86, 255, 255))  # hsv
    mask = np.bitwise_not(mask)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)

    return mask


def extract_video(
        src_path: str,
        dst_path: str,
        detector: Any,
        skip_frame_count: int,
        frame_prefix_idx: int
) -> None:
    # init opencv input
    cap = cv2.VideoCapture(src_path)
    frame_start = 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_idx in tqdm(range(frame_start, frame_count, skip_frame_count)):
        # read frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, bgr_img = cap.read()
        if not success:
            break

        # inference, detect person and head
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        results = detector(rgb_img)

        # extract results
        for i, det in enumerate(results.xywh[0]):
            det = det.to(int)
            box_size = min((det[2], det[3]))
            box_size = torch.div(box_size, 2, rounding_mode='floor')
            if box_size < 16:
                continue  # skip small bbox
            x_min = det[0] - box_size - int(box_size * 0.2)
            y_min = det[1] - box_size - int(box_size * 0.2)
            x_max = det[0] + box_size + int(box_size * 0.2)
            y_max = det[1] + box_size + int(box_size * 0.2)
            class_name = results.names[det[5]]

            # get detection roi and write result
            filename = f"{frame_prefix_idx:0>4}_{frame_idx:0>4}_{i}.png"
            roi = bgr_img[y_min:y_max, x_min:x_max, :]
            if roi.size == 0:
                continue
            cv2.imwrite(os.path.join(dst_path, class_name, filename), roi)

            # subtract green-screen background and write result
            mask = green_screen_mask(roi)
            roi_mask = np.append(roi, mask[:, :, None], axis=2)
            cv2.imwrite(os.path.join(dst_path, class_name+"_mask", filename), roi_mask)


def create_yolov5_crowdhuman(ckpt_path: str, use_gpu: bool) -> Any:
    model = torch.hub.load('deepakcrk/yolov5-crowdhuman', 'yolov5m', pretrained=False, classes=2)

    # copy from hubconf.py
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))  # load
    state_dict = ckpt['model'].float().state_dict()  # to FP32
    state_dict = {k: v for k, v in state_dict.items() if model.state_dict()[k].shape == v.shape}  # filter
    model.load_state_dict(state_dict, strict=False)  # load
    if len(ckpt['model'].names) == 2:
        model.names = ckpt['model'].names  # set class names attribute

    # ##model.half() # cant half
    model.fuse()
    if use_gpu:
        model = model.to(torch.device('cuda'))
    model = model.autoshape()
    model.conf = 0.8  # detection confidence

    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, help='source, file/folder, 0 for webcam', required=True)
    parser.add_argument('--dst_path', type=str, help='destination, folder', required=True)
    parser.add_argument('--ckpt_path', type=str, help='crowdhuman_yolov5m.pt path', required=True)
    parser.add_argument('--no_gpu', action='store_false', help='disable inference on gpu')
    parser.add_argument('--skip_frame_count', type=int, default=5, help='skip frames in input video')
    parser.add_argument('--video_prefix_int', type=int, default=0, help='prefix for extracted single video')
    args = parser.parse_args()

    if not os.path.exists(args.ckpt_path):
        print("Please download crowdhuman_yolov5m.pt from https://github.com/deepakcrk/yolov5-crowdhuman")
        exit(-1)

    use_gpu = True if args.no_gpu else False
    detector = create_yolov5_crowdhuman(args.ckpt_path, use_gpu)

    # set up extraction folders
    os.makedirs(args.dst_path, exist_ok=True)
    for class_name in detector.names:
        os.makedirs(os.path.join(args.dst_path, class_name), exist_ok=True)
        os.makedirs(os.path.join(args.dst_path, class_name+"_mask"), exist_ok=True)

    if os.path.isdir(args.src_path):
        vid_filenames = os.listdir(args.src_path)
        for idx, vid_filename in tqdm(enumerate(vid_filenames)):
            args.video_prefix_int = int(os.path.splitext(vid_filename)[0])
            if args.video_prefix_int == 0:
                args.video_prefix_int = idx  # video file name is not a number, use enumerate
            vid_path = os.path.join(args.src_path, vid_filename)
            try:
                extract_video(vid_path, args.dst_path, detector, args.skip_frame_count, args.video_prefix_int)
            except RuntimeError as e:
                print("Failed to extract {0}: {1}".format(vid_filename, str(e)))
    else:
        extract_video(args.src_path, args.dst_path, detector, args.skip_frame_count, args.video_prefix_int)


if __name__ == "__main__":
    main()
