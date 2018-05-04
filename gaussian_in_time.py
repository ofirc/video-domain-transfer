'''Given a input folder of frames, apply gaussian filter in time.

Takes ~20 seconds per frame.

Input: Folder with video frames
Output: Frames after gaussian filter

Given frame i:
(1) Take frames i-2, i-1, i+1, i+2
(2) Run face landmarks on each of the 5 frames.
(3) Compute & apply similarity between face landmarks of each frame in (1) to frame i.
    Get 4 new frames: j-2, j-1, j+1, j+2
(4) Apply gaussian filter per pixel in the time domain.
    Meaning: Take face pixels from frame i and replace them with:
    Pixel(j-2)*0.06136 + Pixel(j-1)*0.24477 + Pixel(i)*0.38774 + Pixel(j+1)*0.24477 + Pixel(j+2)*0.06136
(5) Save the new frame to output folder

'''

import os
import sys

import cv2
import numpy as np

from argparse import ArgumentParser
from datetime import datetime

from face_swap import FaceSwap
from face_landmarks import find_landmarks, find_average_face, read_landmarks, NUM_LANDMARKS
from similarity_matrix import get_alignment
from utils import apply_alignment_impl

def parse_args(argv):
    parser = ArgumentParser(description="Video smoothing by Gaussian filter")
    parser.add_argument("--in_frames",
                        required=True,
                        help="Input frames directory.")
    parser.add_argument("--out_frames",
                        required=True,
                        help="Output frames directory.")
    # parser.add_argument("--skip_average_face",
    #                     type=int,
    #                     default=1,
    #                     help="Skips finding average landmarks locations from CelebA.")
    # parser.add_argument("--average_face_file",
    #                     #meta="FILE",
    #                     required=True,
    #                     help="Use the average face from a file.")
    args = parser.parse_args(argv)

    return args, parser

def find_seg(frame_path):
    """Returns a numpy array of the segmented image.

    Arguments:
        frame_path - path to an input frame.
    """
    fs = FaceSwap()
    fs.run(frame_path, frame_path)
    return fs.face_seg()


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    args, parser = parse_args(argv)

    if not os.path.exists(args.out_frames):
        os.mkdir(args.out_frames)

    input_frames = [os.path.join(args.in_frames, f) for f in os.listdir(args.in_frames)]

    start = datetime.now()
    i = 2
    for in_frame in input_frames[2:-2]:
        start_iter = datetime.now()
        window_frames = input_frames[i-2:i+3]

        # Find face landmarks
        window_frames_landmarks = np.empty((len(window_frames), NUM_LANDMARKS, 2))
        for idx, w_f in enumerate(window_frames):
            window_frames_landmarks[idx] = find_landmarks(w_f)

        # Compute similarity + apply
        neighbor_frames = np.delete(window_frames, 2, 0)
        img = cv2.imread(neighbor_frames[0])
        aligned_frames = np.empty((len(neighbor_frames), *img.shape))
        target_frame_landmarks = window_frames_landmarks[2]
        neighbor_frames_landmarks = np.delete(window_frames_landmarks, 2, 0)
        for j, nfl in enumerate(neighbor_frames_landmarks):
            A = get_alignment(nfl, target_frame_landmarks)
            img = cv2.imread(neighbor_frames[j])
            aligned_frames[j] = apply_alignment_impl(A[:-1], img)

        # TODO: Apply gaussian filter
        seg = find_seg(window_frames[2])
        seg = seg.astype(float) # needed for arithmetic operations below
        seg /= 255
        seg_inv = 1 - seg
        masks = [0.38774*seg + seg_inv, 0.24477*seg, 0.06136*seg]
        i_frame = cv2.imread(window_frames[2])
        filtered_frame = (aligned_frames[0] * masks[2] + # j - 2
                          aligned_frames[1] * masks[1] + # j - 1
                          i_frame           * masks[0] + # i
                          aligned_frames[2] * masks[1] + # j + 1
                          aligned_frames[3] * masks[2])  # j + 2

        # Save new image
        out_path = os.path.join(args.out_frames, os.path.basename(in_frame))
        cv2.imwrite(out_path, filtered_frame)

        i += 1
        end_iter = datetime.now()
        print("took %s for iteration, total time = %s" %
              (end_iter - start_iter, end_iter - start))


if __name__ == "__main__":
    sys.exit(main())
