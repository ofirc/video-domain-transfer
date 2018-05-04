"""Preprocess alignment step for input frames to StarGAN.

Problem statement: when applying StarGAN on datasets different than
CelebA, we get undesired jittering / compression effects.
Possible reason is that CelebA is already aligned and our dataset isn't.

e.g. python align_to_CelebA.py \
  --average_face_file mean_CelebA_1000_samples_952_actual_2018-02-21_19_2_1.csv \
  --frames c:\stargan_results\orig_frames

The output will be aligned images (to StarGAN average face)
I1', ... , In'.

Pipeline:
1. Find average face
   Run find_face_landmarks on CelebA, find average pixel location
   for each landmark.
   This yields average face, i.e. average positions for each landmark.
   We will use this for determining the "alignment matrix" for each image.

2. Split video to frames (optional)
   If input is video, otherwise use frames.

3. Determine and apply alignment matrix (A)
   Between each frame I1, ..., In and `I_AverageFace`.
   Apply the alignment matrix and generate I1', ... , In'.

-- end of preprocessing --

4. Evaluate aligned images quality
   Apply StarGAN on I1', ... , In', this yields J1', ... , Jn'.
   (use filter old and change gender).
   Report quality.

   Expected: less jittering effects than before.

5. Evaluate original pose quality
   Apply A^-1 to J1', ... , Jn' to restore original pose & view.
   Report quality.
   Expected: less jittering effects than before.

"""
import os
import sys

import numpy as np

from argparse import ArgumentParser

from scipy.misc import imread

from face_landmarks import find_landmarks, read_landmarks
from similarity_matrix import get_alignment
from utils import get_frame_paths, apply_alignment, dump_alignment_matrix


def parse_args(argv):
    parser = ArgumentParser(description="Preprocesses input images to StarGAN.")
    parser.add_argument("--video",
                        help="Input video file.")
    parser.add_argument("--frames",
                        help="Input frames directory.")
    parser.add_argument("--skip_average_face",
                        type=int,
                        default=1,
                        help="Skips finding average landmarks locations from CelebA.")
    parser.add_argument("--average_face_file",
                        required=True,
                        help="Use the average face from a file.")
    args = parser.parse_args(argv)

    if not args.video and not args.frames:
        parser.error('you must pass either --video or --frames.')

    if args.video:
        parser.error('--video is not implemented.')

    if args.skip_average_face == 0:
        parser.error('Average face is not implemented.')

    return args, parser


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    args, parser = parse_args(argv)
    avg_face_landmarks = read_landmarks(args.average_face_file)
    frame_paths = get_frame_paths(args.frames)
    num_frames = len(frame_paths)
    print("Processing {} frames".format(num_frames))
    failed_frames = []
    for i, frame in enumerate(frame_paths):
        print("[{}/{}] {}".format(i + 1, num_frames, frame))
        try:
            frame_landmarks = find_landmarks(frame)
        except RuntimeError as e:
            failed_frames.append(frame)
            continue

        img = imread(frame)

        A = get_alignment(frame_landmarks, avg_face_landmarks)
        dir_ = os.path.dirname(frame)
        path, ext = os.path.splitext(frame)
        path = os.path.basename(path)
        align_path = os.path.join(dir_, 'aligned', path + '.csv')
        dump_alignment_matrix(A, align_path)

        landmarks_path = os.path.join(dir_, 'aligned', path + '_landmarks.csv')
        with open(landmarks_path, 'w') as f:
            f.write('x,y\n')
            for l in frame_landmarks:
                f.write('%s,%s\n'% (l[0], l[1]))

        A_minus_one = np.linalg.pinv(A)
        align_path = os.path.join(dir_, 'aligned', path + '_minus_one.csv')
        dump_alignment_matrix(A_minus_one, align_path)

        I_hat = apply_alignment(A[:-1], frame)

    print("number of failed frames:", len(failed_frames))
    for f in failed_frames: print(f)
    print("Done.")


if __name__ == "__main__":
    sys.exit(main())
