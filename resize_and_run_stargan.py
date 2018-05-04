"""Applies StarGAN to a video with rescaling.

StarGAN performs better when the input frames are inserted in lower resolution
(while keeping the original aspect ratio), e.g. 1920x1800 --> 505x284, or 917x516.

To run:
./resize_and_run_stargan.py --input_video <input video> --output_video [<output video>]

"""
import os
import sys

import cv2
import numpy as np
import face_detect
import face_swap
import gaussian_in_time

from argparse import ArgumentParser
from datetime import datetime

from img2video import split_to_frames, write_video
from stargan import run_stargan

STARGAN_OUTPUT_SIZE = 128
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

def parse_args(argv):
    parser = ArgumentParser(description="Evaluates face swap pixel alignment on video.")
    parser.add_argument("--input_video",
                        required=True,
                        help="Path to input video.")
    parser.add_argument("--output_video",
                        required=True,
                        help="Path to output video.")
    parser.add_argument("--skip_split_to_frames",
                        action='store_true',
                        default=False,
                        help="Skips input video split to frames.")
    parser.add_argument("--disable_full_run",
                        action='store_true',
                        default=False,
                        help="Partial run. Skip some steps. This flag is meant for development / debug.")

    args = parser.parse_args(argv)

    return args, parser

def get_video_path(video_path, flavor, output_dir):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir  = os.path.join(output_dir, video_name)
    video_output_dir += '_%s' % flavor
    if not os.path.exists(video_output_dir):
        os.makedirs(video_output_dir)
    return video_output_dir

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    started = datetime.now()
    args, parser = parse_args(argv)
    output_dir = os.path.dirname(args.output_video)

    print("Start time:  {}".format(started))
    print("Input video: {}".format(args.input_video))

    stargan_dir = get_video_path(args.input_video, 'stargan', output_dir)
    smoothed_dir = get_video_path(args.input_video, 'stargan_smoothed', output_dir)

    if not args.disable_full_run:
        # 1. Split video frames
        splt = 'Skipping ' if args.skip_split_to_frames else ''
        print("%sSplitting to frames..." % (splt))
        frames_dir = get_video_path(args.input_video, 'original', output_dir)
        frames = split_to_frames(args.input_video, args.skip_split_to_frames,
                                 output_dir=frames_dir)
        input_frames = frames
        print("Elapsed:  {}".format(datetime.now() - started))

        # 2. Run face_detect on each frames
        #    Input: original frame 1920x1080
        #    Output: cropped face, resized to 178x218
        #            w - width of face bounding box
        #
        print("Running face detection on frames...")
        cropped_dir = get_video_path(args.input_video, 'cropped', output_dir)
        w_list = np.empty(len(frames))
        for i, frame in enumerate(frames):
            output_frame = os.path.join(cropped_dir, os.path.basename(frame))
            w = face_detect._main(frame, output_frame)
            if not w:
                raise RuntimeError('failed to run face detection.')
            w_list[i] = w

        # 2.5 Determine average face bounding box width
        w_avg = w_list.mean()
        print("Elapsed:  {}".format(datetime.now() - started))

        # 3. Apply StarGAN on resized frames (178x218)
        #    Input: frames 178x218
        #    Output: fake image 128x128
        print("Applying StarGAN detection on frames...")
        stargan_frames = run_stargan(cropped_dir, stargan_dir)
        print("Elapsed:  {}".format(datetime.now() - started))

        # 3.5. Apply gaussian filter on StarGAN frames
        #    Input: fake images 128x128
        #    Output: smooth fake images 128x128
        print("Smoothing StarGAN frames...")
        gaussian_in_time.main(["--in_frames",  os.path.join(stargan_dir, "fake_frames"),
                               "--out_frames", smoothed_dir])
        print("Elapsed:  {}".format(datetime.now() - started))

    stargan_smoothed_fake_files = os.listdir(smoothed_dir)
    stargan_smoothed_frames = [os.path.join(smoothed_dir, f) for f in stargan_smoothed_fake_files]

    # 4. Resize input frames to smaller size
    #    Input: original frame 1920x1080
    #    Output: resized frame, e.g. 505x284
    #    Motivation: face_swap works better with resized frames
    #                in which the size of the face in the image is
    #                (now) proportional to StarGAN output images.

    # 4.5. face swap preprocessing step
    #    Approach #1: determine average ratio, resize all original frames
    #                 (before running face swap) to average ratio.
    #    Approach #2: before generating output video resize all frames
    #                 to minimum frame size.

    # Implementing approach #1 - resize all original frames
    #orig_width = input_frames[0].shape[0]
    #assert orig_width == FRAME_WIDTH
    orig_width = FRAME_WIDTH

    resized_dir = get_video_path(args.input_video, 'resized', output_dir)
    if not args.disable_full_run:
        resized_frames = []
        print("Resizing frames...")
        min_w_frame, min_h_frame = FRAME_WIDTH, FRAME_HEIGHT
        for w_cropped, frame in zip(w_list, input_frames):
            new_w_frame = int((orig_width / w_cropped) * STARGAN_OUTPUT_SIZE)
            new_h_frame = int(new_w_frame * float(FRAME_HEIGHT) / FRAME_WIDTH)
            frame_name = os.path.basename(frame)
            output_path = os.path.join(resized_dir, frame_name)
            frame_ = cv2.imread(frame)
            resized_frame = cv2.resize(frame_, (new_w_frame, new_h_frame))
            cv2.imwrite(output_path, resized_frame)
            resized_frames.append(output_path)
            min_w_frame = min(min_w_frame, new_w_frame)
            min_h_frame = min(min_h_frame, new_h_frame)
        print("Elapsed:  {}".format(datetime.now() - started))
    else:
        resized_frames = [os.path.join(resized_dir, f) for f in os.listdir(resized_dir)]
        min_w_frame, min_h_frame = FRAME_WIDTH, FRAME_HEIGHT
        for rf in resized_frames:
            frame = cv2.imread(rf)
            h, w, _ = frame.shape
            min_w_frame = min(min_w_frame, w)
            min_h_frame = min(min_h_frame, h)

    # 5. Apply face swap on StarGAN frames + resized frames
    #    Input: 2 frames: StarGAN frame, resized frame (from 4)
    #    Output: blended (swapped) frame
    face_swap_dir = get_video_path(args.input_video, 'face_swap', output_dir)
    if not args.disable_full_run:
        print("Applying face swap...")
        face_swapped_frames = []
        for stargan_frame, resized_frame in zip(stargan_smoothed_frames, resized_frames):
            frame_name = os.path.basename(resized_frame)
            out_path = os.path.join(face_swap_dir, frame_name)
            face_swap.main([stargan_frame, resized_frame, out_path])
            face_swapped_frames.append(out_path)
        print("Elapsed:  {}".format(datetime.now() - started))
    else:
        face_swapped_frames = [os.path.join(face_swap_dir, f) for f in os.listdir(face_swap_dir)]
        face_swapped_frames = [f for f in face_swapped_frames if '_render' not in f]

    # 5.5 Resize all frames to minimal w x h
    face_swap_resized_dir = get_video_path(args.input_video, 'face_swap_resized', output_dir)
    print("Resizing to %s x %s..." % (min_w_frame, min_h_frame))
    face_swap_resized = []
    for frame in face_swapped_frames:
        out_path = os.path.join(face_swap_resized_dir, os.path.basename(frame))
        image = cv2.imread(frame)
        resized_frame = cv2.resize(image, (min_w_frame, min_h_frame))
        cv2.imwrite(out_path, resized_frame)
        face_swap_resized.append(out_path)

    print("Elapsed:  {}".format(datetime.now() - started))
    # 6. Generate output video
    #    Input: face swapped frames
    #    Output: video
    print("Generating output video...")
    write_video(args.output_video, face_swap_resized)

    print("Output video: {}".format(args.output_video))
    print("End time:  {}".format(datetime.now()))
    print("Total time:  {}".format(datetime.now() - started))

    return 0

if __name__ == "__main__":
    sys.exit(main())
