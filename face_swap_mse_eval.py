"""Evaluate face swap pixel alignment on video (with MSE).

e.g. python eval_face_swap.py --video VID_20180131_200933.mp4
     python eval_face_swap.py --frames <frames dir>

Pipeline:
1) Take a video break it into frames (or use --frames frames)
   (generate I1, ... , IN)

2) Apply StarGAN on each of the input frame
   (and yield J1, ... , JN)

3) Run face_swap

a) Run face_swap on every consecutive pair of frames
   determine warping W for them
   calculate MSE: W(I1) - W2

b) Apply W on on every consecutive pair of (StarGAN) frames
   calculate MSE: W(J1) - J2

c) Summarize results
                Test1, Test2, ... , TestN, Average
W(I1)<--> I2  
W(J1)<--> J2
V(J1)<--> J2
"""
import os
import sys
import shutil

import stargan
import utils

from argparse import ArgumentParser

from img2video import split_to_frames
from stargan import run_stargan
from face_swap import FaceSwap


def parse_args(argv):
    parser = ArgumentParser(description="Evaluates face swap pixel alignment on video.")
    parser.add_argument("--video",
                        help="Input video.")
    parser.add_argument("--frames",
                        help="Input frames dir.")
    parser.add_argument("--skip_split_to_frames",
                        required=False,
                        action='store_true',
                        help="Skip writing of splited frames to disk. Can be used if the frames were already generated")
    parser.add_argument("--dont_call_stargan",
                        required=False,
                        action='store_true',
                        help="Don't call StarGAN. Can be used if StarGAN results already exists")

    args = parser.parse_args(argv)

    if args.video and args.frames:
        parser.error('you cannot select both --frames and --video.')

    if not args.video and not args.frames:
        parser.error('you must select at least video or frames.')

    return args, parser


def copy_faceswap_result(source_file, faceswap_res_path, dst_file):
    # Backup faceswap results for debug
    debug = True
    if debug:
        file_path = os.path.join(faceswap_res_path, dst_file)
        shutil.copy(source_file, file_path)


def main(argv=None):
    for i in range(3):
        print("***********************************************************************************")

    if argv is None:
        argv = sys.argv[1:]

    args, parser = parse_args(argv)

    print("[*] Input video: {}".format(args.video))
    if args.video:
        frames = split_to_frames(args.video, args.skip_split_to_frames)
    else:
        frames = [os.path.join(args.frames, f) for f in os.listdir(args.frames)
                  if f.endswith('.png') and not 'original' in f]

    assert frames, "No frames were generated!"
    frames_dir = os.path.dirname(frames[0])
    results_dir = os.path.dirname(frames_dir)
    stargan_res_dir = os.path.join(results_dir, "stargan_results")
    orig_frames, fake_frames = run_stargan(frames_dir, stargan_res_dir, args.dont_call_stargan, stargan.OLD)
    
    # Make FaceSwap results dirs
    i_to_w_i = os.path.join(results_dir, "i_to_w_i")
    j_to_w_j = os.path.join(results_dir, "j_to_w_j")
    j_to_v_j = os.path.join(results_dir, "j_to_v_j")
    if not os.path.exists(i_to_w_i):
        os.makedirs(i_to_w_i)
    if not os.path.exists(j_to_w_j):
        os.makedirs(j_to_w_j)
    if not os.path.exists(j_to_v_j):
        os.makedirs(j_to_v_j)
    
    mse_results = []
    failed_orig_frames = []
    failed_fake_frames = []
    
    # run face_swap on every consecutive pair of orig_frames
    prev_frames = orig_frames[:-1]
    cur_frames  = orig_frames[1:]
    i = 0
    fs_orig    = FaceSwap()
    fs_stargan = FaceSwap()

    results_file = open(os.path.join(results_dir, "results.csv"), "w")
    #(prev_frame, cur_frame, fake_frames[i], fake_frames[i + 1], mse_orig, mse_stargan_W, mse_stargan_V)        
    results_file.write("Original frame current,Original frame next,Fake frame current,Fake frame next,MSE_orig,MSE_W_fake,MSE_V" + "\n")

    failed_orig_file = open(os.path.join(results_dir, "failed_orig.txt"), "w")
    failed_orig_file.write("************************** \n")
    failed_orig_file.write("*** Failed orig frames *** \n")
    failed_orig_file.write("************************** \n")

    failed_fake_file = open(os.path.join(results_dir, "failed_fake.txt"), "w")
    failed_fake_file.write("************************** \n")
    failed_fake_file.write("*** Failed fake frames *** \n")
    failed_fake_file.write("************************** \n")

    for prev_frame, cur_frame in zip(prev_frames, cur_frames):        
        try:
            fs_orig.run(prev_frame, cur_frame)
        except:
            failed_orig_frames.append(prev_frame)
            failed_orig_file.write(prev_frame + "\n")
            failed_orig_file.flush()
            i += 1
            continue
        copy_faceswap_result(r'C:\face_swap\face_swap\build\install\bin\w_i1.jpg', i_to_w_i, os.path.basename(prev_frame))
        
        mse_orig = fs_orig.MSE()
        W = fs_orig.W()

        try:
            fs_stargan.run(fake_frames[i], fake_frames[i + 1])
        except:
            failed_fake_frames.append(fake_frames[i])
            failed_fake_file.write(fake_frames[i] + "\n")
            failed_fake_file.flush()
            i += 1
            continue
        copy_faceswap_result(r'C:\face_swap\face_swap\build\install\bin\w_i1.jpg', j_to_v_j, os.path.basename(fake_frames[i]))
        
        mse_stargan_V = fs_stargan.MSE()

        mse_stargan_W = utils.MSE(fs_stargan.env, W, fake_frames[i], fake_frames[i + 1])       
        copy_faceswap_result(r'C:\face_swap\face_swap\build\install\bin\W_j1_new.png', j_to_w_j, os.path.basename(fake_frames[i]))
        
        mse_result = (prev_frame, cur_frame, fake_frames[i], fake_frames[i + 1], mse_orig, mse_stargan_W, mse_stargan_V)
        mse_results.append(mse_result)
        str_array = (str(m) for m in mse_result)
        results_file.write(",".join(str_array) + "\n")
        results_file.flush()

        i += 1

    results_file.close()
    failed_orig_file.close()
    failed_fake_file.close()


if __name__ == "__main__":
    sys.exit(main())


