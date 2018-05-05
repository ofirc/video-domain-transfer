"""
python write_inverse_fake_video.py \
  stargan_results\orig_frames\stargan_results\fake_frames \
  stargan_results\orig_frames\aligned \
  stargan_results\orig_frames\stargan_results\stargan_results_inverse_fake.mp4
"""
import os
import sys

import skvideo.io
import numpy as np

from utils import apply_alignment
from scipy.misc import imread

def read_images(dir_):
    images = [os.path.join(dir_, f) for f in os.listdir(dir_)]
    return images

def read_inverse_alignments(dir_):
    files = [f for f in os.listdir(dir_) if f.endswith('minus_one.csv')]
    ret = np.empty((len(files), 2, 3))
    for num_alignment, fname in enumerate(files):
        with open(os.path.join(dir_, fname), 'r') as f:
            lines = [l.strip() for l in f.readlines()]
            for i, line in enumerate(lines):
                for j, flt in enumerate(line.split(',')):
                    ret[num_alignment, i, j] = np.float(flt)
    return ret

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    if len(argv) != 3:
        print("Usage: %s <input images dir> <a_minus_one_dir> <output video>" % sys.argv[0])
        return 1

    input_images_dir = argv[0]
    inverse_alignments_dir = argv[1]
    output_video_path = argv[2]

    print("Writing video to:", output_video_path)
    orig_vid = skvideo.io.FFmpegWriter(output_video_path,
                                       outputdict={'-vcodec': 'libx264', '-b': '300000000'})

    images = read_images(input_images_dir)
    inv_alignments = read_inverse_alignments(inverse_alignments_dir)
    assert len(images) == len(inv_alignments)
    temp_out = r'c:\face_swap\automation\aligned_image.png'
    for alignment, image in zip(inv_alignments, images):
        apply_alignment(alignment, image, temp_out)
        orig_vid.writeFrame(imread(temp_out))

    orig_vid.close()
    print("Success.")

if __name__ == '__main__':
    exit(main())
