import os
import sys

import skvideo.io

from scipy.misc import imread


def write_video(path, frames):
    video = skvideo.io.FFmpegWriter(path,
                                    outputdict={'-vcodec': 'libx264', '-b': '300000000'})
    for frame in frames:
        video.writeFrame(imread(frame))

    video.close()


def split_to_frames(video, skip_write_to_disk=False, output_dir=None):
    """Splits `video` into frames and return them.

    Arguments:
        video - path to video file
        output_dir - where the output frames will be stored.

    Returns:
        a list of file names on the disk.
    """
    video_base = os.path.basename(video)
    video_name = os.path.splitext(video_base)[0]

    if not output_dir:
        output_dir = os.path.join(os.getcwd(), video_name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    prefix = os.path.join(output_dir, video_name)
    videogen = skvideo.io.vreader(video)

    output_dir = os.path.join(output_dir, 'original_frames')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frames = []
    for i, frame in enumerate(videogen):
        frame_path = os.path.join(output_dir, '{:03}.png'.format(i+1))
        frames.append(frame_path)
        if not skip_write_to_disk:
            skvideo.io.vwrite(frame_path, frame)

    return frames


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    if len(argv) != 2:
        print("usage: img2video.py <video path> <output dir>")
        return -1

    input_video = argv[0]
    output_dir = argv[1]
    split_to_frames(input_video, output_dir=output_dir)


if __name__ == "__main__":
    sys.exit(main())
