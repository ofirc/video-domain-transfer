import sys
import subprocess
import os
import shutil

# StarGAN filters.
YOUNG=1
OLD=2

mswindows = (sys.platform == "win32")

if mswindows:
    STARGAN_DEFAULT_DIR=r"C:\face_swap\StarGAN"
else:
    STARGAN_DEFAULT_DIR="~/StarGAN"

def run_stargan(input_dir, output_dir, dont_call_stargan_cmd=False,
                filter_=YOUNG, crop_size="178", stargan_dir=STARGAN_DEFAULT_DIR):
    """Applies StarGAN on `frames` and returns modified frames.

    Args:
        input_dir - a directory containing frames as input for StarGAN
        output_dir - a directory for StarGAN results
        filter - the selected StarGAN filter (currently not in use
                 and hard coded in StarGAN implementation).
        crop_size - the size for (center)-cropping from the input images.
        stargan_dir - the root directory of StarGAN Git repository.

    Returns:
        a list of frames after applying StarGAN with filter `filter_`.
    """

    # Note: Crop size may differ from clip to clip
    cmd = ["python", "main.py",
           "--mode", "test",
           "--dataset", "VidTIMIT",
           "--c_dim", "5",
           "--image_size", "128",
           "--test_model", "20_3000",
           "--model_save_path", "stargan_celebA/models",
           "--vidtimit_image_path", input_dir,
           "--result_path", output_dir,
           "--vidtimit_crop_size", crop_size]

    #print("Running: %s" % " ".join(cmd))

    if not dont_call_stargan_cmd:
        subprocess.check_call(cmd, cwd=r"C:\face_swap\StarGAN")

    result = []
    for prefix in ('fake',):
        frames_dir = os.path.join(output_dir, '%s_frames' % prefix)
        files = [f for f in os.listdir(frames_dir) if f.endswith('.png')]
        files = [os.path.join(frames_dir, f) for f in files]
        assert len(files), "StarGAN didn't generate any %s frames." % prefix
        result.append(files)

    return result[0]


def copy_frames(frames, suffix):
    for frame in frames:
        dir_ = os.path.dirname(frame)
        name = os.path.basename(frame)
        
        new_dir = dir_ + '_' + suffix
        dst = os.path.join(new_dir, name)
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)

        shutil.copy(frame, new_dir)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    if len(argv) != 2:
        print("usage: stargan.py <input dir> <output dir>")
        return -1

    input_dir = argv[0]
    output_dir = argv[1]

    frames = run_stargan(input_dir, output_dir)

    for frame in frames:
        print(frame)


if __name__ == "__main__":
    sys.exit(main())
