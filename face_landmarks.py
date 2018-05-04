import os
import sys
import subprocess

import numpy as np

from datetime import datetime

from utils import get_frame_paths

mswindows = (sys.platform == "win32")

NUM_LANDMARKS = 68

if mswindows:
    CelebA_DIR = r'C:\face_swap\StarGAN\data\CelebA_nocrop\images\128x128'
else:
    CelebA_DIR = r'~/StarGAN/data/CelebA_nocrop/images/128x128'


def read_landmarks(path):
    """Reads landmark from path into a numpy array.

    Arguments:
        path - path to the landmarks csv file.

    Returns:
        a NUM_LANDMARKSx2 numpy array with pixel location for each landmark.

    """
    with open(path, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()][1:]

    landmarks = np.empty((NUM_LANDMARKS,2))
    for i, l in enumerate(lines):
        x_y = [int(elem) for elem in l.split(',')]
        landmarks[i,:] = x_y

    return landmarks


def find_landmarks(frame, find_face_landmarks_dir):
    """Determine landmarks for the given frame.

    Arguments:
        frame - path to the input frame.
        find_face_landmarks_dir - root directory of find_face_landmarks Git repository

    Returns:
        a NUM_LANDMARKSx2 numpy array with pixel location for each landmark.
    """
    env = os.environ.copy()
    cwd = os.path.join(find_face_landmarks_dir, 'build', 'install', 'bin')
    face_landmarks = os.path.join(cwd, 'find_face_landmarks.exe')
    landmarks_path = r'C:\face_swap\face_swap\build\install\data\shape_predictor_68_face_landmarks.dat'
    output_csv = os.path.join(cwd, 'landmarks.csv')
    cmd = [face_landmarks,
           '-i', frame,
           '-l', landmarks_path,
           '-o', output_csv]
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ret = proc.wait()
    out, err = proc.communicate()
    if ret:
        raise RuntimeError("Failed to run find_face_landmarks.exe: " + out.decode('utf-8'))

    return read_landmarks(output_csv)


def find_average_face(dir_, num_samples=0):
    """Determine average landmarks location within a set of frames.

    Arguments:
        dir_ - path to directory containing the frames
        num_samples - number of samples to use from dir_. Use 0 for all.

    Returns:
        a tuple of: (num_images,
                     NUM_LANDMARKSx2 numpy array with average locations for each landmark.)

    """
    files = get_frame_paths(dir_)

    num_frames = len(files)
    assert num_frames > 0, "no frames in directory!"
    #landmarks = np.empty((num_frames, NUM_LANDMARKS, 2))
    landmarks = []

    if num_samples > 0:
        files = np.random.permutation(files)[:num_samples]

    failed_frames = []
    print("Processing {} frames".format(num_frames))
    for i, f in enumerate(files):
        print("[{}/{}] {}".format(i + 1, num_frames, f))
        try:
            landmarks.append(find_landmarks(f))
        except RuntimeError as e:
            failed_frames.append(f)

    print("number of failed frames:", len(failed_frames))
    for f in failed_frames: print(f)

    assert len(landmarks) > 0, "no landmarks found!"
    if not len(landmarks):
        return 0, None

    landmarks = np.array(landmarks)

    """
    # test
    x, y = np.zeros(1), np.zeros(1)
    for l in landmarks:
        x += l[0,0]
        y += l[0,1]

    x /= len(landmarks)
    y /= len(landmarks)
    """

    mean_landmarks = np.mean(landmarks, axis=0)
    mean_landmarks = np.clip(mean_landmarks, 0, 255).astype(np.uint8)
    return len(landmarks), mean_landmarks

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    max_num_frames = 1000
    if len(argv) == 1:
        max_num_frames = int(argv[0])
    
    print("Num of samples:", max_num_frames)
    print("Input dir:", CelebA_DIR)
    
    output_path = 'mean_CelebA_%d_samples_face.csv'

    ret = 0
    msg = ''
    start = datetime.now()
    actual_num_frames = 0
    csv_output_path = 'mean_CelebA_{ns}_samples_%s_actual_{date}_{hour}_{minute}_{sec}.csv'.format(
        ns=max_num_frames, date=str(start.date()), hour=start.hour, minute=start.minute, sec=start.second, usec=start.microsecond)

    try:
        actual_num_frames, mean_landmarks = find_average_face(CelebA_DIR, max_num_frames)
        csv_output_path = csv_output_path % actual_num_frames
        print("Dumping results to:", csv_output_path)
        with open(csv_output_path, 'w') as f:
            f.write('x,y\n')
            for l in mean_landmarks:
                f.write('%s,%s\n'% (l[0], l[1]))
    except Exception as e:
        msg = "%s" % e
        ret = -1

    end = datetime.now()
    
    if ret:
        print("ERROR: %s" % msg)
    else:
        print("Success.")

    total_time = end - start
    print("Average execution time per frame: %s" % (total_time / actual_num_frames))
    print("Total time: %s over %d images" % (total_time, actual_num_frames))

    return ret


if __name__ == '__main__':
    sys.exit(main())
