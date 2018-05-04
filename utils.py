import os
import sys
import subprocess

import cv2

mswindows = (sys.platform == "win32")

if mswindows:
    FACE_SWAP_DEFAULT_PATH = r'C:\face_swap\face_swap'
else:
    FACE_SWAP_DEFAULT_PATH = '~/face_swap'

def MSE(env, W, src, tgt):
    cwd = os.path.join(FACE_SWAP_DEFAULT_PATH, 'build', 'install', 'bin')
    apply_w = os.path.join(cwd, 'apply_w')
    apply_w = apply_w + '.exe' if mswindows else apply_w
    cmd = [apply_w, W, src, tgt]
    subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE)
    out, err = proc.communicate()
    ret = proc.wait()
    if ret:
        raise RuntimeError("Failed to run apply_w: " + out)

    return float(out.splitlines()[-1].lstrip(b'Score Avg :'))

def get_frame_paths(dir_):
    exts = ('.png', '.jpg')
    files = [os.path.join(dir_, f) for f in os.listdir(dir_)]
    files = [f for f in files if os.path.splitext(f)[1].lower() in exts]
    return files

def apply_alignment_impl(A, img):
    """Applies the alignment `A` on input frame.

    The new frame is returned.

    Args:
        A - a 2x3 alignment matrix (numpy array).
        img - cXmXn numpy array

    Returns:
        Output frame
    """
    rows, cols, ch = img.shape
    dst = cv2.warpAffine(img, A, (cols,rows))

    return dst


def apply_alignment(A, frame, out_path=None):
    """Applies the alignment `A` on input frame.

    The new frame is written to out_path (see below).

    Args:
        A - a 2x3 alignment matrix (numpy array).
        frame - path of the input frame
        out_path - Optional, path to the output frame.
                   if not specified, we create an `aligned` folder
                   and add _aligned suffix to every frame processed.

    Returns:
        path to output frame
    """
    img = cv2.imread(frame)
    dst = apply_alignment_impl(A, img)

    if not out_path:
        base, dir_ = os.path.basename(frame), os.path.dirname(frame)
        name, ext = os.path.splitext(base)
        new_name = '%s_aligned%s' % (name, ext)
        new_dir = os.path.join(dir_, 'aligned')
        try:
            os.makedirs(new_dir)
        except Exception as e:
            if not os.path.exists(new_dir):
                raise
        out_path = os.path.join(new_dir, new_name)

    cv2.imwrite(out_path, dst)
    return out_path


def dump_alignment_matrix(A, path):
    A = A[:-1]
    dir_ = os.path.dirname(path)
    if not os.path.exists(dir_):
        os.makedirs(dir_)
    with open(path, 'w') as f:
        txt = []
        for i in range(2):
            txt.append(','.join([str(e) for e in A[i]]))
        f.write('\n'.join(txt))