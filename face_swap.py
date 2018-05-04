import os
import shutil
import subprocess
import sys

import cv2

import utils

class FaceSwap:
    PATH = r"""C:\qt_\5.6.0\msvc2015_64\bin;C:\face_swap\face_swap\build\install\bin;c:\face_swap\glew\bin\Release\x64;C:\Users\ocohen11\.caffe\dependencies\libraries_v140_x64_py27_1.1.0\libraries\bin;C:\Program Files (x86)\common files\intel\Shared Libraries\redist\intel64_win\compiler;C:\Users\ocohen11\.caffe\dependencies\libraries_v140_x64_py27_1.1.0\libraries\x64\vc14\bin;C:\Users\ocohen11\.caffe\dependencies\libraries_v140_x64_py27_1.1.0\libraries\lib;C:\face_swap\find_face_landmarks\build\install\bin"""
    QT_WIDGETS = r"C:\QT_\5.6.0\msvc2015_64"
    QT_QPA_PLATFORM_PLUGIN_PATH = r"C:\QT_\5.6.0\msvc2015_64\plugins\platforms"
    CFG_TEMPLATE = """landmarks = ../data/shape_predictor_68_face_landmarks.dat       # path to landmarks model file
model_3dmm_h5 = ../data/BaselFaceModel_mod_wForehead_noEars.h5  # path to 3DMM file (.h5)
model_3dmm_dat = ../data/BaselFace.dat                          # path to 3DMM file (.dat)
reg_model = ../data/3dmm_cnn_resnet_101.caffemodel              # path to 3DMM regression CNN model file (.caffemodel)
reg_deploy = ../data/3dmm_cnn_resnet_101_deploy.prototxt        # path to 3DMM regression CNN deploy file (.prototxt)
reg_mean = ../data/3dmm_cnn_resnet_101_mean.binaryproto         # path to 3DMM regression CNN mean file (.binaryproto)
seg_model = ../data/face_seg_fcn8s.caffemodel                   # path to face segmentation CNN model file (.caffemodel)
seg_deploy = ../data/face_seg_fcn8s_deploy.prototxt             # path to face segmentation CNN deploy file (.prototxt)
generic = 0                                 # use generic mode (disable shape regression)
expressions = 1                             # use expression regression
gpu = 0                                     # toggle GPU / CPU
gpu_id = 0                                  # GPU's device id
verbose = 1                                 # 1 = before blend image, 2 += projected meshes, 3 += landmarks, 4 += meshes ply
input = {src}       # source image
input = {tgt}       # target image
output = {out}                            # output image or directory"""

    def __init__(self):
        """Initializes a face swap instance."""
        self._prepare_env()

    def _prepare_env(self):
        env = os.environ.copy()
        env['PATH'] = '{};{}'.format(self.PATH, env['PATH'])
        env['Qt5Widgets_DIR'] = self.QT_WIDGETS
        env['QT_QPA_PLATFORM_PLUGIN_PATH'] = self.QT_QPA_PLATFORM_PLUGIN_PATH
        self.env = env

    def _dump_env(self):
        envs = ('PATH', 'Qt5Widgets_DIR', 'QT_QPA_PLATFORM_PLUGIN_PATH')
        for env in envs:
            print('set {}={}'.format(env, self.env[env]))
    
    def _set_cfg(self, src, tgt, out):
        """Prepares an input configuration file for face_swap_image.
        
        Arguments:
            src - source frame
            tgt - target frame
            out - output image (blended frame)

        Returns the path to the configuration file.
        """
        CFG_NAME = 'test.cfg'
        with open(CFG_NAME, 'w') as f:
            f.write(self.CFG_TEMPLATE.format(src=src.replace('\\', '/'), tgt=tgt.replace('\\', '/'), out=out))
        
        return os.path.abspath(CFG_NAME)

    @staticmethod
    def _get_absolute_mappings_path(src, tgt):
        dir_ = os.path.dirname(src)
        src_fname = os.path.splitext(os.path.basename(src))[0]
        dst_fname = os.path.splitext(os.path.basename(tgt))[0]
        return os.path.join(dir_, 'mappings_%s_%s.txt' % (src_fname, dst_fname))

    def run(self, src, tgt, out='out.jpg'):
        """Runs face swap on source and target input frames.

        We also backup the W (absolute_mappings.txt) in order to
        evaluate the MSE diff, i.e. W(src) - tgt
        as well as reuse it later for applying to other images.

        Arguments:
            src - source frame
            tgt - target frame
            out - output image (blended frame)
        """
        assert hasattr(self, 'env')
        cwd = r'C:\face_swap\face_swap\build\install\bin'
        cfg = self._set_cfg(src, tgt, out)
        exe = os.path.join(cwd, 'face_swap_image.exe')
        cmd = [exe, '--cfg', cfg]
        shutil.copy(cfg, cwd)
        subprocess.check_call(cmd, cwd=cwd, env=self.env)

        """
        # backup the mappings
        abs_orig = os.path.join(cwd, 'absolute_mappings.txt')
        abs_backup_path = FaceSwap._get_absolute_mappings_path(src, tgt)
        
        print("Coping from : " + abs_orig)
        shutil.copy(abs_orig, abs_backup_path)
        self._W = abs_backup_path

        # determine the MSE
        # apply_w = os.path.join(cwd, 'apply_w.exe')
        # cmd = [apply_w, self._W, src, tgt]
        # proc = subprocess.Popen(cmd, cwd=cwd, env=self.env, stdout=subprocess.PIPE)
        # out, err = proc.communicate()
        # ret = proc.wait()
        # if ret:
            # raise RuntimeError("Failed to run apply_w.exe: " + out)

        # self._MSE = float(out.splitlines()[-1].lstrip('Score Avg :'))
        self._MSE = utils.MSE(self.env, self._W, src, tgt)
        
        print("MSE : " + str(self._MSE))"""

    def W(self):
        assert hasattr(self, '_W'), "Did you forget to initialize the W?"
        return self._W

    def MSE(self):
        assert hasattr(self, '_MSE'), "Did you forget to initialize the MSE?"
        return self._MSE

    def face_seg(self):
        """Returns a segmentation of the src image.
        
        Returns a numpy array of the src image with 0 for background
        pixels and 255 for face pixels.
        """
        seg_path = r'c:\face_swap\temp\out.jpg'
        return cv2.imread(seg_path)

def main(argv=None):
    src = r'C:\face_swap\automation\VID_20180131_200933\VID_20180131_200933_000.png'
    tgt = r'C:\face_swap\automation\VID_20180131_200933\VID_20180131_200933_001.png'
    if not argv:
        print("Usage: %s <src> <tgt> [<merged>]" % __file__)
        return -1

    src = argv[0]
    tgt = argv[1]
    merged = argv[2] if len(argv) == 3 else 'out.png'
    fs = FaceSwap()
    #fs._dump_env()
    fs.run(src, tgt, merged)
    return 0

if __name__ == '__main__':
    main(sys.argv)