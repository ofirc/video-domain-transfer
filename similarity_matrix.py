import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy.misc import imread

def get_alignment(src_points, tgt_points):
    """Computes similarity matrix between src_points and tgt_points.
    
    Returns:
        a 2x3 numpy array, similarity matrix.
    """
    N = len(src_points)

    # We would like to solve: Ax=b
    ''' Build A and b '''
    A = np.empty((2*N, 4))
    b = np.empty((2*N, 1))
    i = 0
    for x, y in src_points:
        A[i,:]  = np.array([x, -y, 1, 0])
        A[i+1,:]= np.array([y, x, 0, 1])
        i += 2

    i = 0
    for x, y in tgt_points:
        b[i,:]  = np.array([x])
        b[i+1,:]  = np.array([y])
        i += 2

    ''' Inverse A and solve for x '''
    A_inv = np.linalg.pinv(A)
    x = np.matmul(A_inv, b)

    ''' Create Similarity Matrix from x '''
    '''         | a_1   -a_2   a_3 | '''
    ''' s_mat = | a_2    a_1   a_4 | '''
    '''         |  0      0     1  | '''
    #print('----------------')
    a_1 = x[0][0]
    a_2 = x[1][0]
    a_3 = x[2][0]
    a_4 = x[3][0]
    s_mat = np.array([[a_1, -a_2, a_3], [a_2, a_1, a_4], [0, 0, 1]])
    return s_mat


def test_alignment():
    """Tests our alignment matrix with 90 deg counter-clockwise rotation."""
    lena = imread('lena.png')

    #
    # Obtain mapping.
    #

    # Square
    src_points_x = [100, 100, 400, 400, 450]
    src_points_y = [100, 400, 100, 400, 450]
    tgt_points_x = [150, 150, 350, 350, 300]
    tgt_points_y = [150, 350, 150, 350, 300]

    src_points = zip(src_points_x, src_points_y)
    tgt_points = zip(tgt_points_x, tgt_points_y)
    s_mat = get_alignment(list(src_points), list(tgt_points))

    #
    # Apply on a test point
    #
    x, y = 400, 100
    src_point = np.array([x, y, 1])
    tgt_point = s_mat.dot(src_point)
    tgt_point_x = tgt_point[0]
    tgt_point_y = tgt_point[1]
    print('**** tgt_point ****')
    print(tgt_point)

    implot = plt.imshow(lena)
    plt.scatter(src_points_x, src_points_y, c='r', s=20)
    plt.scatter(tgt_points_x, tgt_points_y, c='b', s=20)
    plt.scatter(tgt_point_x, tgt_point_y, c='g', s=20)
    plt.show()

def test_rotate():
    # Obtain mapping.
    src_points = np.array([[100, 100], [400, 100], [400, 400], [100, 400]])
    dst_points = np.roll(src_points, 1, axis=0)
    s_mat = get_alignment(src_points, dst_points)
    M = s_mat[:-1]

    # Apply mapping.
    #lena = imread('lena.png')
    #implot = plt.imshow(lena)
    #plt.show()
    img = cv2.imread('lena.png')
    rows,cols,ch = img.shape
    #M = cv2.getAffineTransform(pts1,pts2)

    dst = cv2.warpAffine(img,M,(cols,rows))
    plt.subplot(121),plt.imshow(img),plt.title('Input')
    plt.subplot(122),plt.imshow(dst),plt.title('Output')
    plt.show()    

def test_identity():
    # Obtain mapping.
    src_points = np.array([[100, 100], [400, 100], [400, 400], [100, 400]])
    dst_points = src_points
    s_mat = get_alignment(src_points, dst_points)
    M = s_mat[:-1]

    # Apply mapping.
    #lena = imread('lena.png')
    #implot = plt.imshow(lena)
    #plt.show()
    img = cv2.imread('lena.png')
    rows,cols,ch = img.shape
    #M = cv2.getAffineTransform(pts1,pts2)

    dst = cv2.warpAffine(img,M,(cols,rows))
    plt.subplot(121),plt.imshow(img),plt.title('Input')
    plt.subplot(122),plt.imshow(dst),plt.title('Output')
    plt.show()    


def main():
    #test_alignment()
    #test_rotate()
    test_identity()


if __name__ == '__main__':
    sys.exit(main())