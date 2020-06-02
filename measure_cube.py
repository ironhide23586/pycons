"""  _
    |_|_
   _  | |
 _|_|_|_|_
|_|_|_|_|_|_
  |_|_|_|_|_|
    | | |_|
    |_|_
      |_|

Author: Souham Biswas
Website: https://www.linkedin.com/in/souham/
"""

from glob import glob
import os
import itertools

import numpy as np
import cv2


ROOT_DIR = 'data'
IMAGE_DIR = ROOT_DIR + os.sep + 'RGB'
MASKS_DIR = ROOT_DIR + os.sep + 'Segmentation'
POSE_DIR = ROOT_DIR + os.sep + 'Pose'


class Camera:

    def __init__(self, cam_xyz, quaternion, im_w, im_h, hor_fov_deg=90, focal_len_meters=1.):
        self.cam_xyz = cam_xyz
        self.quaternion = quaternion
        self.im_w = im_w
        self.im_h = im_h
        self.K = np.eye(3)
        self.f = focal_len_meters
        self.hor_fov_deg = hor_fov_deg
        self.frame_w_meters = 2 * self.f * np.tan(np.deg2rad(self.hor_fov_deg / 2.))
        self.frame_h_meters = (self.im_h / self.im_w) * self.frame_w_meters
        self.ssx = self.frame_w_meters / self.im_w
        self.ssy = self.frame_h_meters / self.im_h
        self.K[0, 0] = 1
        self.K[1, 1] = -1
        self.K[0, 2] = (self.im_w / 2.) * self.ssx
        self.K[1, 2] = (self.im_h / 2.) * self.ssy
        a, b, c, d = self.quaternion
        self.R = np.array([[2 * a ** 2 - 1 + 2 * b ** 2, 2 * b * c + 2 * a * d, 2 * b * d - 2 * a * c],
                           [2 * b * c - 2 * a * d, 2 * a ** 2 - 1 + 2 * c ** 2, 2 * c * d + 2 * a * b],
                           [2 * b * d + 2 * a * c, 2 * c * d - 2 * a * b, 2 * a ** 2 - 1 + 2 * d ** 2]])
        x, y, z = self.cam_xyz
        self.T = np.array([-x, -y, -z])
        self.RT = np.eye(4)
        self.RT[:3, :3] = self.R
        self.RT[:3, 3] = self.T

    def project_xyzs_to_imcoords(self, world_xyzs_in):
        world_xyzs = np.ones([world_xyzs_in.shape[0], 4])
        world_xyzs[:, :3] = world_xyzs_in
        cam_xyzs = np.dot(self.RT, world_xyzs.T).T
        # cam_xyzs = cam_xyzs[np.logical_and(cam_xyzs[:, 1] > 0, cam_xyzs[:, 2] > 0)]
        cam_xyzs = cam_xyzs[cam_xyzs[:, 1] > 0]
        num_points = cam_xyzs.shape[0]
        im_xys = np.ones([num_points, 3])
        im_xys[:, :2] = self.f * cam_xyzs[:, [0, 2]] / cam_xyzs[:, [1, 1]]
        im_uvs = (np.dot(self.K, im_xys.T).T[:, :2] / [self.ssx, self.ssy]).astype(np.int)
        return im_uvs

    def viz_points(self, world_xyzs_in):
        xys = self.project_xyzs_to_imcoords(world_xyzs_in)
        xs, ys = xys.T
        # org_x, org_y = self.project_xyzs_to_imcoords(np.array([[0, 0, 0]]))
        im = np.zeros([self.im_h, self.im_w]).astype(np.uint8)
        f = np.logical_and(np.logical_and(xs > 0, xs < self.im_w), np.logical_and(ys > 0, ys < self.im_h))
        im[ys[f], xs[f]] = 255
        return im

    def viz_plane(self, side=100, plane_idx=0, plane_loc=0):
        xs = np.arange(-side, side)
        ys = np.arange(-side, side)
        zs = np.arange(-side, side)
        if plane_idx == 0:
            xs = np.array([plane_loc])
        elif plane_idx == 1:
            ys = np.array([plane_loc])
        elif plane_idx == 2:
            zs = np.array([plane_loc])
        xyzs = np.array(list(itertools.product(xs, ys, zs)))
        plane_viz = self.viz_points(xyzs)
        return plane_viz


class View:

    def __init__(self, im_fpath, mask_fpath, pose_fpath):
        self.im_fpath = im_fpath
        self.mask_fpath = mask_fpath
        self.pose_fpath = pose_fpath
        self.im = None
        self.mask = None
        self.pose = None
        self.m_plane_bgr = [241, 104, 146]
        self.m_cube_bgr = [248, 71, 170]
        self.m_sky_bgr = [73, 226, 210]
        self.cam = None
        self.im_h = 0
        self.im_w = 0
        # self.init()

    def read_pose_loc(self, d):
        line2num = lambda i: float(d[i].replace('}', '').strip().split(':')[-1].split(',')[0].strip())
        q = np.array([line2num(i) for i in range(4)])
        xyz = np.array([line2num(i) for i in range(4, 7)])
        return q, xyz

    def init(self, loc_xyz=None, quaternion=None):
        self.im = cv2.imread(self.im_fpath)
        self.im_h, self.im_w, _ = self.im.shape
        self.mask = cv2.imread(self.mask_fpath)
        with open(self.pose_fpath) as f:
            data = f.readlines()
            self.quaternion, self.loc_xyz = self.read_pose_loc(data)
        if loc_xyz is not None:
            self.loc_xyz = np.array(loc_xyz)
        if quaternion is not None:
            self.quaternion = np.array(quaternion)
        self.cam = Camera(self.loc_xyz, self.quaternion, self.im_w, self.im_h)


if __name__ == '__main__':
    # im_fpaths = glob(IMAGE_DIR + os.sep + '*')
    # ids = [n.split(os.sep)[-1].split('_')[0] for n in im_fpaths]
    # id2fpath = lambda dir, id, suffix: dir + os.sep + id + '_' + suffix
    # mask_fpaths = [id2fpath(MASKS_DIR, id, '1.png') for id in ids]
    # pose_fpaths = [id2fpath(POSE_DIR, id, '2.txt') for id in ids]
    # num_images = len(mask_fpaths)
    #
    # views = [View(im_fpaths[i], mask_fpaths[i], pose_fpaths[i]) for i in range(num_images)]
    # views[0].loc_xyz = [100, 100, 0]
    # views[0].init([0, 10, 0], [.003, .423, .855, .3])

    cam = Camera([0, 10, 10], [1, 0, 0, 0], 200, 200)
    # cam.project_xyzs_to_imcoords(np.array([[0, 10, -20]]))
    p = cam.viz_plane(50, 1, 30)
    cv2.imwrite('p.png', p)
    k = 0
