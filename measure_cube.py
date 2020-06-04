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

    def __init__(self, cam_world_loc_enu, im_w, im_h, cam_quaternion=[1, 0, 0, 0], hor_fov_deg=90):
        self.R = None
        self.T = None
        self.RT = None
        self.cam_world_loc_xyz = None
        self.cam_world_loc_enu = None
        self.cam_quaternion = None
        self.update_cam_loc_pose(cam_world_loc_enu, cam_quaternion)
        self.im_w = im_w
        self.im_h = im_h
        self.hor_fov_deg = hor_fov_deg
        self.K = self.get_camera_calibration_matrix(self.im_w, self.im_h, self.hor_fov_deg)
        self.projection_matrix = np.dot(self.K, self.RT)

    def get_camera_calibration_matrix(self, im_w, im_h, hor_fov_deg):
        f = 1
        frame_w_meters = 2 * f * np.tan(np.deg2rad(hor_fov_deg / 2.))  # assuming unit focal length
        K = np.eye(3)
        K[0, 0] = im_w / frame_w_meters
        K[1, 1] = im_w / frame_w_meters
        K[0, 2] = (im_w / 2.)
        K[1, 2] = (im_h / 2.)
        return K

    def get_cam_rot_trans_matrix(self, cam_world_loc_xyz, cam_quaternion):
        a, b, c, d = cam_quaternion
        rotation_matrix = np.array([[2 * a ** 2 - 1 + 2 * b ** 2,       2 * b * c + 2 * a * d,        2 * b * d - 2 * a * c],
                                    [2 * b * c - 2 * a * d,       2 * a ** 2 - 1 + 2 * c ** 2,        2 * c * d + 2 * a * b],
                                    [2 * b * d + 2 * a * c,             2 * c * d - 2 * a * b,  2 * a ** 2 - 1 + 2 * d ** 2]])
        x, y, z = cam_world_loc_xyz
        translation_vector = np.array([-x, -y, -z])
        rotation_translation_matrix = np.eye(4)[:3]
        rotation_translation_matrix[:3, :3] = rotation_matrix
        rotation_translation_matrix[:3, 3] = translation_vector
        return rotation_translation_matrix, translation_vector, rotation_matrix
    
    def update_cam_loc_pose(self, cam_world_loc_enu, cam_quaternion):  # quaternion expected in xyz format (e, -u, n)
        self.cam_world_loc_enu = np.array(cam_world_loc_enu)
        e, n, u = self.cam_world_loc_enu
        self.cam_world_loc_xyz = np.array([e, -u, n])
        self.cam_quaternion = cam_quaternion
        self.RT, self.R, self.T = self.get_cam_rot_trans_matrix(self.cam_world_loc_xyz, self.cam_quaternion)

    def enu2xyz(self, enus):
        es, ns, us = enus.T
        xyzs = np.vstack([es, -us, ns]).T
        return xyzs

    def world_to_camera_xyzs(self, world_xyzs_in, cam_extrinsic_matrix):
        world_xyzs = np.ones([world_xyzs_in.shape[0], 4])
        world_xyzs[:, :3] = world_xyzs_in
        cam_coords_xyz = np.dot(cam_extrinsic_matrix, world_xyzs.T).T
        return cam_coords_xyz

    def camera_xyz_to_image_xy(self, cam_xyzs, cam_intrinsic_calibration_matrix):
        im_uvs = np.dot(cam_intrinsic_calibration_matrix, cam_xyzs.T).T
        im_uvs = (im_uvs[:, :2] / im_uvs[:, [2, 2]]).astype(np.int)
        return im_uvs

    def project_enus_to_imcoords(self, world_enus_in, cam_extrinsic_matrix=None):
        world_xyzs_in = self.enu2xyz(world_enus_in)
        world_enus = world_enus_in
        cam_plane_chosen_filt = np.ones(world_enus.shape[0]).astype(np.bool)
        if cam_extrinsic_matrix is not None:
            cam_xyzs = self.world_to_camera_xyzs(world_xyzs_in, cam_extrinsic_matrix)
            cam_plane_chosen_filt = cam_xyzs[:, 2] > 0
            world_xyzs_in = world_xyzs_in[cam_plane_chosen_filt]
            world_enus = world_enus_in[cam_plane_chosen_filt]

        world_xyzs = np.ones([world_enus.shape[0], 4])
        world_xyzs[:, :3] = world_xyzs_in

        im_uvs = np.dot(self.projection_matrix, world_xyzs.T).T
        im_uvs = (im_uvs[:, :2] / im_uvs[:, [2, 2]]).astype(np.int)

        l0 = np.dot(vec2zeromat(im_uvs[0]), np.dot(self.projection_matrix, world_xyzs.T)).T
        l0 = l0[:, :2] / l0[:, [2, 2]]

        l1 = np.dot(np.dot(vec2zeromat(im_uvs[0]), self.projection_matrix), world_xyzs.T).T
        l1 = l1[:, :2] / l1[:, [2, 2]]

        return im_uvs, world_enus_in[cam_plane_chosen_filt], cam_plane_chosen_filt

    def viz_points(self, world_enus_in):
        xys, chosen_world_enus, chosen_points_filt = self.project_enus_to_imcoords(world_enus_in, self.RT)
        xs, ys = xys.T
        im = np.zeros([self.im_h, self.im_w]).astype(np.uint8)
        f = np.logical_and(np.logical_and(xs > 0, xs < self.im_w), np.logical_and(ys > 0, ys < self.im_h))
        xs = xs[f]
        ys = ys[f]
        enus = chosen_world_enus[f]
        im[ys, xs] = 255
        for i in range(enus.shape[0]):
            cv2.putText(im, ','.join(list(map(str, enus[i]))), (int(xs[i]), int(ys[i])), cv2.FONT_HERSHEY_SIMPLEX,
                        .4, 200, thickness=1)

        return im, chosen_points_filt

    def viz_plane(self, side=100, plane_idx=0, plane_loc=0):
        es = np.arange(-side, side)
        ns = np.arange(-side, side)
        us = np.arange(-side, side)
        if plane_idx == 0:
            es = np.array([plane_loc])
        elif plane_idx == 1:
            ns = np.array([plane_loc])
        elif plane_idx == 2:
            us = np.array([plane_loc])
        enus = np.array(list(itertools.product(es, ns, us)))
        plane_viz, _ = self.viz_points(enus)
        return plane_viz


class View:

    def __init__(self, im_fpath, mask_fpath, pose_fpath):
        self.im_fpath = im_fpath
        self.mask_fpath = mask_fpath
        self.pose_fpath = pose_fpath
        self.im = None
        self.mask = None
        self.pose = None
        self.m_plane_bgr = np.array([241, 104, 146])
        self.m_cube_bgr = np.array([248, 71, 170])
        self.m_sky_bgr = np.array([73, 226, 210])
        self.cube_face_bgrs = np.array([[13, 125, 201],
                                        [165, 0, 69],
                                        [153, 0, 0],
                                        [0, 127, 104],
                                        [42, 46, 242]])
        self.cam = None
        self.im_h = 0
        self.im_w = 0
        self.loc_xyz = None
        self.loc_enu = None
        self.cam_quaternion = None
        self.init()

    def read_pose_loc(self, d):
        line2num = lambda i: float(d[i].replace('}', '').strip().split(':')[-1].split(',')[0].strip())
        q = np.array([line2num(i) for i in range(4)])
        xyz = np.array([line2num(i) for i in range(4, 7)])
        return q, xyz

    def init(self, loc_xyz=None, quaternion=None):
        self.im = cv2.imread(self.im_fpath)
        self.im_h, self.im_w, _ = self.im.shape
        self.mask = cv2.imread(self.mask_fpath)
        if loc_xyz is None or quaternion is None:
            with open(self.pose_fpath) as f:
                data = f.readlines()
                self.cam_quaternion, self.loc_xyz = self.read_pose_loc(data)
        else:
            if loc_xyz is not None:
                self.loc_xyz = np.array(loc_xyz)
            if quaternion is not None:
                self.cam_quaternion = np.array(quaternion)
        x, y, z = self.loc_xyz
        self.loc_enu = np.array([x, z, -y])
        self.cam = Camera(self.loc_enu, self.im_w, self.im_h, cam_quaternion=self.cam_quaternion)

    def find_right_top_bottom_tracking_points(self):
        m = np.abs(self.mask.astype(np.float) - self.m_cube_bgr.astype(np.float)).max(axis=-1)
        m[m == 0] = -1
        m[m > -1] = 0
        m[m == -1] = 255
        m = m.astype(np.uint8)
        polys, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        xys = np.squeeze(polys[0])
        xs = xys[:, 0]
        right_border_xys = xys[xs >= xs.max() - 3]
        top_xy = right_border_xys[right_border_xys[:, 1].argmin()]
        bottom_xy = right_border_xys[right_border_xys[:, 1].argmax()]
        cv2.imwrite('m.png', m)
        return top_xy, bottom_xy


class Build3D:

    def __init__(self, im_fpaths=None, mask_fpaths=None, pose_fpaths=None):
        if im_fpaths is not None and mask_fpaths is not None and pose_fpaths is not None:
            self.num_images = len(mask_fpaths)
            self.views = [View(im_fpaths[i], mask_fpaths[i], pose_fpaths[i]) for i in range(self.num_images)]

    def find_cube_side(self, cam_idx=[1, 2]):
        tracking_points_all = np.array([self.views[j].find_right_top_bottom_tracking_points() for j in cam_idx])
        top_bottom_xyzs = []
        errs = []
        for i in range(tracking_points_all.shape[1]):  # TODO: verify tracking_points
            xyz, enu, err = self.extract_3d_points(tracking_points_all[:, i], [self.views[j].cam for j in cam_idx])
            top_bottom_xyzs.append(xyz)
            errs.append(err)
        top_bottom_xyzs = np.array(top_bottom_xyzs)
        side_len = np.linalg.norm(top_bottom_xyzs[0] - top_bottom_xyzs[1])
        return side_len, top_bottom_xyzs

    def extract_3d_points(self, im_uvs, cams):
        num_views = len(cams)
        im_xy_zero_mats = np.array([vec2zeromat(xy) for xy in im_uvs])
        a = np.vstack([np.dot(im_xy_zero_mats[i], cams[i].projection_matrix) for i in range(num_views)])
        y = np.zeros(a.shape[0])
        y[-1] = 1
        a_inv = np.linalg.pinv(a)
        homogenous_coords = np.dot(a_inv, y).T
        x, y, z = homogenous_coords[:-1] / homogenous_coords[-1]
        enu = np.array([x, z, -y])
        xyz = np.array([x, y, z])
        err = np.linalg.norm((np.dot(a, a_inv) - np.eye(a.shape[0])).flatten())
        return xyz, enu, err


def vec2zeromat(xy):
    x, y = xy
    return np.array([[0, -1,  y],
                     [1,  0, -x],
                     [-y, x,  0]])


if __name__ == '__main__':
    # im_fpaths = np.array(glob(IMAGE_DIR + os.sep + '*'))
    # ids = [n.split(os.sep)[-1].split('_')[0] for n in im_fpaths]
    # id2fpath = lambda dir, id, suffix: dir + os.sep + id + '_' + suffix
    # mask_fpaths = [id2fpath(MASKS_DIR, id, '1.png') for id in ids]
    # pose_fpaths = [id2fpath(POSE_DIR, id, '2.txt') for id in ids]
    #
    # build3d = Build3D(im_fpaths, mask_fpaths, pose_fpaths)
    # cube_side, top_bottom_xyzs = build3d.find_cube_side([1, 2])
    # print('Cube side is', cube_side)

    # cube_side, top_bottom_xyzs = build3d.find_cube_side([4, 5])
    # print('Cube side is', cube_side)

    targ_enu = np.array([[-10, 20, 9]])
    # p0 = 639, 143
    # p1 = 963, 569

    # targ_enu = np.array([[-1, 20, 1]])
    # p0 = 1216, 656
    # p1 = 1699, 1144

    w = 2560
    h = 1440
    build3d = Build3D()

    cam0 = Camera([0, 0, 0], w, h)
    # v0 = cam0.viz_plane(10, 1, 20)
    # cv2.imwrite('v0.png', v0)
    im_uvs0, world_enus_in0, cam_plane_chosen_filt0 = cam0.project_enus_to_imcoords(targ_enu)

    cam1 = Camera([-2, 5, 3], w, h, [1, 0.08, -0.1, 0.04])
    # v1 = cam1.viz_plane(10, 1, 20)
    # cv2.imwrite('v1.png', v1)
    im_uvs1, world_enus_in1, cam_plane_chosen_filt1 = cam1.project_enus_to_imcoords(targ_enu)

    # tracked_xys = np.array([p0, p1])
    tracked_xys = np.vstack([im_uvs0, im_uvs1])

    pred_xyz, pred_enu, match_error = build3d.extract_3d_points(tracked_xys, [cam0, cam1])
    recons_err = np.linalg.norm(targ_enu - pred_enu)
    print(recons_err, match_error, targ_enu, pred_enu)
    k = 0

