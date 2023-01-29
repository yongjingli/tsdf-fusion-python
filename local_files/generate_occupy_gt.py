import os
import copy
import cv2
import json
import numpy as np
from local_utils import save_pc_to_ply, vox2world, cam2pix, depth_2_points, save_plt_with_img


def generate_vox_coords(xmin=-5, ymin=-2, zmin=0,
                        xmax=5, ymax=2, zmax=5,
                        voxel_size=0.2):

    vol_bnds = np.array([xmin,  xmax, ymin, ymax,  zmin, zmax]).reshape(3, 2)
    voxel_size = float(voxel_size)

    vol_dim = np.ceil((vol_bnds[:, 1] - vol_bnds[:, 0])/voxel_size).copy(order='C').astype(int)

    # Adjust volume bounds and ensure C-order contiguous
    # self._vol_dim = np.ceil((self._vol_bnds[:,1]-self._vol_bnds[:,0])/self._voxel_size).copy(order='C').astype(int)
    vol_bnds[:, 1] = vol_bnds[:, 0] + vol_dim * voxel_size
    vol_origin = vol_bnds[:, 0].copy(order='C').astype(np.float32)


    print("Voxel volume size: {} x {} x {} - # points: {:,}".format(
      vol_dim[0], vol_dim[1], vol_dim[2],
      vol_dim[0]*vol_dim[1]*vol_dim[2])
    )

    xv, yv, zv = np.meshgrid(
        range(vol_dim[0]),
        range(vol_dim[1]),
        range(vol_dim[2]),
        indexing='ij'
    )
    vox_coords = np.concatenate([
        xv.reshape(1, -1),
        yv.reshape(1, -1),
        zv.reshape(1, -1)
    ], axis=0).astype(int).T

    return vox_coords, vol_origin, vol_bnds, vol_dim


def generate_occupy_gt():
    xmin = -5
    ymin = -2
    zmin = 0
    xmax = 5
    ymax = 2
    zmax = 5
    voxel_size = 0.1
    trunc_margin = 1 * voxel_size  # truncation on SDF

    left_img_path = "/mnt/data10/liyj/programs/tsdf-fusion-python/local_files/data/kujiale_data/stereo_left/218.jpg"
    right_img_path = "/mnt/data10/liyj/programs/tsdf-fusion-python/local_files/data/kujiale_data/stereo_right/218.jpg"

    left_depth_path = "/mnt/data10/liyj/programs/tsdf-fusion-python/local_files/data/kujiale_data/stereo_left/218dep.npy"
    right_depth_path = "/mnt/data10/liyj/programs/tsdf-fusion-python/local_files/data/kujiale_data/stereo_right/218dep.npy"
    instrinsic_path = "/mnt/data10/liyj/programs/tsdf-fusion-python/local_files/data/kujiale_data/intrinsic/218.json"

    img_left = cv2.imread(left_img_path)
    img_right = cv2.imread(right_img_path)
    depth_img_left = np.load(left_depth_path)
    # depth_img_left = depth_img_left * 1000  # m->mm
    depth_img_left = depth_img_left

    depth_img_right = np.load(right_depth_path)
    depth_img_right = depth_img_right * 1000  # m->mm

    with open(instrinsic_path, 'r') as fp:
        instrinsic_info = json.load(fp)
        instrinsic = instrinsic_info['stereo_left']['pinhole_mat']
        instrinsic = np.array(instrinsic)

        instrinsic_right = instrinsic_info['stereo_right']['pinhole_mat']
        instrinsic_right = np.array(instrinsic_right)

        baseline = 110  # fix


    vox_coords, vol_origin, vol_bnds, vol_dim = generate_vox_coords(xmin=xmin, ymin=ymin, zmin=zmin,
                        xmax=xmax, ymax=ymax, zmax=zmax,
                        voxel_size=voxel_size)

    # Convert voxel grid coordinates to pixel coordinates
    cam_pts = vox2world(vol_origin, vox_coords, voxel_size)
    # 将世界坐标转换到相机坐标系下，由于只预测单帧的结果，因此不进行这样的转换
    # cam_pts = rigid_transform(cam_pts, np.linalg.inv(cam_pose))

    pix_z = cam_pts[:, 2]
    # 将点云投影回到图像上，跟利用点云生成深度估计预测的真数类似
    pix = cam2pix(cam_pts, instrinsic)
    pix_x, pix_y = pix[:, 0], pix[:, 1]


    im_h, im_w = depth_img_left.shape
    # Eliminate pixels outside view frustum
    # frustum 的点在图像内
    valid_pix = np.logical_and(pix_x >= 0, np.logical_and(pix_x < im_w,
                               np.logical_and(pix_y >= 0, np.logical_and(pix_y < im_h, pix_z > 0))))

    depth_val = np.zeros(pix_x.shape)
    depth_val[valid_pix] = depth_img_left[pix_y[valid_pix], pix_x[valid_pix]]


    # TODO, 判断该点是否为一定范围内的点被占据的点， 其他不被占据的点是在前面还是被前面的点挡住了， 被挡住的点也是无法进行判断的，明天继续， 20230128
    # Integrate TSDF
    depth_diff = depth_val - pix_z
    valid_pts = np.logical_and(depth_val > 0, depth_diff >= -trunc_margin)    # 可以被判断的点，被占据的点或者不被占据的点

    # 可判断是否被占据的点
    depth_diff_abs = abs(depth_val - pix_z)
    valid_pts_occupy = np.logical_and(valid_pts, depth_diff_abs <= trunc_margin)    # 在图像内被占据的点
    valid_pts_free = np.logical_and(valid_pts, depth_diff_abs > trunc_margin)       # 在图像内不被占据的点

    # 在图像外或者被挡住无法判断的点
    unvalid_pts_outside = ~valid_pix       # 不在图像内的点
    unvalid_pts_block = np.logical_and(depth_val > 0, depth_diff < -trunc_margin)   # 在图像内，但是被前面的点挡住

    # 在图像内但是取到深度为0的点，在计算loss的时候需要mask
    unvalid_pts_mask = np.logical_and(valid_pix, depth_val == 0)  # 在图像内但是取到的深度为0的点

    print("all points:", pix_x.shape[0])
    print("occupy points:", valid_pts_occupy.sum())
    print("free points:", valid_pts_free.sum())
    print("outside points:", unvalid_pts_outside.sum())
    print("block points:", unvalid_pts_block.sum())
    print("mask points:", unvalid_pts_mask.sum())
    print("sum points:", valid_pts_occupy.sum() +
                          valid_pts_free.sum() +
                          unvalid_pts_outside.sum() +
                          unvalid_pts_block.sum() +
                          unvalid_pts_mask.sum())

    save_pc_to_ply("./data/vox_coords.ply", cam_pts)
    save_pc_to_ply("./data/vox_coords_occupy.ply", cam_pts[valid_pts_occupy], np.array([255, 0, 0]))
    save_pc_to_ply("./data/vox_coords_free.ply", cam_pts[valid_pts_free], np.array([0, 255, 0]))
    save_pc_to_ply("./data/vox_coords_outside.ply", cam_pts[unvalid_pts_outside], np.array([255, 255, 255]))
    save_pc_to_ply("./data/vox_coords_block.ply", cam_pts[unvalid_pts_block], np.array([0, 0, 255]))
    save_pc_to_ply("./data/vox_coords_mask.ply", cam_pts[unvalid_pts_mask], np.array([0, 0, 0]))

    # 将深度图保存成点云
    save_plt_with_img("./data/pts.ply", depth_img_left * 1000, img_left, instrinsic)

    # 将占据的点投影到原图中
    pix_x_occupy = pix_x[valid_pts_occupy]
    pix_y_occupy = pix_y[valid_pts_occupy]
    img_occupy = copy.deepcopy(img_left)
    colors = []
    for i in range(pix_x_occupy.shape[0]):
        occupy_x = int(pix_x_occupy[i])
        occupy_y = int(pix_y_occupy[i])
        colors.append(img_left[occupy_y, occupy_x])
        cv2.circle(img_occupy, (occupy_x, occupy_y), 2, (0, 0, 255), 1)
    cv2.imwrite("./data/img_occupy.jpg", img_occupy)


    # 为占据的网格分配颜色
    colors = np.array(colors).reshape(-1, 3)
    save_pc_to_ply("./data/vox_coords_occupy_color.ply", cam_pts[valid_pts_occupy], colors[:, ::-1])


if __name__ == "__main__":
    print("Start...")
    # generate_vox_coords()
    generate_occupy_gt()


