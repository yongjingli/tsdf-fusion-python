import os
import cv2
import json
import copy
import numpy as np
import shutil
import trimesh
from local_utils import read_pc, project_points_to_depth, plot_depth_on_img
from local_utils import save_pc_to_ply, vox2world, cam2pix, \
    depth_2_points, save_plt_with_img, set_frustrum_by_cam_pts, \
    get_frustrum_cam_pts, get_corners, draw_projected_box3d, heading2rotmat, \
    generate_vox_coords


def get_input_infos(src_root, img_name):
    left_root = os.path.join(src_root, "leftPng")
    right_root = os.path.join(src_root, "rightPng")
    calib_root = os.path.join(src_root, "calib")
    label_root = os.path.join(src_root, "label")
    ply_root = os.path.join(src_root, "pcd")

    left_img_path = os.path.join(left_root, img_name)
    right_img_path = os.path.join(right_root, img_name)
    calib_path = os.path.join(calib_root, img_name.split(".")[0] + '.json')
    pcd_path = os.path.join(ply_root, img_name.split(".")[0] + '.pcd')
    label_mask_path = os.path.join(label_root, img_name.split(".")[0] + '.json')

    base_info = dict()
    base_info["left_img_path"] = left_img_path
    base_info["right_img_path"] = right_img_path
    base_info["calib_path"] = calib_path
    base_info["pcd_path"] = pcd_path
    base_info["label_mask_path"] = label_mask_path
    return base_info


def generat_occupy_gt_lidar(base_info):
    save_root = base_info["save_root"]
    if os.path.exists(save_root):
        shutil.rmtree(save_root)
    os.mkdir(save_root)

    xmin = base_info["xmin"]
    ymin = base_info["ymin"]
    zmin = base_info["zmin"]
    xmax = base_info["xmax"]
    ymax = base_info["ymax"]
    zmax = base_info["zmax"]
    voxel_size = base_info["voxel_size"]
    trunc_margin = 1 * voxel_size  # truncation on SDF

    src_root = base_info["src_root"]
    img_name = base_info["img_name"]
    if 1:
        base_info = get_input_infos(src_root, img_name)
        img_left = cv2.imread(base_info["left_img_path"])
        img_right = cv2.imread(base_info["right_img_path"])
        pts = read_pc(base_info["pcd_path"], min_distance=None, max_distance=None)

        calib_info = json.load(open(base_info["calib_path"]))
        cam_K = np.array(calib_info['intrinsics']).reshape(3, 3)
        baseline = calib_info['baseline']
        resolution = calib_info['resolution']

        if img_left.shape[0] != int(resolution[1]) or \
                img_left.shape[1] != int(resolution[0]):
            if 0:
                print("warnning no match resolution:{}".format(img_name))
                cam_K = cam_K * (img_left.shape[0] / float(resolution[1]))
                cam_K[-1, -1] = 1.
            else:
                scale = float(resolution[1]) / img_left.shape[0]
                img_left = cv2.resize(img_left, None, fx=scale, fy=scale)
                img_right = cv2.resize(img_right, None, fx=scale, fy=scale)

        res_h, res_w, _ = img_left.shape
        depth_img_left, disp_img_left = project_points_to_depth(pts, cam_K, res_w, res_h, baseline)

        # show by opencv
        y, x = np.nonzero(depth_img_left)
        # print(y.shape)
        # print(depth_img_left.shape)
        # print(y.shape[0]/ (depth_img_left.shape[0] * depth_img_left.shape[1]))
        # exit(1)
        z_no_overlap = depth_img_left[y, x]
        viz_depth_img = copy.deepcopy(img_left)
        if 0 not in x.shape:
            viz_depth_img = plot_depth_on_img(viz_depth_img, y, x, z_no_overlap)
        viz_depth_img = cv2.hconcat([img_left, viz_depth_img])
        cv2.imwrite(os.path.join(save_root, img_name.replace(".png", ".jpg")), viz_depth_img)

        depth_img_left = depth_img_left/1000

    else:
        # 112
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
        # depth_img_right = depth_img_right * 1000  # m->mm

        with open(instrinsic_path, 'r') as fp:
            instrinsic_info = json.load(fp)
            instrinsic = instrinsic_info['stereo_left']['pinhole_mat']
            instrinsic = np.array(instrinsic)

            instrinsic_right = instrinsic_info['stereo_right']['pinhole_mat']
            instrinsic_right = np.array(instrinsic_right)

            baseline = 110  # fix
        cam_K = instrinsic


    vox_coords, vol_origin, vol_bnds, vol_dim = generate_vox_coords(xmin=xmin, ymin=ymin, zmin=zmin,
                        xmax=xmax, ymax=ymax, zmax=zmax,
                        voxel_size=voxel_size)
    # Convert voxel grid coordinates to pixel coordinates
    cam_pts = vox2world(vol_origin, vox_coords, voxel_size)
    # 将世界坐标转换到相机坐标系下，由于只预测单帧的结果，因此不进行这样的转换
    # cam_pts = rigid_transform(cam_pts, np.linalg.inv(cam_pose))

    pix_z = cam_pts[:, 2]
    # 将点云投影回到图像上，跟利用点云生成深度估计预测的真数类似
    pix = cam2pix(cam_pts, cam_K)
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
    unvalid_pts_mask = np.logical_and(valid_pix, depth_val == 0)  # 在图像内但是取到的深度为0的点

    if 0:
        # 在图像内但是取到深度为0的点，在计算loss的时候需要mask
        # 这个campoint 代表的是grid的中心的点，假设grid有点云，即一个voxel中存在点，
        # 但是voxel里的点跟voxel的点投影到图像中，并不一定是重叠的图像坐标，此时该campoint代表的voxel
        # 将取不到深度值从而被mask掉，这种情况发现出现的非常多
        # 对这些被mask的点再处理下

        cam_pts_mask = cam_pts[unvalid_pts_mask]
        cam_mask_num = cam_pts_mask.shape[0]
        dx_dy_dz = np.ones((cam_mask_num, 3), dtype=cam_pts_mask.dtype) * voxel_size
        angle = np.zeros((cam_mask_num, 1), dtype=cam_pts_mask.dtype)
        mask_boxes3d = np.concatenate([cam_pts_mask, dx_dy_dz, angle], axis=1)
        mast_corners = get_corners(mask_boxes3d)


        # (N, 8, 3) -> (N*8, 3)
        corners_mask_pts = mast_corners.reshape(-1, 3)
        # mask_pix_z = corners_mask_pts[:, 2]
        mask_pix = cam2pix(corners_mask_pts, cam_K)
        mask_pix_x, mask_pix_y = mask_pix[:, 0], mask_pix[:, 1]

        im_h, im_w = depth_img_left.shape
        mask_valid_pix = np.logical_and(mask_pix_x >= 0, np.logical_and(mask_pix_x < im_w,
                                   np.logical_and(mask_pix_y >= 0, np.logical_and(mask_pix_y < im_h, pix_z > 0))))
        mask_depth_val = np.zeros(mask_valid_pix.shape)
        mask_depth_val[mask_valid_pix] = depth_img_left[mask_pix_y[valid_pix], mask_pix_x[valid_pix]]

        # 8个角点，将取到的深度的平均作为该grid的深度
        # TODo, 这个思想放宽到上面的处理，重新在第二个版本中实现

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

    save_pc_to_ply(os.path.join(save_root, "vox_coords.ply"), cam_pts)
    save_pc_to_ply(os.path.join(save_root, "vox_coords_occupy.ply"), cam_pts[valid_pts_occupy], np.array([255, 0, 0]))
    save_pc_to_ply(os.path.join(save_root, "vox_coords_free.ply"), cam_pts[valid_pts_free], np.array([0, 255, 0]))
    save_pc_to_ply(os.path.join(save_root, "vox_coords_outside.ply"), cam_pts[unvalid_pts_outside], np.array([255, 255, 255]))
    save_pc_to_ply(os.path.join(save_root, "vox_coords_block.ply"), cam_pts[unvalid_pts_block], np.array([0, 0, 255]))
    save_pc_to_ply(os.path.join(save_root, "vox_coords_mask.ply"), cam_pts[unvalid_pts_mask], np.array([0, 0, 0]))

    # 将深度图保存成点云
    save_plt_with_img(os.path.join(save_root, "pts.ply"), depth_img_left * 1000, img_left, cam_K)

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
    cv2.imwrite(os.path.join(save_root, "img_occupy.jpg"), img_occupy)


    # 为占据的网格分配颜色
    colors = np.array(colors).reshape(-1, 3)
    save_pc_to_ply(os.path.join(save_root,"vox_coords_occupy_color.ply"), cam_pts[valid_pts_occupy], colors[:, ::-1])

    # save gt
    print("vox_coords:", vox_coords)
    print("vol_origin:", vol_origin)
    print("vol_bnds:", vol_bnds)
    print("vol_dim:", vol_dim)

    # 0: for mask, 1: for unknow, 2: for occupy, 3: for free

    gt_frustrum = np.zeros(vol_dim, dtype=np.int32)
    cam_pts_occupy = cam_pts[valid_pts_occupy]
    cam_pts_free = cam_pts[valid_pts_free]
    cam_pts_outside = cam_pts[unvalid_pts_outside]
    cam_pts_block = cam_pts[unvalid_pts_block]

    outside_value = 1
    gt_frustrum = set_frustrum_by_cam_pts(cam_pts_outside, vol_origin, voxel_size, outside_value, gt_frustrum)

    block_value = 1
    gt_frustrum = set_frustrum_by_cam_pts(cam_pts_block, vol_origin, voxel_size, block_value, gt_frustrum)

    occupy_value = 2
    gt_frustrum = set_frustrum_by_cam_pts(cam_pts_occupy, vol_origin, voxel_size, occupy_value, gt_frustrum)

    free_value = 3
    gt_frustrum = set_frustrum_by_cam_pts(cam_pts_free, vol_origin, voxel_size, free_value, gt_frustrum)

    np.save(os.path.join(save_root, "gt_frustrum.npy"), gt_frustrum)
    np.save(os.path.join(save_root, "gt_cam_pts_occupy.npy"), cam_pts_occupy)

    # draw_occupy_grid_on_img()
    # 将cam_pts转为boxes3d的形式, 这个坐标系是y向前的坐标系
    # boxes3d: (N, 7)[x, y, z, dx, dy, dz, heading],
    # (x, y, z) is the box center
    # 这个点就是grid的中心，因为是按照这个点投影到图像上的
    cam_num = cam_pts_occupy.shape[0]
    dx_dy_dz = np.ones((cam_num, 3), dtype=cam_pts_occupy.dtype) * voxel_size
    angle = np.zeros((cam_num, 1), dtype=cam_pts_occupy.dtype)

    boxes3d = np.concatenate([cam_pts_occupy, dx_dy_dz, angle], axis=1)
    # boxes3d = boxes3d.astype(gt_cam_pts_occupy.dtype)

    corners = get_corners(boxes3d)
    corners_pts = corners.reshape(-1, 3)

    pix_corners = cam2pix(corners_pts, cam_K)
    pix_corners = pix_corners.reshape(-1, 8, 2)
    for pix_corner in pix_corners:
        img_left = draw_projected_box3d(img_left, pix_corner, color=(0, 255, 0), thickness=1)

    pix = cam2pix(cam_pts_occupy, cam_K)
    pix_x, pix_y = pix[:, 0], pix[:, 1]
    colors = []
    for i in range(pix_x.shape[0]):
        occupy_x = int(pix_x[i])
        occupy_y = int(pix_y[i])
        colors.append(img_left[occupy_y, occupy_x])
        cv2.circle(img_left, (occupy_x, occupy_y), 2, (0, 0, 255), 1)
    cv2.imwrite(os.path.join(save_root, "img_occupy_grid.jpg"), img_left)


    # draw_occupy_grid_on_pc()
    cam_num = cam_pts_occupy.shape[0]
    dx_dy_dz = np.ones((cam_num, 3), dtype=cam_pts_occupy.dtype) * voxel_size
    angle = np.zeros((cam_num, 1), dtype=cam_pts_occupy.dtype)

    scene = trimesh.scene.Scene()
    for i in range(cam_num):
    # for i in range(1):
        ctr = cam_pts_occupy[i, :3]
        lengths = dx_dy_dz[i, :3]
        trns = np.eye(4)
        trns[0:3, 3] = ctr
        trns[3, 3] = 1.0
        trns[0:3, 0:3] = heading2rotmat(angle[i, 0])
        box_trimesh_fmt = trimesh.creation.box(lengths, trns)
        box_trimesh_fmt.visual.face_colors = (.6, .6, .6)
        scene.add_geometry(box_trimesh_fmt)

    mesh_list = trimesh.util.concatenate(scene.dump())
    # save to obj file
    out_filename = os.path.join(save_root, "pc_occupy_grid.obj")
    trimesh.io.export.export_mesh(mesh_list, out_filename, file_type='obj')


if __name__ == "__main__":
    print("Start")
    base_info = {
            "src_root": "/mnt/data10/liyj/programs/tsdf-fusion-python/local_files/data/px1_occupy_example",
            "img_name": "7b5d6ecc-7f15-11ec-bbd6-7c10c921acb3.png",
            "save_root": "./data/occupy_gt_lidar",
            "xmin": -4,
            "xmax": 4,
            "ymin": -2,
            "ymax": 1,
            "zmin": 0,
            "zmax": 5,
            "voxel_size": 0.1,
            # "trunc_margin": 1 * voxel_size,   # truncation on SDF
    }
    generat_occupy_gt_lidar(base_info)

    print("End")