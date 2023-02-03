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
    generate_vox_coords, save_pc_to_voxel, get_img_view_line_points, save_img_view_line, \
    get_img_center_view_line_points, get_view_lines_and_depth, get_occupy_free_block_pts, \
    get_gt_frustrum, draw_pts_voxel_on_img, pts_project_in_img, get_outside_pts


# 前面几个是采用tsdf的方法进行真值的生成
# 采用tsdf的方式，将voxel投影会深度图的时候，将voxel的中心点代表了voxel取深度
# 这种取深度的方式，有一些问题，voxel是一个有长宽高的box，当voxel的存在点时，这些点投影到图像中与voxel中心
# 的坐标差异可能比较大，这样就会导致，voxel里存在点，但是由于voxel的中心投影取不到深度而mask掉
# 也即，voxel里存在点云，但是生成真值是被mask的状态

# 下面采用视锥的方式
# 对于图像中的每个点， 都对应一条光线，将这条光线上的voxel进行赋值
# 对于frustrum中没有对应视锥线的voxel进行mask
# 对于将光锥线对于voxel设置为occupy，或者前后的几个voxel也设置为occupy
# 对于光锥线上occupy前面的voxel设置为free，后面的voxel则设置为block

# 对图像上的某个像素，如何判断frustrum中哪些voxel是属于这个像素对应的视锥线
# 方法1：参考lss的做法，对于这个像素，穷举不同的深度，将不同深度对于voxel作为该视锥上的voxel
# 这样的话，该视锥上某些voxel可能也是没法设置，需要看穷举的分辨率

# 方法2：得到视锥线在3d中表示（起点和终点），得到所有经过这个视锥线的voxel
# 先采用方法2看看效果


def get_frustrum_pts(base_info):
    xmin = base_info["xmin"]
    ymin = base_info["ymin"]
    zmin = base_info["zmin"]
    xmax = base_info["xmax"]
    ymax = base_info["ymax"]
    zmax = base_info["zmax"]
    voxel_size = base_info["voxel_size"]

    vox_coords, vol_origin, vol_bnds, vol_dim = generate_vox_coords(xmin=xmin, ymin=ymin, zmin=zmin,
                        xmax=xmax, ymax=ymax, zmax=zmax,
                        voxel_size=voxel_size)
    # Convert voxel grid coordinates to pixel coordinates
    frustrum_pts = vox2world(vol_origin, vox_coords, voxel_size)
    return frustrum_pts


def get_input_infos(base_info):
    src_root = base_info["src_root"]
    img_name = base_info["img_name"]

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

    img_left = cv2.imread(left_img_path)
    img_right = cv2.imread(right_img_path)
    # pts = read_pc(base_info["pcd_path"], min_distance=None, max_distance=None)

    calib_info = json.load(open(calib_path))
    cam_K = np.array(calib_info['intrinsics']).reshape(3, 3)
    baseline = calib_info['baseline']
    resolution = calib_info['resolution']
    # use crestereo to get dense img depth
    disp_img_left_path = "/mnt/data10/liyj/programs/tsdf-fusion-python/local_files/data/px1_occupy_example/cre_stereo_disp/7b5d6ecc-7f15-11ec-bbd6-7c10c921acb3_disp.npy"
    disp_img_left = np.load(disp_img_left_path)
    fx, fy = cam_K[0, 0], cam_K[1, 1]
    depth_img_left = fx * baseline / (disp_img_left + 1e-8)
    depth_img_left = depth_img_left / 1000 # mm -> m
    return img_left, depth_img_left, cam_K


def get_input_infos2(base_info):
    src_root = base_info["src_root"]
    img_name = base_info["img_name"]

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

    img_left = cv2.imread(left_img_path)
    img_right = cv2.imread(right_img_path)
    pts = read_pc(pcd_path, min_distance=None, max_distance=None)

    calib_info = json.load(open(calib_path))
    cam_K = np.array(calib_info['intrinsics']).reshape(3, 3)
    baseline = calib_info['baseline']
    resolution = calib_info['resolution']

    scale = float(resolution[1]) / img_left.shape[0]
    img_left = cv2.resize(img_left, None, fx=scale, fy=scale)
    img_right = cv2.resize(img_right, None, fx=scale, fy=scale)


    res_h, res_w, _ = img_left.shape
    depth_img_left, disp_img_left = project_points_to_depth(pts, cam_K, res_w, res_h, baseline)
    depth_img_left = depth_img_left/1000
    return img_left, depth_img_left, cam_K


def generate_occupy_gt_by_view_line(base_info):
    save_root = base_info["save_root"]
    if os.path.exists(save_root):
        shutil.rmtree(save_root)
    os.mkdir(save_root)

    save_voxel = 0

    # 得到frustrum_pts，在相机坐标系的坐标，代表每个voxel的中心
    frustrum_pts = get_frustrum_pts(base_info)
    save_pc_to_ply(os.path.join(save_root, "frustrum_pts.ply"), frustrum_pts)
    # if save_voxel:
    #     save_pc_to_voxel(os.path.join(save_root, "frustrum_pts_voxel.ply"), frustrum_pts[:100, :],
    #                      base_info["voxel_size"], color=[0, 0, 255])

    # img_left, depth_img_left, cam_K = get_input_infos(base_info)   # cre stereo pred dense disp
    img_left, depth_img_left, cam_K = get_input_infos2(base_info)  # lidar project parse disp
    # 将深度图保存成点云
    save_plt_with_img(os.path.join(save_root, "pts.ply"), depth_img_left * 1000, img_left, cam_K)

    # 得到图像上每个点，在深度为max-depth时对应的点坐标, 当对应的深度图位置为0时，过滤，减少处理的像素数量
    view_points, img_points = get_img_view_line_points(base_info, img_left, cam_K, depth_img_left, interval=1)
    # view_points, img_points = get_img_center_view_line_points(base_info, img_left, cam_K, interval=100)
    # 绘制[0, 0, 0]到在深度为max-depth时对应的点坐标的连线
    save_img_view_line(os.path.join(save_root, "view_lines.poly"), view_points)

    # 每个坐标点对应一条view-line, 每条view-line包含
    # 在z方向上，[0, z_max, voxel_size]列举的z中，对应的voxel坐标
    view_lines, view_lines_depth = get_view_lines_and_depth(base_info, view_points, img_points, depth_img_left)
    occupy_pts, free_pts, block_pts = get_occupy_free_block_pts(base_info, view_lines, view_lines_depth)

    # 因为occupy取的视锥前后的grid，并非原来视锥的点云，作了取整操作，投影回来的时候，可能超出图像的坐标
    occupy_pts = pts_project_in_img(occupy_pts, cam_K, depth_img_left)
    free_pts = pts_project_in_img(free_pts, cam_K, depth_img_left)
    block_pts = pts_project_in_img(block_pts, cam_K, depth_img_left)
    outside_pts = get_outside_pts(frustrum_pts, cam_K, depth_img_left)

    save_pc_to_ply(os.path.join(save_root, "pts_occupy.ply"), occupy_pts, color=np.array([255, 0, 0]))
    save_pc_to_ply(os.path.join(save_root, "pts_free.ply"), free_pts, color=np.array([0, 255, 0]))
    save_pc_to_ply(os.path.join(save_root, "pts_block.ply"), block_pts, color=np.array([0, 0, 255]))
    save_pc_to_ply(os.path.join(save_root, "pts_outside.ply"), outside_pts, color=np.array([255, 255, 255]))

    if save_voxel:
        # 将占据绘制成网格的形式
        print("start save pts_occupy_voxel.ply....")
        save_pc_to_voxel(os.path.join(save_root, "pts_occupy_voxel.ply"), occupy_pts,
                         base_info["voxel_size"], color=[255, 0, 0])
        print("start save pts_occupy_voxel.ply")

    gt_frustrum = get_gt_frustrum(base_info, occupy_pts, free_pts, block_pts, outside_pts)
    np.save(os.path.join(save_root, "gt_frustrum.npy"), gt_frustrum)

    # 将占据的voxel绘制到图像上
    img_occupy_voxel = copy.deepcopy(img_left)
    img_occupy_voxel = draw_pts_voxel_on_img(img_occupy_voxel, occupy_pts, cam_K, base_info["voxel_size"])
    cv2.imwrite(os.path.join(save_root, "img_occupy_voxel.jpg"), img_occupy_voxel)


def parse_gt_frustrum(base_info):
    save_root = base_info["save_root"]

    gt_frustrum_path = os.path.join(save_root, "gt_frustrum.npy")
    gt_frustrum = np.load(gt_frustrum_path)

    xmin = base_info["xmin"]
    ymin = base_info["ymin"]
    zmin = base_info["zmin"]
    xmax = base_info["xmax"]
    ymax = base_info["ymax"]
    zmax = base_info["zmax"]
    voxel_size = base_info["voxel_size"]

    vox_coords, vol_origin, vol_bnds, vol_dim = generate_vox_coords(xmin=xmin, ymin=ymin, zmin=zmin,
                        xmax=xmax, ymax=ymax, zmax=zmax,
                        voxel_size=voxel_size)

    block_value = base_info["block_value"]
    free_value = base_info["free_value"]
    occupy_value = base_info["occupy_value"]
    outside_value = base_info["outside_value"]
    mask_value = base_info["mask_value"]

    cam_pts_block = get_frustrum_cam_pts(gt_frustrum, vol_origin, voxel_size, block_value)
    cam_pts_occupy = get_frustrum_cam_pts(gt_frustrum, vol_origin, voxel_size, occupy_value)
    cam_pts_free = get_frustrum_cam_pts(gt_frustrum, vol_origin, voxel_size, free_value)
    cam_pts_outside = get_frustrum_cam_pts(gt_frustrum, vol_origin, voxel_size, outside_value)
    cam_pts_mask = get_frustrum_cam_pts(gt_frustrum, vol_origin, voxel_size, mask_value)

    save_pc_to_ply(os.path.join(save_root, "pts_occupy_parse.ply"), cam_pts_occupy, color=np.array([255, 0, 0]))
    save_pc_to_ply(os.path.join(save_root, "pts_free_parse.ply"), cam_pts_free, color=np.array([0, 255, 0]))
    save_pc_to_ply(os.path.join(save_root, "pts_block_parse.ply"), cam_pts_block, color=np.array([0, 0, 255]))
    save_pc_to_ply(os.path.join(save_root, "pts_outside_parse.ply"), cam_pts_outside, color=np.array([255, 255, 255]))
    save_pc_to_ply(os.path.join(save_root, "pts_mask_parse.ply"), cam_pts_mask, color=np.array([0, 0, 0]))



if __name__ == "__main__":
    print("Start")
    base_info = {
        "src_root": "/mnt/data10/liyj/programs/tsdf-fusion-python/local_files/data/px1_occupy_example",
        "img_name": "7b5d6ecc-7f15-11ec-bbd6-7c10c921acb3.png",
        "save_root": "./data/lss_occupy_gt_lidar",
        "xmin": -4,
        "xmax": 4,
        "ymin": -2,
        "ymax": 1,
        "zmin": 0,
        "zmax": 5,
        "voxel_size": 0.1,
        # "trunc_margin": 1 * voxel_size,   # truncation on SDF
        "mask_value": 0,
        "block_value": 1,
        "free_value": 2,
        "occupy_value": 3,
        "outside_value": 4,
    }
    # generate_occupy_gt_by_view_line(base_info)
    parse_gt_frustrum(base_info)

    print("End")