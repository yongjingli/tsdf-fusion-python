import numpy as np
from numba import njit, prange
import cv2
from matplotlib import cm
import open3d as o3d


@njit(parallel=True)
def vox2world(vol_origin, vox_coords, vox_size):
    """Convert voxel grid coordinates to world coordinates.
    """
    vol_origin = vol_origin.astype(np.float32)
    vox_coords = vox_coords.astype(np.float32)
    cam_pts = np.empty_like(vox_coords, dtype=np.float32)
    for i in prange(vox_coords.shape[0]):
        for j in range(3):
            cam_pts[i, j] = vol_origin[j] + (vox_size * vox_coords[i, j])
    return cam_pts


@njit(parallel=True)
def cam2pix(cam_pts, intr):
    """Convert camera coordinates to pixel coordinates.
    """
    intr = intr.astype(np.float32)
    fx, fy = intr[0, 0], intr[1, 1]
    cx, cy = intr[0, 2], intr[1, 2]

    pix = np.empty((cam_pts.shape[0], 2), dtype=np.int64)
    for i in prange(cam_pts.shape[0]):
        pix[i, 0] = int(np.round((cam_pts[i, 0] * fx / cam_pts[i, 2]) + cx))
        pix[i, 1] = int(np.round((cam_pts[i, 1] * fy / cam_pts[i, 2]) + cy))
    return pix


def save_pc_to_ply(s_path, pc, color=None):
    x = pc[:, 0]
    y = pc[:, 1]
    z = pc[:, 2]

    points = []
    if color is None:
        color = np.array([255, 255, 255])
    if len(color.shape) == 1:
        color = color.reshape(-1, 3).repeat(x.shape[0], axis=0)

    for X, Y, Z, C in zip(x, y, z, color):
        points.append("%f %f %f %d %d %d 0\n" % (X, Y, Z, C[0], C[1], C[2]))

    file = open(s_path, "w")
    file.write('''ply
              format ascii 1.0
              element vertex %d
              property float x
              property float y
              property float z
              property uchar red
              property uchar green
              property uchar blue
              property uchar alpha
              end_header
              %s
              ''' % (len(points), "".join(points)))
    file.close()


def depth_2_points(depth_img, img_left, cam_K):
    fx = cam_K[0, 0]
    fy = cam_K[1, 1]
    cx = cam_K[0, 2]
    cy = cam_K[1, 2]
    x, y, z, color = dep_to_pts(depth_img, img_left, fx,  fy, cx, cy)

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    z = z.reshape(-1, 1)
    pts = np.concatenate((x, y, z), axis=1)
    # pts = np.concatenate((pts, z), axis=1)
    return pts


def dep_to_pts(depth_img, rgb_img, fx, fy, cx, cy):
    # dont save max distance in plt, max distance 30 meter
    # depth_img[depth_img > (30 * 1000.)] = 0
    v, u = np.nonzero(depth_img)
    z = depth_img[v, u]
    z /= 1000.

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    color = rgb_img[v, u]
    return x, y, z, color

def save_plt_with_img(s_path, depth_img, img_left, cam_K):
    fx = cam_K[0, 0]
    fy = cam_K[1, 1]
    cx = cam_K[0, 2]
    cy = cam_K[1, 2]

    x, y, z, color = dep_to_pts(depth_img, img_left, fx,  fy, cx, cy)

    points = []
    for X, Y, Z, C in zip(x, y, z, color):
        points.append("%f %f %f %d %d %d 0\n" % (X, Y, Z, C[0], C[1], C[2]))

    file = open(s_path, "w")
    file.write('''ply
              format ascii 1.0
              element vertex %d
              property float x
              property float y
              property float z
              property uchar red
              property uchar green
              property uchar blue
              property uchar alpha
              end_header
              %s
              ''' % (len(points), "".join(points)))
    file.close()


def set_frustrum_by_cam_pts(cam_pts, vol_origin, voxel_size, value, frustrum):
    frustrum_indxs = (cam_pts - vol_origin)/float(voxel_size)
    frustrum_indxs = np.round(frustrum_indxs).astype(np.int32)
    indx_x = frustrum_indxs[:, 0]
    indx_y = frustrum_indxs[:, 1]
    indx_z = frustrum_indxs[:, 2]
    frustrum[indx_x, indx_y, indx_z] = value
    # indx_x, indx_y, indx_z = np.where(frustrum == value)
    return frustrum


def get_frustrum_cam_pts(frustrum, vol_origin, voxel_size, value):
    indx_x, indx_y, indx_z = np.where(frustrum == value)
    indx_x = indx_x.reshape(-1, 1)
    indx_y = indx_y.reshape(-1, 1)
    indx_z = indx_z.reshape(-1, 1)
    frustrum_indxs = np.concatenate([indx_x, indx_y, indx_z], axis=1)
    cam_pts = frustrum_indxs * voxel_size + vol_origin

    return cam_pts


def get_corners(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading],
            (x, y, z) is the box center
    Returns:
    """
    # template = (np.array((
    #     [1, 1, -1],
    #     [1, -1, -1],
    #     [-1, -1, -1],
    #     [-1, 1, -1],
    #     [1, 1, 1],
    #     [1, -1, 1],
    #     [-1, -1, 1],
    #     [-1, 1, 1],
    # )) / 2)

    ''' Draw 3d bounding box in image
        qs: (8,2) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''
    # 修改为当前的坐标系， x向右，0，3, 4，7 为正  z向前，0,1,4,5为正  y向下， 4, 5, 6, 7 为正

    template = (np.array((
        [1, -1, 1],     # 0
        [-1, -1, 1],   # 1
        [-1, -1, -1],   # 2
        [1, -1, -1],    # 3
        [1, 1, 1],      # 4
        [-1, 1, 1],     # 5
        [-1, 1, -1],    # 6
        [1, 1, -1],     # 7
    )) / 2)

    corners3d = np.tile(boxes3d[:, None, 3:6],
                        [1, 8, 1]) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.reshape(-1, 8, 3),
                                      boxes3d[:, 6]).reshape(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d


def draw_projected_box3d(image, qs, color=(255,255,255), thickness=2):
    ''' Draw 3d bounding box in image
        qs: (8,2) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''
    qs = qs.astype(np.int32)
    for k in range(0,4):
       #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
       i,j=k,(k+1)%4
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness) # use LINE_AA for opencv3

       i,j=k+4,(k+1)%4 + 4
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness)

       i,j=k,k+4
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness)
    return image


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:
    """
    cosa = np.cos(angle)
    sina = np.sin(angle)
    zeros = np.zeros(points.shape[0])
    ones = np.ones(points.shape[0])
    rot_matrix = np.stack(
        (cosa, sina, zeros, -sina, cosa, zeros, zeros, zeros, ones),
        axis=1).reshape(-1, 3, 3)
    points_rot = np.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = np.concatenate((points_rot, points[:, :, 3:]), axis=-1)
    return points_rot


def heading2rotmat(heading_angle):
    rotmat = np.zeros((3, 3))
    rotmat[2, 2] = 1
    cosval = np.cos(heading_angle)
    sinval = np.sin(heading_angle)
    rotmat[0:2, 0:2] = np.array([[cosval, -sinval], [sinval, cosval]])
    return rotmat


def read_pc(pcloud_filename, min_distance=None, max_distance=None):
    pc = o3d.io.read_point_cloud(pcloud_filename)
    pc = np.asarray(pc.points)
    pc = pc.reshape(-1, 3).astype(np.float32)
    if None not in [min_distance, max_distance]:
        distance_mask = np.bitwise_and(pc[:, 2] > min_distance, pc[:, 2] < max_distance)
        pc = pc[distance_mask, :]

    return pc


def remove_overlaps(u, v, z, img_w):
    u = np.array(u).flatten()
    v = np.array(v).flatten()
    z = np.array(z).flatten()
    # convert into 1d array
    ind = v*img_w+u
    sorted = np.argsort(ind)
    sorted_ind = ind[sorted]
    z = z[sorted]
    uni_val, idx_start, count = np.unique(sorted_ind, return_counts=True, return_index=True)
    res_u = []
    res_v = []
    res_z = []
    for u_val, cnt, id_st in zip(uni_val, count, idx_start):
        res_v.append(u_val // img_w)
        res_u.append(u_val % img_w)
        if cnt == 1:
            res_z.append(z[id_st])
        else:
            dup_z = z[id_st:id_st+cnt]
            res_z.append(dup_z[np.argsort(dup_z)[0]])
    return np.array(res_u), np.array(res_v), np.array(res_z)


def project_points_to_image(points, cam_K, filter_x):
    if cam_K.shape[1] == 3:
        cam_K = np.concatenate((cam_K, np.matrix([[0], [0], [0]])), axis=1)  # extend to 3x4
    # cam_points_list = []
    points = np.insert(points, 3, 1, axis=1).T
    # cond2: x<0 in lidar coordd
    if filter_x:
        points = np.delete(points,np.where(points[0,:]<0),axis=1)  # filter points

    cam_points = cam_K * points
    # get u,v,z
    cam_points[:2] /= cam_points[2, :]
    # data shape 4 x N
    return np.array(cam_points)


def project_points_to_depth(pts, cam_K,  res_w, res_h, baseline):
    fx = cam_K[0, 0]
    # fy = cam_K[1, 1]
    # cx = cam_K[0, 2]
    # cy = cam_K[1, 2]

    uvz = project_points_to_image(pts, cam_K, filter_x=False)

    # cond1: filter point out of canvas
    u, v, z = uvz[0, :], uvz[1, :], uvz[2, :]
    u_out = np.logical_or(u < 0, u >= res_w)
    v_out = np.logical_or(v < 0, v >= res_h)
    outlier = np.logical_or(u_out, v_out)
    uvz = np.delete(uvz, np.where(outlier), axis=1)
    u, v, z = uvz[0, :], uvz[1, :], uvz[2, :]

    # cond2: get closer points for overlap points
    x = np.clip(np.round(u).astype(np.int), 0, res_w - 1)
    y = np.clip(np.round(v).astype(np.int), 0, res_h - 1)
    x, y, z_no_overlap = remove_overlaps(x, y, z, res_w)

    # create depth image
    dep_img = np.zeros((res_h, res_w), dtype=np.float32)
    disp_img = np.zeros((res_h, res_w), dtype=np.float32)
    z_no_overlap *= 1000.  # m to mm
    disp = fx * baseline / z_no_overlap
    if 0 not in x.shape and 0 not in y.shape:
        dep_img[y, x] = z_no_overlap
        disp_img[y, x] = disp
    return dep_img, disp_img


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


def clip_cord(x, max_border):
    x = np.minimum(np.maximum(0, x), max_border - 1)
    return x


def plot_depth_on_img(viz_img, y, x, z_no_overlap):
    cmap = cm.get_cmap('rainbow_r')
    z_no_overlap = z_no_overlap / np.max(z_no_overlap)  # 归一化处理
    heatmap_cmapped = cmap(z_no_overlap)
    heatmap_cmapped = heatmap_cmapped[..., 0:3] * 255

    img_h, img_w, _ = viz_img.shape

    viz_img[y, x] = heatmap_cmapped
    viz_img[clip_cord(y - 1, img_h), x] = heatmap_cmapped
    viz_img[clip_cord(y + 1, img_h), x] = heatmap_cmapped
    viz_img[y, clip_cord(x - 1, img_w)] = heatmap_cmapped
    viz_img[y, clip_cord(x + 1, img_w)] = heatmap_cmapped
    viz_img[clip_cord(y - 1, img_h), clip_cord(x - 1, img_w)] = heatmap_cmapped
    viz_img[clip_cord(y + 1, img_h), clip_cord(x + 1, img_w)] = heatmap_cmapped
    viz_img[clip_cord(y - 1, img_h), clip_cord(x + 1, img_w)] = heatmap_cmapped
    viz_img[clip_cord(y + 1, img_h), clip_cord(x - 1, img_w)] = heatmap_cmapped
    viz_img = cv2.cvtColor(viz_img, cv2.COLOR_RGB2BGR)
    return viz_img


def save_pc_to_voxel(out_filename, pts, voxel_size, color=[0, 255, 0]):
    import trimesh
    cam_num = pts.shape[0]
    dx_dy_dz = np.ones((cam_num, 3), dtype=pts.dtype) * voxel_size
    angle = np.zeros((cam_num, 1), dtype=pts.dtype)

    ctr = pts[:, :3]
    lengths = dx_dy_dz[:, :3]

    scene = trimesh.scene.Scene()
    for i in range(cam_num):
        trns = np.eye(4)
        trns[0:3, 3] = ctr[i]
        trns[3, 3] = 1.0
        trns[0:3, 0:3] = heading2rotmat(angle[i, 0])
        box_trimesh_fmt = trimesh.creation.box(lengths[i], trns)

        box_trimesh_fmt.visual.face_colors = trimesh.visual.to_rgba(colors=color)
        box_trimesh_fmt.visual.vertex_colors = trimesh.visual.to_rgba(colors=color)
        scene.add_geometry(box_trimesh_fmt)

    mesh_list = trimesh.util.concatenate(scene.dump())
    # trimesh.io.export.export_mesh(mesh_list, out_filename, file_type='obj')
    trimesh.io.export.export_mesh(mesh_list, out_filename, file_type='ply')



def save_img_view_line(file_path, points):
    with open(file_path, "w") as fp:
        for point in points:
            fp.write("{} {} {}\n".format(0, 0, 0))
            fp.write("{} {} {}\n\n".format(point[0], point[1], point[2]))


def get_img_view_line_points(base_info, img_left, cam_K, depth_img_left, interval=10):
    points = []
    img_points = []
    # 给定一个图像坐标，得到视锥线
    img_h, img_w, _ = img_left.shape
    for pix_x in range(0, img_w-1, interval):
        for pix_y in range(0, img_h - 1, interval):
            pix_depth = depth_img_left[pix_y, pix_x]
            if pix_depth > 0:
                depth_max = base_info["zmax"]
                fx = cam_K[0, 0]
                fy = cam_K[1, 1]
                cx = cam_K[0, 2]
                cy = cam_K[1, 2]

                x = (pix_x - cx) * depth_max / fx
                y = (pix_y - cy) * depth_max / fy  # fy is the same as fx
                z = depth_max
                # 保存视锥线的终点
                points.append([x, y, z])
                img_points.append([pix_x, pix_y])
    return points, img_points


def get_img_center_view_line_points(base_info, img_left, cam_K, interval=10):
    points = []
    img_points = []
    # 给定一个图像坐标，得到视锥线
    img_h, img_w, _ = img_left.shape
    pix_x, pix_y = img_w // 2, img_h // 2
    depth_max = base_info["zmax"]

    fx = cam_K[0, 0]
    fy = cam_K[1, 1]
    cx = cam_K[0, 2]
    cy = cam_K[1, 2]

    x = (pix_x - cx) * depth_max / fx
    y = (pix_y - cy) * depth_max / fy  # fy is the same as fx
    z = depth_max

    points.append([x, y, depth_max])
    img_points.append([pix_x, pix_y])
    return points, img_points


def get_view_lines_and_depth(base_info, view_points, img_points, depth_img_left):
    z_max = base_info["zmax"]
    voxel_size = base_info["voxel_size"]
    view_lines = []
    view_lines_depth = []
    for i, view_point in enumerate(view_points):
        pix_x, pix_y = img_points[i]
        view_lines_depth.append(depth_img_left[pix_y, pix_x])

        x1, y1, z1 = 0, 0, 0
        x2, y2, z2 = view_point[0], view_point[1], view_point[2]
        # 在z方向上，求[0, z_max, voxel_size]范围内, 该视锥在相应深度对应的grid
        view_line = []
        for i in range(0, int(z_max/voxel_size)):
            z_voxel = voxel_size * i
            # # 空间直线的方程 (x - x1) / (x2 - x1) = (y - y1) / (y2 - y1) = (z - z1) / (z2 - z1)
            x_voxel = (z_voxel - z1) / (z2 - z1) * (x2 - x1) + x1
            y_voxel = (z_voxel - z1) / (z2 - z1) * (y2 - y1) + y1

            x_voxel = round(x_voxel/voxel_size) * voxel_size
            y_voxel = round(y_voxel/voxel_size) * voxel_size

            # view_line.append(frustrum_pts[x_voxel_indx, y_voxel_indx, i])
            view_line.append([x_voxel, y_voxel, round(z_voxel/voxel_size) * voxel_size])
        view_lines.append(np.array(view_line))
    return view_lines, view_lines_depth


def get_occupy_free_block_pts(base_info, view_lines, view_lines_depth):
    # 将view-line中，对应的实际深度前后的voxel作为occupy， 将后面挡住的的voxel作为block， 将前面没挡住的作为free
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

    vol_bnds_x = vol_bnds[0]
    vol_bnds_y = vol_bnds[1]
    vol_bnds_z = vol_bnds[2]
    occupy_pts = []
    free_pts = []
    block_pts = []

    for view_line, view_line_depth in zip(view_lines, view_lines_depth):
        if view_line_depth != 0:
            valid_pix = np.logical_and(view_line[:, 0] >= vol_bnds_x[0],
                                       np.logical_and(view_line[:, 0] < vol_bnds_x[1],
                                                      np.logical_and(view_line[:, 1] >= vol_bnds_y[0],
                                                                     np.logical_and(view_line[:, 1] < vol_bnds_y[1], view_line[:, 1] < vol_bnds_z[1]))))

            occupy_points_index = np.logical_and(view_line[:, 2] < view_line_depth + voxel_size,
                                           view_line[:, 2] > view_line_depth - voxel_size)

            occupy_points_index = np.logical_and(valid_pix, occupy_points_index)
            occupy_pts.append(view_line[occupy_points_index])

            free_points_index = view_line[:, 2] < view_line_depth - voxel_size
            free_points_index = np.logical_and(valid_pix, free_points_index)
            free_pts.append(view_line[free_points_index])

            block_points_index = view_line[:, 2] > view_line_depth + voxel_size
            block_points_index = np.logical_and(valid_pix, block_points_index)
            block_pts.append(view_line[block_points_index])

    occupy_pts = np.concatenate(occupy_pts, axis=0)
    free_pts = np.concatenate(free_pts, axis=0)
    block_pts = np.concatenate(block_pts, axis=0)
    return occupy_pts, free_pts, block_pts


def pts_project_in_img(pts, cam_K, depth_img):
    img_h, img_w = depth_img.shape
    pix = cam2pix(pts, cam_K)
    pix_x, pix_y = pix[:, 0], pix[:, 1]
    valid_pix = np.logical_and(pix_x >= 0,
                               np.logical_and(pix_x < img_w -1,
                                              np.logical_and(pix_y >= 0,
                                                             pix_y < img_h)))
    pts = pts[valid_pix]
    return pts


def get_gt_frustrum(base_info, occupy_pts, free_pts, block_pts, outside_pts):
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

    gt_frustrum = np.zeros(vol_dim, dtype=np.int32)

    block_value = base_info["block_value"]
    free_value = base_info["free_value"]
    occupy_value = base_info["occupy_value"]
    outside_value = base_info["outside_value"]

    gt_frustrum = set_frustrum_by_cam_pts(block_pts, vol_origin, voxel_size, block_value, gt_frustrum)
    gt_frustrum = set_frustrum_by_cam_pts(free_pts, vol_origin, voxel_size, free_value, gt_frustrum)
    gt_frustrum = set_frustrum_by_cam_pts(occupy_pts, vol_origin, voxel_size, occupy_value, gt_frustrum)
    gt_frustrum = set_frustrum_by_cam_pts(outside_pts, vol_origin, voxel_size, outside_value, gt_frustrum)

    return gt_frustrum


def draw_pts_voxel_on_img(img, cam_pts, cam_K, voxel_size):
    cam_num = cam_pts.shape[0]
    dx_dy_dz = np.ones((cam_num, 3), dtype=cam_pts.dtype) * voxel_size
    angle = np.zeros((cam_num, 1), dtype=cam_pts.dtype)

    boxes3d = np.concatenate([cam_pts, dx_dy_dz, angle], axis=1)
    # boxes3d = boxes3d.astype(gt_cam_pts_occupy.dtype)

    corners = get_corners(boxes3d)
    corners_pts = corners.reshape(-1, 3)

    pix_corners = cam2pix(corners_pts, cam_K)
    pix_corners = pix_corners.reshape(-1, 8, 2)
    for pix_corner in pix_corners:
        img = draw_projected_box3d(img, pix_corner, color=(0, 255, 0), thickness=1)

    pix = cam2pix(cam_pts, cam_K)
    pix_x, pix_y = pix[:, 0], pix[:, 1]
    colors = []
    for i in range(pix_x.shape[0]):
        occupy_x = int(pix_x[i])
        occupy_y = int(pix_y[i])

        colors.append(img[occupy_y, occupy_x])
        cv2.circle(img, (occupy_x, occupy_y), 2, (0, 0, 255), 1)
    return img


def get_outside_pts(cam_pts, cam_K, depth_img):
    pix = cam2pix(cam_pts, cam_K)
    pix_z = cam_pts[:, 2]
    pix_x, pix_y = pix[:, 0], pix[:, 1]

    im_h, im_w = depth_img.shape

    valid_pix = np.logical_and(pix_x >= 0, np.logical_and(pix_x < im_w,
                               np.logical_and(pix_y >= 0, np.logical_and(pix_y < im_h, pix_z > 0))))
    outside_pts = cam_pts[~valid_pix]
    return outside_pts


if __name__ == "__main__":
    print("Start")
