import numpy as np
from numba import njit, prange


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

if __name__ == "__main__":
    print("Start")
