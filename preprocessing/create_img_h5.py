import h5py
import os
import numpy as np
CUR_PATH = os.path.dirname(os.path.realpath(__file__))
import cv2
import trimesh
import create_file_lst
from joblib import Parallel, delayed

rot90y = np.array([[0, 0, -1],
                   [0, 1, 0],
                   [1, 0, 0]], dtype=np.float32)

def getBlenderProj(az, el, distance_ratio, img_w=137, img_h=137):
    """Calculate 4x3 3D to 2D projection matrix given viewpoint parameters."""
    F_MM = 35.  # Focal length
    SENSOR_SIZE_MM = 32.
    PIXEL_ASPECT_RATIO = 1.  # pixel_aspect_x / pixel_aspect_y
    RESOLUTION_PCT = 100.
    SKEW = 0.
    CAM_MAX_DIST = 1.75
    CAM_ROT = np.asarray([[1.910685676922942e-15, 4.371138828673793e-08, 1.0],
                      [1.0, -4.371138828673793e-08, -0.0],
                      [4.371138828673793e-08, 1.0, -4.371138828673793e-08]])

    # Calculate intrinsic matrix.
# 2 atan(35 / 2*32)
    scale = RESOLUTION_PCT / 100
    # print('scale', scale)
    f_u = F_MM * img_w * scale / SENSOR_SIZE_MM
    f_v = F_MM * img_h * scale * PIXEL_ASPECT_RATIO / SENSOR_SIZE_MM
    # print('f_u', f_u, 'f_v', f_v)
    u_0 = img_w * scale / 2
    v_0 = img_h * scale / 2
    K = np.matrix(((f_u, SKEW, u_0), (0, f_v, v_0), (0, 0, 1)))

    # Calculate rotation and translation matrices.
    # Step 1: World coordinate to object coordinate.
    sa = np.sin(np.radians(-az))
    ca = np.cos(np.radians(-az))
    se = np.sin(np.radians(-el))
    ce = np.cos(np.radians(-el))
    R_world2obj = np.transpose(np.matrix(((ca * ce, -sa, ca * se),
                                          (sa * ce, ca, sa * se),
                                          (-se, 0, ce))))

    # Step 2: Object coordinate to camera coordinate.
    R_obj2cam = np.transpose(np.matrix(CAM_ROT))
    R_world2cam = R_obj2cam * R_world2obj
    cam_location = np.transpose(np.matrix((distance_ratio * CAM_MAX_DIST,
                                           0,
                                           0)))
    # print('distance', distance_ratio * CAM_MAX_DIST)
    T_world2cam = -1 * R_obj2cam * cam_location

    # Step 3: Fix blender camera's y and z axis direction.
    R_camfix = np.matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))
    R_world2cam = R_camfix * R_world2cam
    T_world2cam = R_camfix * T_world2cam

    RT = np.hstack((R_world2cam, T_world2cam))

    return K, RT

def get_rotate_matrix(rotation_angle1):
    cosval = np.cos(rotation_angle1)
    sinval = np.sin(rotation_angle1)

    rotation_matrix_x = np.array([[1, 0,        0,      0],
                                  [0, cosval, -sinval, 0],
                                  [0, sinval, cosval, 0],
                                  [0, 0,        0,      1]])
    rotation_matrix_y = np.array([[cosval, 0, sinval, 0],
                                  [0,       1,  0,      0],
                                  [-sinval, 0, cosval, 0],
                                  [0,       0,  0,      1]])
    rotation_matrix_z = np.array([[cosval, -sinval, 0, 0],
                                  [sinval, cosval, 0, 0],
                                  [0,           0,  1, 0],
                                  [0,           0,  0, 1]])
    scale_y_neg = np.array([
        [1, 0,  0, 0],
        [0, -1, 0, 0],
        [0, 0,  1, 0],
        [0, 0,  0, 1]
    ])

    neg = np.array([
        [-1, 0,  0, 0],
        [0, -1, 0, 0],
        [0, 0,  -1, 0],
        [0, 0,  0, 1]
    ])
    # y,z swap = x rotate -90, scale y -1
    # new_pts0[:, 1] = new_pts[:, 2]
    # new_pts0[:, 2] = new_pts[:, 1]
    #
    # x y swap + negative = z rotate -90, scale y -1
    # new_pts0[:, 0] = - new_pts0[:, 1] = - new_pts[:, 2]
    # new_pts0[:, 1] = - new_pts[:, 0]

    # return np.linalg.multi_dot([rotation_matrix_z, rotation_matrix_y, rotation_matrix_y, scale_y_neg, rotation_matrix_z, scale_y_neg, rotation_matrix_x])
    return np.linalg.multi_dot([neg, rotation_matrix_z, rotation_matrix_z, scale_y_neg, rotation_matrix_x])


def get_norm_matrix(sdf_h5_file):
    with h5py.File(sdf_h5_file, 'r') as h5_f:
        norm_params = h5_f['norm_params'][:]
        center, m, = norm_params[:3], norm_params[3]
        x,y,z = center[0], center[1], center[2]
        M_inv = np.asarray(
            [[m, 0., 0., 0.],
             [0., m, 0., 0.],
             [0., 0., m, 0.],
             [0., 0., 0., 1.]]
        )
        T_inv = np.asarray(
            [[1.0 , 0., 0., x],
             [0., 1.0 , 0., y],
             [0., 0., 1.0 , z],
             [0., 0., 0., 1.]]
        )
    return np.matmul(T_inv, M_inv)


def convert_img2h5(source_dir, target_dir, file_lst_dir, cats, sdf_dir):
    os.makedirs(target_dir, exist_ok=True)
    for keys, vals in cats.items():
        fs = []
        cat_target_dir = os.path.join(target_dir, vals)
        os.makedirs(cat_target_dir, exist_ok=True)
        lst_train_file = os.path.join(file_lst_dir, vals+"_train.lst")
        lst_test_file = os.path.join(file_lst_dir, vals+"_test.lst")

        with open(lst_train_file, "r") as f:
            list_obj = f.readlines()
            list_obj = [obj.rstrip('\r\n') for obj in list_obj]
            fs+=list_obj
        with open(lst_test_file, "r") as f:
            list_obj = f.readlines()
            list_obj = [obj.rstrip('\r\n') for obj in list_obj]
            fs+=list_obj
        repeat = len(fs)
        source_dir_lst = [source_dir for i in range(repeat)]
        cat_target_dir_lst = [cat_target_dir for i in range(repeat)]
        sdf_dir_lst = [sdf_dir for i in range(repeat)]
        vals_lst = [vals for i in range(repeat)]

        with Parallel(n_jobs=10) as parallel:
            parallel(delayed(gen_obj_img_h5)
            (source_dir, cat_target_dir, sdf_dir,vals, obj)
            for source_dir, cat_target_dir, sdf_dir, vals, obj in
                zip(source_dir_lst, cat_target_dir_lst, sdf_dir_lst, vals_lst, fs))


def gen_obj_img_h5(source_dir, cat_target_dir, sdf_dir, vals, obj):
    img_dir = os.path.join(source_dir, vals, obj, "rendering")
    tar_obj_dir = os.path.join(cat_target_dir, obj)
    os.makedirs(tar_obj_dir, exist_ok=True)
    sdf_fl = os.path.join(sdf_dir, vals, obj, "ori_sample.h5")
    norm_mat = get_norm_matrix(sdf_fl)
    rot_mat = get_rotate_matrix(-np.pi / 2)
    with open(img_dir + "/renderings.txt", 'r') as f:
        lines = f.read().splitlines()
        file_lst = [line.strip() for line in lines]
        params = np.loadtxt(img_dir + "/rendering_metadata.txt")
        param_lst = [params[num, ...].astype(np.float32) for num in range(len(file_lst))]
        for i in range(len(file_lst)):
            h5_file = os.path.join(tar_obj_dir, file_lst[i][:2] + ".h5")
            if os.path.exists(h5_file):
                try:
                    with h5py.File(h5_file, 'r') as h5_f:
                        trans_mat_shape = h5_f["trans_mat"][:].shape[0]
                        print("{} exist! trans first shape is {}".format(h5_file, str(trans_mat_shape)))
                        continue
                except:
                    print("{} exist! but file error".format(h5_file))
            camR, _ = get_img_cam(param_lst[i])
            obj_rot_mat = np.dot(rot90y, camR)
            img_file = os.path.join(img_dir, file_lst[i])
            img_arr = cv2.imread(img_file, cv2.IMREAD_UNCHANGED).astype(np.uint8)
            az, el, distance_ratio = param_lst[i][0], param_lst[i][1], param_lst[i][3]
            K, RT = getBlenderProj(az, el, distance_ratio, img_w=137, img_h=137)
            trans_mat = np.linalg.multi_dot([K, RT, rot_mat, norm_mat])
            trans_mat_right = np.transpose(trans_mat)
            regress_mat = np.transpose(np.linalg.multi_dot([RT, rot_mat, norm_mat]))

            with h5py.File(h5_file, 'w') as f1:
                f1.create_dataset('img_arr', data=img_arr, compression='gzip',
                                  dtype='uint8', compression_opts=4)
                f1.create_dataset('trans_mat', data=trans_mat_right, compression='gzip',
                                  dtype='float32', compression_opts=4)
                f1.create_dataset('K', data=K, compression='gzip',
                                  dtype='float32', compression_opts=4)
                f1.create_dataset('RT', data=RT, compression='gzip',
                                  dtype='float32', compression_opts=4)
                f1.create_dataset('obj_rot_mat', data=obj_rot_mat, compression='gzip',
                                  dtype='float32', compression_opts=4)
                f1.create_dataset('regress_mat', data=regress_mat, compression='gzip',
                                  dtype='float32', compression_opts=4)
                print("write:", h5_file)


def get_img_cam(param):
    cam_mat, cam_pos = camera_info(degree2rad(param))

    return cam_mat, cam_pos

def degree2rad(params):
    params_new = np.zeros_like(params)
    params_new[0] = np.deg2rad(params[0] + 180.0)
    params_new[1] = np.deg2rad(params[1])
    params_new[2] = np.deg2rad(params[2])
    return params_new

def unit(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def camera_info(param):
    az_mat = get_az(param[0])
    el_mat = get_el(param[1])
    inl_mat = get_inl(param[2])
    cam_mat = np.transpose(np.matmul(np.matmul(inl_mat, el_mat), az_mat))
    cam_pos = get_cam_pos(param)
    return cam_mat, cam_pos

def get_cam_pos(param):
    camX = 0
    camY = 0
    camZ = param[3]
    cam_pos = np.array([camX, camY, camZ])
    return -1 * cam_pos

def get_az(az):
    cos = np.cos(az)
    sin = np.sin(az)
    mat = np.asarray([cos, 0.0, sin, 0.0, 1.0, 0.0, -1.0 * sin, 0.0, cos], dtype=np.float32)
    mat = np.reshape(mat, [3, 3])
    return mat

def get_el(el):
    cos = np.cos(el)
    sin = np.sin(el)
    mat = np.asarray([1.0, 0.0, 0.0, 0.0, cos, -1.0 * sin, 0.0, sin, cos], dtype=np.float32)
    mat = np.reshape(mat, [3, 3])
    return mat

def get_inl(inl):
    cos = np.cos(inl)
    sin = np.sin(inl)
    # zeros = np.zeros_like(inl)
    # ones = np.ones_like(inl)
    mat = np.asarray([cos, -1.0 * sin, 0.0, sin, cos, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    mat = np.reshape(mat, [3, 3])
    return mat

def get_img_points(sample_pc, trans_mat_right):
    sample_pc = sample_pc.reshape((-1,3))
    homo_pc = np.concatenate((sample_pc, np.ones((sample_pc.shape[0],1),dtype=np.float32)),axis=-1)
    pc_xyz = np.dot(homo_pc, trans_mat_right).reshape((-1,3))
    print("pc_xyz shape: ", pc_xyz.shape)
    pc_xy = pc_xyz[:,:2] / np.expand_dims(pc_xyz[:,2], axis=1)
    return pc_xy.astype(np.int32)

def get_points(obj_fl):
    sample_pc = np.zeros((0,3), dtype=np.float32)
    mesh_lst = trimesh.load_mesh(obj_fl, process=False)
    if not isinstance(mesh_lst, list):
        mesh_lst = [mesh_lst]
    for mesh in mesh_lst:
        choice = np.random.randint(mesh.vertices.shape[0], size=1000)
        sample_pc = np.concatenate((sample_pc, mesh.vertices[choice,...]), axis=0) #[choice,...]
    # color = [[255,0,0], [0,255,0], [0,0,255], [255, 0, 255]]
    color = 255*np.ones_like(sample_pc,dtype=np.uint8)
    color[:,0] = 0
    color[:,1] = 0
    return sample_pc, np.asarray(color, dtype=np.uint8)

def get_img(img_h5):
    with h5py.File(img_h5, 'r') as h5_f:
        print("read", img_h5)
        trans_mat= h5_f["trans_mat"][:].astype(np.float32)
        regress_mat= h5_f["regress_mat"][:].astype(np.float32)
        obj_rot_mat= h5_f["obj_rot_mat"][:].astype(np.float32)
        K= h5_f["K"][:].astype(np.float32)
        img_arr = h5_f["img_arr"][:][:, :, :3].copy()
        trans_mat2 = np.dot(regress_mat, np.transpose(K))
        print(trans_mat2-trans_mat)
        return img_arr, trans_mat, obj_rot_mat, regress_mat

def test_img_h5(img_h5_fl, num, march_obj_fl):
    img_arr, trans_mat, obj_rot_mat, regress_mat = get_img(img_h5_fl)
    new_pts, colors = get_points(march_obj_fl)
    pc_xy = get_img_points(new_pts, trans_mat)
    for j in range(pc_xy.shape[0]):
        y = int(pc_xy[j, 1])
        x = int(pc_xy[j, 0])
        # print (y,x)
        # print(img_arr[y, x, :])
        # print(tuple([int(x) for x in colors[j]]))
        cv2.circle(img_arr, (x, y), 3, tuple([int(x) for x in colors[j]]), -1)
    rot_pc = np.dot(new_pts, obj_rot_mat)
    np.savetxt(os.path.join("send/", "{}_{}_{}.txt".format("03001627", "184b4797cea77beb5ca1c42bb8ac17a", str(num))), rot_pc)

    print("send/184b4797cea77beb5ca1c42bb8ac17a_{}.png".format(str(num)))
    cv2.imwrite("send/184b4797cea77beb5ca1c42bb8ac17a_{}.png".format(str(num)), img_arr)


if __name__ == "__main__":

    # nohup python -u create_file_lst.py &> create_imgh5.log &

    lst_dir, cats, all_cats, raw_dirs = create_file_lst.get_all_info()
    convert_img2h5(raw_dirs["rendered_dir"],
                   raw_dirs["renderedh5_dir"],
                   lst_dir, cats,
                   raw_dirs["sdf_dir"])


    # test_img_h5("/ssd1/datasets/ShapeNet/ShapeNetRenderingh5_v1/03001627/184b4797cea77beb5ca1c42bb8ac17a/05.h5", 5,
    #             "/ssd1/datasets/ShapeNet/march_cube_objs_v1/03001627/184b4797cea77beb5ca1c42bb8ac17a/isosurf.obj")

    # gen_obj_img_h5("/ssd1/datasets/ShapeNet/ShapeNetRendering",
    #    "send/03001627", "/ssd1/datasets/ShapeNet/SDF_v1/256_expr_1.2_bw_0.1/", "03001627", "184b4797cea77beb5ca1c42bb8ac17a")
    #
    # test_img_h5("send/03001627/184b4797cea77beb5ca1c42bb8ac17a/00.h5",0,
    #                 "/ssd1/datasets/ShapeNet/march_cube_objs_v1/03001627/184b4797cea77beb5ca1c42bb8ac17a/isosurf.obj")
    # test_img_h5("send/03001627/184b4797cea77beb5ca1c42bb8ac17a/01.h5",1,
    #             "/ssd1/datasets/ShapeNet/march_cube_objs_v1/03001627/184b4797cea77beb5ca1c42bb8ac17a/isosurf.obj")
