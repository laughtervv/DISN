import create_file_lst
import h5py
import os
import numpy as np
from joblib import Parallel, delayed
import trimesh
from scipy.interpolate import RegularGridInterpolator
import time

CUR_PATH = os.path.dirname(os.path.realpath(__file__))

def get_sdf_value(sdf_pt, sdf_params_ph, sdf_ph, sdf_res):
    num_point = sdf_pt.shape[0]
    x = np.linspace(sdf_params_ph[0], sdf_params_ph[3], num=sdf_res+1)
    y = np.linspace(sdf_params_ph[1], sdf_params_ph[4], num=sdf_res+1)
    z = np.linspace(sdf_params_ph[2], sdf_params_ph[5], num=sdf_res+1)
    my_interpolating_function = RegularGridInterpolator((z, y, x), sdf_ph)
    sdf_value = my_interpolating_function(sdf_pt)
    print("sdf_value:", sdf_value.shape)
    return np.expand_dims(sdf_value, axis=1)

def get_sdf(sdf_file, sdf_res):
    intsize = 4
    floatsize = 8
    sdf = {
        "param": [],
        "value": []
    }
    with open(sdf_file, "rb") as f:
        try:
            bytes = f.read()
            ress = np.fromstring(bytes[:intsize * 3], dtype=np.int32)
            if -1 * ress[0] != sdf_res or ress[1] != sdf_res or ress[2] != sdf_res:
                raise Exception(sdf_file, "res not consistent with ", str(sdf_res))
            positions = np.fromstring(bytes[intsize * 3:intsize * 3 + floatsize * 6], dtype=np.float64)
            # bottom left corner, x,y,z and top right corner, x, y, z
            sdf["param"] = [positions[0], positions[1], positions[2],
                            positions[3], positions[4], positions[5]]
            sdf["param"] = np.float32(sdf["param"])
            sdf["value"] = np.fromstring(bytes[intsize * 3 + floatsize * 6:], dtype=np.float32)
            sdf["value"] = np.reshape(sdf["value"], (sdf_res + 1, sdf_res + 1, sdf_res + 1))
        finally:
            f.close()
    return sdf

def get_offset_ball(num, bandwidth):
    u = np.random.normal(0, 1, size=(num,1))
    v = np.random.normal(0, 1, size=(num,1))
    w = np.random.normal(0, 1, size=(num,1))
    r = np.random.uniform(0, 1, size=(num,1)) ** (1. / 3) * bandwidth
    norm = np.linalg.norm(np.concatenate([u, v, w], axis=1),axis=1, keepdims=1)
    # print("u.shape",u.shape)
    # print("norm.shape",norm.shape)
    # print("r.shape",r.shape)
    (x, y, z) = r * (u, v, w) / norm
    return np.concatenate([x,y,z],axis=1)

def get_offset_cube(num, bandwidth):
    u = np.random.normal(0, 1, size=(num,1))
    v = np.random.normal(0, 1, size=(num,1))
    w = np.random.normal(0, 1, size=(num,1))
    r = np.random.uniform(0, 1, size=(num,1)) ** (1. / 3) * bandwidth
    norm = np.linalg.norm(np.concatenate([u, v, w], axis=1),axis=1, keepdims=1)
    # print("u.shape",u.shape)
    # print("norm.shape",norm.shape)
    # print("r.shape",r.shape)
    (x, y, z) = r * (u, v, w) / norm
    return np.concatenate([x,y,z],axis=1)

def sample_sdf(cat_id, num_sample, bandwidth, iso_val, sdf_dict, sdf_res, reduce):
    start = time.time()
    params = sdf_dict["param"]
    sdf_values = sdf_dict["value"].flatten()
    # print("np.min(sdf_values), np.mean(sdf_values), np.max(sdf_values)",
    #       np.min(sdf_values), np.mean(sdf_values), np.max(sdf_values))
    x = np.linspace(params[0], params[3], num=sdf_res // reduce + 1).astype(np.float32)
    y = np.linspace(params[1], params[4], num=sdf_res // reduce + 1).astype(np.float32)
    z = np.linspace(params[2], params[5], num=sdf_res // reduce + 1).astype(np.float32)
    z_vals, y_vals, x_vals = np.meshgrid(z, y, x, indexing='ij')
    print("x_vals", x_vals[0, 0, sdf_res // reduce])
    x_original = np.linspace(params[0], params[3], num=sdf_res + 1).astype(np.float32)
    y_original = np.linspace(params[1], params[4], num=sdf_res + 1).astype(np.float32)
    z_original = np.linspace(params[2], params[5], num=sdf_res + 1).astype(np.float32)
    x_ind = np.arange(sdf_res // reduce + 1).astype(np.int32)
    y_ind = np.arange(sdf_res // reduce + 1).astype(np.int32)
    z_ind = np.arange(sdf_res // reduce + 1).astype(np.int32)
    zv, yv, xv = np.meshgrid(z_ind, y_ind, x_ind, indexing='ij')
    choosen_ind = xv * reduce + yv * (sdf_res+1) * reduce + zv * (sdf_res+1)**2 * reduce
    choosen_ind = np.asarray(choosen_ind, dtype=np.int32).reshape(-1)
    vals = sdf_values[choosen_ind]
    # sdf_pt_val = np.stack((x_vals, y_vals, z_vals, vals), axis = -1)
    sdf_pt_val = np.expand_dims(vals, axis= -1 )
    # print("np.min(vals), np.mean(vals), np.max(vals)", np.min(vals), np.mean(vals), np.max(vals))
    print("sdf_pt_val.shape", sdf_pt_val.shape)
    print("sample_sdf: {} s".format(time.time()-start))
    return sdf_pt_val, check_insideout(cat_id, sdf_values, sdf_res, x_original,y_original,z_original)

def check_insideout(cat_id, sdf_val, sdf_res, x, y, z):
    # "chair": "03001627",
    # "bench": "02828884",
    # "cabinet": "02933112",
    # "car": "02958343",
    # "airplane": "02691156",
    # "display": "03211117",
    # "lamp": "03636649",
    # "speaker": "03691459",
    # "rifle": "04090263",
    # "sofa": "04256520",
    # "table": "04379243",
    # "phone": "04401088",
    # "watercraft": "04530566"
    if cat_id in ["02958343", "02691156", "04530566"]:
        x_ind = np.argmin(np.absolute(x))
        y_ind = np.argmin(np.absolute(y))
        z_ind = np.argmin(np.absolute(z))
        all_val = sdf_val.flatten()
        num_val = all_val[x_ind+y_ind*(sdf_res+1)+z_ind*(sdf_res+1)**2]
        return num_val > 0.0
    else:
        return False

def create_h5_sdf_pt(cat_id, h5_file, sdf_file, flag_file, norm_obj_file,
         centroid, m, sdf_res, num_sample, bandwidth, iso_val, max_verts, normalize, reduce=8):
    sdf_dict = get_sdf(sdf_file, sdf_res)
    ori_verts = np.asarray([0.0,0.0,0.0], dtype=np.float32).reshape((1,3))
    # Nx3(x,y,z)
    print("ori_verts", ori_verts.shape)
    samplesdf, is_insideout = sample_sdf(cat_id, num_sample, bandwidth, iso_val, sdf_dict, sdf_res, reduce)  # (N*8)x4 (x,y,z)
    if is_insideout:
        with open(flag_file, "w") as f:
            f.write("mid point sdf val > 0")
        print("insideout !!:", sdf_file)
    else:
        os.remove(flag_file) if os.path.exists(flag_file) else None
    print("samplesdf", samplesdf.shape)
    print("start to write",h5_file)
    norm_params = np.concatenate((centroid, np.asarray([m]).astype(np.float32)))
    f1 = h5py.File(h5_file, 'w')
    f1.create_dataset('pc_sdf_original', data=ori_verts.astype(np.float32), compression='gzip', compression_opts=4)
    f1.create_dataset('pc_sdf_sample', data=samplesdf.astype(np.float32), compression='gzip', compression_opts=4)
    f1.create_dataset('norm_params', data=norm_params, compression='gzip', compression_opts=4)
    f1.create_dataset('sdf_params', data=sdf_dict["param"], compression='gzip', compression_opts=4)
    f1.close()
    print("end writing",h5_file)
    command_str = "rm -rf " + norm_obj_file
    print("command:", command_str)
    os.system(command_str)
    command_str = "rm -rf " + sdf_file
    print("command:", command_str)
    os.system(command_str)

def get_param_from_h5(sdf_h5_file, cat_id, obj):
    h5_f = h5py.File(sdf_h5_file, 'r')
    try:
        if 'norm_params' in h5_f.keys():
            norm_params = h5_f['norm_params'][:]
        else:
            raise Exception(cat_id, obj, "no sdf and sample")
    finally:
        h5_f.close()
    return norm_params[:3], norm_params[3]

def get_normalize_mesh(model_file, norm_sdf_file, cat_id, obj, sdf_sub_dir):

    print("load mesh from ", model_file)
    mesh_list = trimesh.load_mesh(model_file, process=False)
    if not isinstance(mesh_list, list):
        mesh_list = [mesh_list]
    largest_ind = 0
    largest_sur = 0
    for idx, mesh in enumerate(mesh_list):
        area = np.sum(mesh.area_faces)
        if largest_sur < area:
            largest_ind = idx
            largest_sur = area
    mesh = mesh_list[largest_ind]
    centroid, m = get_param_from_h5(norm_sdf_file, cat_id, obj)
    mesh.vertices = (mesh.vertices - centroid) / float(m)
    obj_file = os.path.join(sdf_sub_dir,"pc_norm.obj")
    print("exporting", obj_file)
    trimesh.exchange.export.export_mesh(mesh, obj_file, file_type="obj")
    print("export_mesh", obj_file)
    return obj_file, centroid, m

def create_one_sdf(sdfcommand, res, expand_rate, sdf_file, obj_file, indx, g=0.0):

    command_str = sdfcommand + " " + obj_file + " " + str(res) + " " + str(res) + \
       " " + str(res) + " -s " + " -e " + str(expand_rate) + " -o " + str(indx) + ".dist -m 1"
    if g > 0.0:
        command_str += " -g " + str(g)
    print("command:", command_str)
    os.system(command_str)
    command_str2 = "mv " + str(indx)+".dist " + sdf_file
    print("command:", command_str2)
    os.system(command_str2)


def create_sdf_obj(sdfcommand, marching_cube_command, cat_mesh_dir, cat_norm_sdf_dir, cat_sdf_dir, obj,
       res, iso_val, expand_rate, indx, ish5, normalize, num_sample, bandwidth, max_verts, cat_id, g, reduce):
    obj=obj.rstrip('\r\n')
    sdf_sub_dir = os.path.join(cat_sdf_dir, obj)
    norm_sdf_sub_dir = os.path.join(cat_norm_sdf_dir, obj)
    if not os.path.exists(sdf_sub_dir): os.makedirs(sdf_sub_dir)
    assert os.path.exists(norm_sdf_sub_dir)
    sdf_file = os.path.join(sdf_sub_dir, "isosurf.sdf")
    flag_file = os.path.join(sdf_sub_dir, "isinsideout.txt")
    norm_sdf_file = os.path.join(norm_sdf_sub_dir, "ori_sample.h5")
    h5_file = os.path.join(sdf_sub_dir, "ori_sample.h5")
    if ish5 and os.path.exists(h5_file) and not os.path.exists(flag_file):
        print("skip existed: ", h5_file)
    elif not ish5 and os.path.exists(sdf_file):
        print("skip existed: ", sdf_file)
    else:
        model_file = os.path.join(cat_mesh_dir, obj, "models", "model_normalized.obj")
        # try:
        print("creating", sdf_file)
        if normalize:
            norm_obj_file, centroid, m = get_normalize_mesh(
                model_file, norm_sdf_file, cat_id, obj, sdf_sub_dir)

        create_one_sdf(sdfcommand, res, expand_rate, sdf_file, norm_obj_file, indx, g=g)
        # create_one_cube_obj(marching_cube_command, iso_val, sdf_file, cube_obj_file)
        # change to h5
        if ish5:
            create_h5_sdf_pt(cat_id,h5_file, sdf_file, flag_file, norm_obj_file,
                 centroid, m, res, num_sample, bandwidth, iso_val, max_verts, normalize, reduce=reduce)
        # except:
        #     print("%%%%%%%%%%%%%%%%%%%%%%%% fail to process ", model_file)

def create_one_cube_obj(marching_cube_command, i, sdf_file, cube_obj_file):
    command_str = marching_cube_command + " " + sdf_file + " " + cube_obj_file + " -i " + str(i)
    print("command:", command_str)
    os.system(command_str)
    return cube_obj_file

def create_sdf(sdfcommand, marching_cube_command, LIB_command, num_sample, bandwidth, res, expand_rate,
               all_cats, cats, raw_dirs, lst_dir, iso_val, max_verts, ish5= True, normalize=True, g=0.00,
               param_dir="256_expr_1.2_bw_0.1", reduce=4):
    '''
    Usage: SDFGen <filename> <dx> <padding>
    Where:
        res is number of grids on xyz dimension
        w is narrowband width
        expand_rate is sdf range of max x,y,z
    '''
    print("command:", LIB_command)
    os.system(LIB_command)
    sdf_dir=raw_dirs["sdf_full_dir"]
    if not os.path.exists(sdf_dir): os.makedirs(sdf_dir)
    start=0
    for catnm in cats.keys():
        cat_id = cats[catnm]
        cat_sdf_dir = os.path.join(sdf_dir, cat_id)
        if not os.path.exists(cat_sdf_dir): os.makedirs(cat_sdf_dir)
        cat_mesh_dir = os.path.join(raw_dirs["mesh_dir"], cat_id)
        cat_norm_sdf_dir = os.path.join(raw_dirs["sdf_dir"], cat_id)
        list_obj=[]
        with open(lst_dir+"/"+str(cat_id)+"_test.lst", "r") as f:
            list_obj = f.readlines()
        with open(lst_dir+"/"+str(cat_id)+"_train.lst", "r") as f:
            list_obj += f.readlines()
        # print(list_obj)
        repeat = len(list_obj)
        indx_lst = [i for i in range(start, start+repeat)]
        sdfcommand_lst=[sdfcommand for i in range(repeat)]
        marching_cube_command_lst=[marching_cube_command for i in range(repeat)]
        cat_mesh_dir_lst=[cat_mesh_dir for i in range(repeat)]
        cat_norm_sdf_dir_lst=[cat_norm_sdf_dir for i in range(repeat)]
        cat_sdf_dir_lst=[cat_sdf_dir for i in range(repeat)]
        res_lst=[res for i in range(repeat)]
        expand_rate_lst=[expand_rate for i in range(repeat)]
        normalize_lst=[normalize for i in range(repeat)]
        iso_val_lst=[iso_val for i in range(repeat)]
        ish5_lst=[ish5 for i in range(repeat)]
        num_sample_lst=[num_sample for i in range(repeat)]
        bandwidth_lst=[bandwidth for i in range(repeat)]
        max_verts_lst=[max_verts for i in range(repeat)]
        cat_id_lst=[cat_id for i in range(repeat)]
        g_lst=[g for i in range(repeat)]
        reduce_lst=[reduce for i in range(repeat)]
        with Parallel(n_jobs=5) as parallel:
            parallel(delayed(create_sdf_obj)
            (sdfcommand, marching_cube_command, cat_mesh_dir, cat_norm_sdf_dir, cat_sdf_dir, obj, res,
             iso_val, expand_rate, indx, ish5, norm, num_sample, bandwidth, max_verts,cat_id, g, reduce)
            for sdfcommand, marching_cube_command, cat_mesh_dir, cat_norm_sdf_dir, cat_sdf_dir, obj,
                res, iso_val, expand_rate, indx, ish5, norm, num_sample, bandwidth, max_verts,cat_id, g, reduce_lst
                in zip(sdfcommand_lst,
                marching_cube_command_lst,
                cat_mesh_dir_lst,
                cat_norm_sdf_dir_lst,
                cat_sdf_dir_lst,
                list_obj,
                res_lst, iso_val_lst,
                expand_rate_lst,
                indx_lst, ish5_lst, normalize_lst,num_sample_lst,
                bandwidth_lst, max_verts_lst,cat_id_lst, g_lst, reduce_lst))
        start+=repeat
    print("finish all")

# def test_sdf(sdf_h5_file):
#     h5_f = h5py.File(sdf_h5_file, 'r')
#     red = np.asarray([255.0, 0, 0]).astype(np.float32)
#     blue = np.asarray([0, 0, 255.0]).astype(np.float32)
#     try:
#         if ('pc_sdf_original' in h5_f.keys() and 'pc_sdf_sample' in h5_f.keys()):
#             ori_sdf = h5_f['pc_sdf_original'][:]
#             sample_sdf = np.reshape(h5_f['pc_sdf_sample'][:], (-1, 4))
#             ori_pt, ori_sdf_val = ori_sdf[:, :3], ori_sdf[:, 3]
#             sample_pt, sample_sdf_val = sample_sdf[:, :3], sample_sdf[:, 3]
#             minval, maxval = np.min(ori_sdf_val), np.max(ori_sdf_val)
#             sdf_pt_color = np.zeros([ori_pt.shape[0], 6], dtype=np.float32)
#             sdf_pt_color[:, :3] = ori_pt
#             for i in range(sdf_pt_color.shape[0]):
#                 sdf_pt_color[i, 3:] = red + (blue - red) * (
#                             float(ori_sdf_val[i] - minval) / float(maxval - minval))
#             np.savetxt("./ori.txt", sdf_pt_color)
#
#             sample_pt_color = np.zeros([sample_pt.shape[0], 6], dtype=np.float32)
#             sample_pt_color[:, :3] = sample_pt
#             minval, maxval = np.min(sample_sdf_val), np.max(sample_sdf_val)
#             for i in range(sample_pt_color.shape[0]):
#                 sample_pt_color[i, 3:] = red + (blue - red) * (
#                         float(sample_sdf_val[i] - minval) / float(maxval - minval))
#             np.savetxt("./sample.txt", sample_pt_color)
#     finally:
#         h5_f.close()



if __name__ == "__main__":

    # nohup python -u create_point_sdf_fullgrid.py &> createfull.log &
    lst_dir, cats, all_cats, raw_dirs = create_file_lst.get_all_info()



    #  full set
    create_sdf("/home/xharlie/dev/isosurface/computeDistanceField",
               "/home/xharlie/dev/isosurface/computeMarchingCubes",
               "source /home/xharlie/dev/isosurface/LIB_PATH", 274625, 0.1,
               256, 1.2, all_cats, cats, raw_dirs,
               lst_dir, 0.003, 16384, ish5=True, normalize=True, g=0.00, reduce=4)


    create_sdf("/home/xharlie/dev/isosurface/computeDistanceField",
               "/home/xharlie/dev/isosurface/computeMarchingCubes",
               "source /home/xharlie/dev/isosurface/LIB_PATH", 274625, 0.1,
               256, 1.2, all_cats, cats, raw_dirs,
               lst_dir, 0.003, 16384, ish5=True, normalize=True, g=0.03, reduce=4)

    create_sdf("/home/xharlie/dev/isosurface/computeDistanceField",
               "/home/xharlie/dev/isosurface/computeMarchingCubes",
               "source /home/xharlie/dev/isosurface/LIB_PATH", 274625, 0.1,
               256, 1.2, all_cats, cats, raw_dirs,
               lst_dir, 0.003, 16384, ish5=True, normalize=True, g=0.05, reduce=4)

    create_sdf("/home/xharlie/dev/isosurface/computeDistanceField",
               "/home/xharlie/dev/isosurface/computeMarchingCubes",
               "source /home/xharlie/dev/isosurface/LIB_PATH", 274625, 0.1,
               256, 1.2, all_cats, cats, raw_dirs,
               lst_dir, 0.003, 16384, ish5=True, normalize=True, g=0.06, reduce=4)

    create_sdf("/home/xharlie/dev/isosurface/computeDistanceField",
               "/home/xharlie/dev/isosurface/computeMarchingCubes",
               "source /home/xharlie/dev/isosurface/LIB_PATH", 274625, 0.1,
               256, 1.2, all_cats, cats, raw_dirs,
               lst_dir, 0.003, 16384, ish5=True, normalize=True, g=0.00, reduce=4)
