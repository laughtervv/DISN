import numpy as np
import math
import os
import threading
import queue
import sys
import h5py
import copy
import random
import cv2
FETCH_BATCH_SIZE = 32
BATCH_SIZE = 32
HEIGHT = 192
WIDTH = 256
POINTCLOUDSIZE = 16384
OUTPUTPOINTS = 1024
REEBSIZE = 1024


def get_filelist(lst_dir, maxnverts, minsurbinvox, cats, cats_info, type):
    for cat in cats:
        cat_id = cats_info[cat]
    inputlistfile = os.path.join(lst_dir, cat_id + type + ".lst")
    with open(inputlistfile, 'r') as f:
        lines = f.read().splitlines()
        file_lst = [[cat_id, line.strip()] for line in lines]
    return file_lst

class Pt_sdf_img(threading.Thread):
    def __init__(self, FLAGS, listinfo=None, info=None, qsize=64, cats_limit=None):
        super(Pt_sdf_img, self).__init__()
        self.queue = queue.Queue(qsize)
        self.stopped = False
        self.bno = 0

        self.listinfo = listinfo
        self.num_points = FLAGS.num_points
        self.gen_num_pt = FLAGS.num_sample_points
        self.batch_size = FLAGS.batch_size
        self.img_dir = info['rendered_dir']
        self.img_dir_v2 = info['rendered_dir_v2']
        self.sdf_dir = info['sdf_dir']
        self.order = list(range(len(listinfo)))
        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 60000
        self.data_num = len(self.listinfo)
        self.FLAGS = FLAGS
        self.num_batches = int(self.data_num / self.FLAGS.batch_size)
        self.fixaxismat = np.array([[-1., 0., 0.],[0., 0., 1],[ 0.,  -1., 0.]], dtype=np.float32)

        self.CAM_ROT = np.array([[1.910685676922942e-15, 4.371138828673793e-08, 1.0],
                                  [1.0, -4.371138828673793e-08, -0.0],
                                  [4.371138828673793e-08, 1.0, -4.371138828673793e-08]], dtype=np.float32)

        self.rot90z = np.array([[0, 1, 0],
                                [-1, 0,0],
                                [ 0,0,1]], dtype=np.float32)
        self.rot90y = np.array([[0, 0, -1],
                                [ 0,1, 0],
                                [ 1,0,0]], dtype=np.float32)
        self.tmp = np.array([0,0,0], dtype = np.float32)
        self.count = 0
        self.num_batches = int(self.data_num / self.FLAGS.batch_size)
        self.cats_limit, self.epoch_amount = self.set_cat_limit(cats_limit)
        self.data_order = list(range(len(listinfo)))

    def set_cat_limit(self, cats_limit):
        epoch_amount = 0
        for cat, amount in cats_limit.items():
            cats_limit[cat] = min(self.FLAGS.cat_limit, amount)
            epoch_amount += cats_limit[cat]
        print("epoch_amount ", epoch_amount)
        print("cats_limit ", cats_limit)
        return cats_limit, epoch_amount

    def get_img_dir(self, cat_id, obj):
        img_dir = os.path.join(self.img_dir, cat_id, obj)
        img_dir_v2 = os.path.join(self.img_dir_v2, cat_id, obj, 'rendering')
        return img_dir, img_dir_v2

    # def get_h5_filenm(self, cat_id, obj):
    #     return os.path.join(self.mesh_dir, cat_id, obj + ".h5")

    def get_sdf_h5_filenm(self, cat_id, obj):
        return os.path.join(self.sdf_dir, cat_id, obj, "ori_sample.h5")

    def pc_normalize(self, pc, centroid=None):

        """ pc: NxC, return NxC """
        l = pc.shape[0]

        if centroid is None:
            centroid = np.mean(pc, axis=0)

        pc = pc - centroid
        # m = np.max(pc, axis=0)
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))

        pc = pc / m

        return pc, centroid, m

    def __len__(self):
        return self.epoch_amount

    def memory(self):
        """
        Get node total memory and memory usage
        """
        with open('/proc/meminfo', 'r') as mem:
            ret = {}
            tmp = 0
            for i in mem:
                sline = i.split()
                if str(sline[0]) == 'MemTotal:':
                    ret['total'] = int(sline[1])
                elif str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                    tmp += int(sline[1])
            ret['free'] = tmp
            ret['used'] = int(ret['total']) - int(ret['free'])
        return ret

    def getitem(self, index):
        cat_id, obj, num = self.listinfo[index]
        sdf_file = self.get_sdf_h5_filenm(cat_id, obj)
        ori_pt, ori_sdf_val, sample_pt, sample_sdf_val, norm_params, sdf_params\
            = self.get_sdf_h5(sdf_file, cat_id, obj)
        img_dir, img_dir_v2 = self.get_img_dir(cat_id, obj)
        return ori_pt, ori_sdf_val, sample_pt, sample_sdf_val, norm_params,\
               sdf_params, img_dir, img_dir_v2, cat_id, obj, num

    def get_sdf_h5(self, sdf_h5_file, cat_id, obj):
        h5_f = h5py.File(sdf_h5_file, 'r')
        try:
            if ('pc_sdf_original' in h5_f.keys()
                    and 'pc_sdf_sample' in h5_f.keys()
                    and 'norm_params' in h5_f.keys()):
                ori_sdf = h5_f['pc_sdf_original'][:].astype(np.float32)
                # sample_sdf = np.reshape(h5_f['pc_sdf_sample'][:],(ori_sdf.shape[0], -1 ,4)).astype(np.float32)
                sample_sdf = h5_f['pc_sdf_sample'][:].astype(np.float32)
                ori_pt = ori_sdf[:,:3]#, ori_sdf[:,3]
                ori_sdf_val = None
                sample_pt, sample_sdf_val = sample_sdf[:,:3], sample_sdf[:,3]
                norm_params = h5_f['norm_params'][:]
                sdf_params = h5_f['sdf_params'][:]
            else:
                raise Exception(cat_id, obj, "no sdf and sample")
        finally:
            h5_f.close()
        return ori_pt, ori_sdf_val, sample_pt, sample_sdf_val, norm_params, sdf_params

    def get_img_cam(self, img_dir, num):
        params = np.loadtxt(img_dir + "/rendering_metadata.txt")
        # img_file = os.path.join(img_dir, file_lst[num])
        # azimuth, elevation, in-plane rotation, distance, the field of view.
        param = params[num, :].astype(np.float32)
        cam_mat, cam_pos = self.camera_info(self.degree2rad(param))
        # img_arr = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)[:,:,:3].astype(np.float32) / 255.
        return cam_mat, cam_pos

    def get_img(self, img_dir, num):
        img_h5 = os.path.join(img_dir, "%02d.h5"%num)
        cam_mat, cam_pos, trans_mat =None, None, None
        with h5py.File(img_h5, 'r') as h5_f:
            if self.FLAGS.img_feat:
                trans_mat = h5_f["trans_mat"][:].astype(np.float32)
                RT = h5_f["regress_mat"][:].astype(np.float32)
                K = h5_f["K"][:].astype(np.float32)
                # self.tmp += RT[3,:]
                # self.count += 1
                # print (self.tmp / self.count)
                # np.linalg.multi_dot([RT, rot_mat, norm_mat])
                # f1.create_dataset('K', data=K, compression='gzip',
                #                     dtype='float32', compression_opts=4)
                # f1.create_dataset('RT', data=RT, compression='gzip',
                #                 dtype='float32', compression_opts=4)
            else:
                cam_mat, cam_pos = h5_f["cam_mat"][:].astype(np.float32), h5_f["cam_pos"][:].astype(np.float32)
            img_arr = h5_f["img_arr"][:][:,:,:4].astype(np.float32) / 255.
            return img_arr, cam_mat, cam_pos, trans_mat, RT

    def degree2rad(self, params):
        params[0] = np.deg2rad(params[0] + 180.0)
        params[1] = np.deg2rad(params[1])
        params[2] = np.deg2rad(params[2])
        return params

    def unit(self, v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def camera_info(self, param):
        az_mat = self.get_az(param[0])
        el_mat = self.get_el(param[1])
        inl_mat = self.get_inl(param[2])
        cam_mat = np.transpose(np.matmul(np.matmul(inl_mat, el_mat), az_mat))
        cam_pos = self.get_cam_pos(param)
        return cam_mat, cam_pos

    def get_cam_pos(self, param):
        camX = 0
        camY = 0
        camZ = param[3]
        cam_pos = np.array([camX, camY, camZ])
        return -1 * cam_pos

    def get_az(self, az):
        cos = np.cos(az)
        sin = np.sin(az)
        mat = np.asarray([cos, 0.0, sin, 0.0, 1.0, 0.0, -1.0*sin, 0.0, cos], dtype=np.float32)
        mat = np.reshape(mat, [3,3])
        return mat
    #
    def get_el(self, el):
        cos = np.cos(el)
        sin = np.sin(el)
        mat = np.asarray([1.0, 0.0, 0.0, 0.0, cos, -1.0*sin, 0.0, sin, cos], dtype=np.float32)
        mat = np.reshape(mat, [3,3])
        return mat
    #
    def get_inl(self, inl):
        cos = np.cos(inl)
        sin = np.sin(inl)
        # zeros = np.zeros_like(inl)
        # ones = np.ones_like(inl)
        mat = np.asarray([cos, -1.0*sin, 0.0, sin, cos, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        mat = np.reshape(mat, [3,3])
        return mat

    def get_batch(self, index, istrain = True):
        if index + self.batch_size > self.epoch_amount:
            index = index + self.batch_size - self.epoch_amount
        batch_pc = np.zeros((self.batch_size, self.num_points, 3)).astype(np.float32)
        batch_sdf_pt = np.zeros((self.batch_size, self.gen_num_pt, 3)).astype(np.float32)
        batch_sdf_pt_rot = np.zeros((self.batch_size, self.gen_num_pt, 3)).astype(np.float32)
        batch_sdf_val = np.zeros((self.batch_size, self.gen_num_pt, 1)).astype(np.float32)
        batch_norm_params = np.zeros((self.batch_size, 4)).astype(np.float32)
        batch_sdf_params = np.zeros((self.batch_size, 6)).astype(np.float32)
        batch_img = np.zeros((self.batch_size, self.FLAGS.img_h, self.FLAGS.img_w, 4), dtype=np.float32)
        batch_img_mat = np.zeros((self.batch_size, 3, 3), dtype=np.float32)
        batch_img_pos = np.zeros((self.batch_size, 3), dtype=np.float32)
        batch_trans_mat = np.zeros((self.batch_size, 4, 3), dtype=np.float32)
        batch_RT_mat = np.zeros((self.batch_size, 4, 3), dtype=np.float32)
        batch_shifts = np.zeros((self.batch_size, 2), dtype=np.float32)
        batch_cat_id = []
        batch_obj_nm = []
        batch_view_id = []
        cnt = 0
        for i in range(index, index + self.batch_size):
            single_obj = self.getitem(self.order[i])

            if single_obj == None:
                raise Exception("single mesh is None!")
            ori_pt, ori_sdf_val, sample_pt, sample_sdf_val, norm_params, sdf_params, img_dir, img_dir_v2, cat_id, obj, num = single_obj
            img, cam_mat, cam_pos, trans_mat, RT = self.get_img(img_dir, num)
            if self.FLAGS.shift:
                alpha_pixels = np.argwhere(img[:, :, 3] > 0.00)
                y_shift_top = min(alpha_pixels[:, 0])
                y_shift_down = self.FLAGS.img_h - max(alpha_pixels[:, 0])

                x_shift_left = min(alpha_pixels[:,1])
                x_shift_right = self.FLAGS.img_w - max(alpha_pixels[:,1])
                y_shift = random.randrange(-y_shift_top * 0.7, y_shift_down * 0.7, 1)
                x_shift = random.randrange(-x_shift_left * 0.7, x_shift_right * 0.7, 1)
                # print(y_shift_top,y_shift_down,y_shift)
                # print(x_shift_left,x_shift_right,x_shift)
                img_new = np.zeros((self.FLAGS.img_h, self.FLAGS.img_w, 4), dtype=np.float32)
                new_alpha_pixels = np.stack((alpha_pixels[:,0] + y_shift, alpha_pixels[:,1] + x_shift), axis=1)
                img_new[new_alpha_pixels[:, 0], new_alpha_pixels[:, 1], :] = img[alpha_pixels[:, 0], alpha_pixels[:, 1], :]
                # cv2.imwrite(os.path.join("/home/xharlie/dev/DISN_codebase/pnt_vis", 'org_%s_%d.png' % (img_dir.split("/")[-1],num)), np.asarray(img*255).astype(np.uint8))
                # cv2.imwrite(os.path.join("/home/xharlie/dev/DISN_codebase/pnt_vis", 'shifted_%s_%d.png' % (img_dir.split("/")[-1],num)), np.asarray(img_new*255).astype(np.uint8))
                # print(os.path.join("/home/xharlie/dev/DISN_codebase/pnt_vis", 'org_%s_%d.png' % (img_dir.split("/")[-1],num)))
                # print(os.path.join("/home/xharlie/dev/DISN_codebase/pnt_vis", 'shifted_%s_%d.png' % (img_dir.split("/")[-1],num)))
            if ori_pt is not None:
                cf_ref_choice = np.random.randint(ori_pt.shape[0], size=self.num_points)
                batch_pc[cnt, :, :] = ori_pt[cf_ref_choice, :]
                # if self.FLAGS.threedcnn:
                #     batch_sdf_pt[cnt, ...] = sample_pt
                #     batch_sdf_val[cnt, :, 0] = sample_sdf_val
                # else:
                choice = np.random.randint(sample_pt.shape[0], size=self.gen_num_pt)
                if self.FLAGS.rotation:
                    batch_sdf_pt[cnt, ...] = sample_pt[choice, :]
                    # R = np.dot(self.CAM_ROT, RT[:3,:3].T).T

                    pc_rot = batch_sdf_pt[cnt, ...].copy()
                    # pc_rot[:,0] = -pc_rot[:,2] 
                    # pc_rot[:,1] = -pc_rot[:,0] 
                    # pc_rot[:,2] = pc_rot[:,1] 
                    R, _ = self.get_img_cam(img_dir_v2, num)
                    R = np.dot(self.rot90y, R)
                    # R = self.rot90y
                    batch_sdf_pt_rot[cnt, ...] = np.dot(batch_sdf_pt[cnt, ...], R)
                else:
                    batch_sdf_pt[cnt, ...] = sample_pt[choice, :]
                batch_sdf_val[cnt, :, 0] = sample_sdf_val[choice]
                batch_norm_params[cnt, ...] = norm_params
                batch_sdf_params[cnt, ...] = sdf_params
            else:
                raise Exception("no verts or binvox")
            # img, cam_mat, cam_pos = self.get_img_old(img_dir, num, img_file_lst)
            if self.FLAGS.shift:
                batch_img[cnt, ...] = img_new.astype(np.float32)
                batch_shifts[cnt, ...] = np.asarray([x_shift, y_shift], dtype=np.float32) * 2 / (self.FLAGS.img_h)
            else:
                batch_img[cnt, ...] = img.astype(np.float32)
            # batch_img[cnt, ...] = self.normalize_color(img.astype(np.float32))
            # batch_img_mat[cnt, ...] = cam_mat
            # batch_img_pos[cnt, ...] = cam_pos
            batch_trans_mat[cnt, ...] = trans_mat
            batch_RT_mat[cnt, ...] = RT
            batch_cat_id.append(cat_id)
            batch_obj_nm.append(obj)
            batch_view_id.append(num)
            cnt += 1
        batch_data = {}
        batch_data['pc'] = batch_pc
        batch_data['sdf_pt'] = batch_sdf_pt
        batch_data['sdf_pt_rot'] = batch_sdf_pt_rot
        batch_data['sdf_val'] = batch_sdf_val
        batch_data['norm_params'] = batch_norm_params
        batch_data['sdf_params'] = batch_sdf_params
        batch_data['img'] = batch_img
        # batch_data['img_mat'] = batch_img_mat
        # batch_data['img_pos'] = batch_img_pos
        batch_data['trans_mat'] = batch_trans_mat
        batch_data['RT'] = batch_RT_mat
        batch_data['cat_id'] = batch_cat_id
        batch_data['obj_nm'] = batch_obj_nm
        batch_data['view_id'] = batch_view_id
        batch_data['shifts'] = batch_shifts
        return batch_data

    def refill_data_order(self):
        temp_order = copy.deepcopy(self.data_order)
        cats_quota = {key: value for key, value in self.cats_limit.items()}
        np.random.shuffle(temp_order)
        pointer = 0
        epoch_order=[]
        while len(epoch_order) < self.epoch_amount:
            cat_id, _, _ = self.listinfo[temp_order[pointer]]
            if cats_quota[cat_id] > 0:
                epoch_order.append(temp_order[pointer])
                cats_quota[cat_id]-=1
            pointer+=1
        return epoch_order

    def work(self, epoch, index):
        if index == 0:
            self.order = self.refill_data_order()
            print("data order reordered!")
        return self.get_batch(index)

    def run(self):
        while (self.bno // (self.num_batches* self.batch_size)) < self.FLAGS.max_epoch and not self.stopped:
            self.queue.put(self.work(self.bno // (self.num_batches* self.batch_size),
                                     self.bno % (self.num_batches * self.batch_size)))
            self.bno += self.batch_size

    def fetch(self):
        if self.stopped:
            return None
        # else:
        #     print("queue length", self.queue.qsize())
        return self.queue.get()

    def shutdown(self):
        self.stopped = True
        while not self.queue.empty():
            self.queue.get()

if __name__ == '__main__':

    sys.path.append('../preprocessing/')
    import create_file_lst as create

    data = Pt_sdf_img(res=256, expr=1.5,
                      listinfo=[["03001627", "ff3581996365bdddc3bd24f986301745"],
                                ["03001627", "ff3581996365bdddc3bd24f986301745"]],
                      info=create.get_all_info(), maxnverts=6000, maxntris=50000,
                      minsurbinvox=4096, num_points=2048, batch_size=2, normalize=False, norm_color=True)
    batch1 = data.get_batch(0)
    print(batch1.keys())
    print(batch1["verts"].shape)
    print(batch1["nverts"])
    print(batch1["tris"].shape)
    print(batch1["ntris"])
    print(batch1["surfacebinvoxpc"].shape)
    print(batch1["sdf"].shape)
    print(batch1["sdf_params"])
    print(batch1["img"].shape, batch1["img"][0, 64, 64, :])
    print(batch1["img_cam"])

    # (2048, 3)
    cloud1 = batch1["surfacebinvoxpc"][0, ...]
    trans1 = batch1["img_cam"][0, ...]
    az1 = float(trans1[0] + 180) * math.pi / 180.0
    el1 = float(trans1[1]) * math.pi / 180.0
    in1 = float(trans1[2]) * math.pi / 180.0
    transmatrix_az1 = [[math.cos(az1), 0, math.sin(az1)],
                       [0, 1, 0],
                       [-math.sin(az1), 0, math.cos(az1)]]
    transmatrix_az1 = np.asarray(transmatrix_az1).astype(np.float32)
    transmatrix_el1 = [[1, 0, 0],
                       [0, math.cos(el1), -math.sin(el1)],
                       [0, math.sin(el1), math.cos(el1)]]
    transmatrix_el1 = np.asarray(transmatrix_el1).astype(np.float32)
    transmatrix_in1 = [[math.cos(in1), -math.sin(in1), 0],
                       [math.sin(in1), math.cos(in1), 0],
                       [0, 0, 1]]
    transmatrix_in1 = np.asarray(transmatrix_in1).astype(np.float32)

    trans = np.matmul(np.matmul(transmatrix_in1, transmatrix_el1), transmatrix_az1)
    translate1 = np.tile(np.expand_dims(np.asarray([-trans1[2], 0, 0]).astype(np.float32), axis=0), (2048, 1))
    points = np.matmul(cloud1, trans.T)
    np.savetxt("ff_rotate.xyz", points)
    np.savetxt("ff.xyz", cloud1)
    # def get_img(img_dir, num):
    #     img_h5 = os.path.join(img_dir, "%02d.h5"%num)
    #     cam_mat, cam_pos, trans_mat =None, None, None
    #     with h5py.File(img_h5, 'r') as h5_f:
    #         trans_mat = h5_f["trans_mat"][:].astype(np.float32)
    #         RT = h5_f["regress_mat"][:].astype(np.float32)
    #         K = h5_f["K"][:].astype(np.float32)
    #         img_arr = h5_f["img_arr"][:][:,:,:4].astype(np.float32) / 255.
    #         return img_arr, cam_mat, cam_pos, trans_mat, RT
    #
    #
    # img, cam_mat, cam_pos, trans_mat, RT = get_img("/ssd1/datasets/ShapeNet/ShapeNetRenderingh5_v1/03636649/7fa0f8d0da975ea0f323a65d99f15033", 8)
    # alpha_pixels = np.argwhere(img[:, :, 3] > 0.00)
    # y_shift_top = min(alpha_pixels[:, 0])
    # y_shift_down = 137 - max(alpha_pixels[:, 0])
    #
    # x_shift_left = min(alpha_pixels[:, 1])
    # x_shift_right = 137 - max(alpha_pixels[:, 1])
    # y_shift = random.randrange(-y_shift_top, y_shift_down, 1)
    # x_shift = random.randrange(-x_shift_left, x_shift_right, 1)
    # print(y_shift_top, y_shift_down, y_shift)
    # print(x_shift_left, x_shift_right, x_shift)
    # img_new = np.zeros((137,137, 4), dtype=np.float32)
    # new_alpha_pixels = np.stack((alpha_pixels[:, 0] + y_shift, alpha_pixels[:, 1] + x_shift), axis=1)
    # print(new_alpha_pixels.shape, alpha_pixels.shape, new_alpha_pixels - alpha_pixels)
    # # for i in range(new_alpha_pixels.shape[0]):
    # #     img_new[new_alpha_pixels[i,0],new_alpha_pixels[i,1],:] = img[alpha_pixels[i,0],alpha_pixels[i,1],:]
    # img_new[new_alpha_pixels[:, 0], new_alpha_pixels[:, 1], :] = img[alpha_pixels[:, 0], alpha_pixels[:, 1], :]
    # cv2.imwrite(os.path.join("/home/xharlie/dev/DISN_codebase/pnt_vis", 'org_%s_%d.png' % ("7fa0f8d0da975ea0f323a65d99f15033",8)), np.asarray(img*255).astype(np.uint8))
    # cv2.imwrite(os.path.join("/home/xharlie/dev/DISN_codebase/pnt_vis", 'shifted_%s_%d.png' % ("7fa0f8d0da975ea0f323a65d99f15033",8)), np.asarray(img_new*255).astype(np.uint8))
    # print(os.path.join("/home/xharlie/dev/DISN_codebase/pnt_vis", 'org_%s_%d.png' % ("7fa0f8d0da975ea0f323a65d99f15033",8)))
    # print(os.path.join("/home/xharlie/dev/DISN_codebase/pnt_vis", 'shifted_%s_%d.png' % ("7fa0f8d0da975ea0f323a65d99f15033",8)))