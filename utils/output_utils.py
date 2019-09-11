import json
import os
import sys
import struct
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.abspath(__file__))) # model
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')

def save_sdf_bin(bin_fn, output_sdf, res):
    f_sdf_bin = open(bin_fn, 'wb')
    
    f_sdf_bin.write(struct.pack('i', -(res-1))) # write an int
    f_sdf_bin.write(struct.pack('i', (res-1))) # write an int
    f_sdf_bin.write(struct.pack('i', (res-1))) # write an int

    positions = [-1.,-1.,-1.,1.,1.,1.]
    positions = struct.pack('d'*len(positions), *positions)
    f_sdf_bin.write(positions)

    sdf = struct.pack('f'*len(output_sdf), *output_sdf)
    f_sdf_bin.write(sdf)
    f_sdf_bin.close()

############################
##    Visualize Results   ##
############################

color_map = json.load(open(os.path.join(BASE_DIR, 'part_color_mapping.json'), 'r'))

def output_bounding_box_withcorners(box_corners, seg, out_file):
    # ##############   0       1       2       3       4       5       6       7
    corner_indexes = [[0, 1, 2], [0, 1, 5], [0, 4, 2], [0, 4, 5], [3, 1, 2], [3, 1, 5], [3, 4, 2], [3, 4, 5]]
    line_indexes = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]
    with open(out_file, 'w') as f:
        l = box_corners.shape[0]
        for i in range(l):
            box = box_corners[i]
            color = color_map[seg[i]]
            for line_index in line_indexes:
                corner0 = box[line_index[0]]
                corner1 = box[line_index[1]]
                print(corner0.shape)
                dist = np.linalg.norm(corner0 - corner1)
                dot_num = int(dist / 0.005)
                delta = (corner1 - corner0) / dot_num
                for idot in range(dot_num):
                    plotdot = corner0 + idot * delta
                    f.write(
                        'v %f %f %f %f %f %f\n' % (plotdot[0], plotdot[1], plotdot[2], color[0], color[1], color[2]))


def output_bounding_box(boxes, seg, out_file):
    # ##############   0       1       2       3       4       5       6       7
    #box:nx8x3
    corner_indexes = [[0, 1, 2], [0, 1, 5], [0, 4, 2], [0, 4, 5], [3, 1, 2], [3, 1, 5], [3, 4, 2], [3, 4, 5]]
    line_indexes = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]
    with open(out_file, 'w') as f:
        l = boxes.shape[0]
        for i in range(l):
            box = boxes[i]
            color = color_map[seg[i]]
            for line_index in line_indexes:
                corner0 = box[corner_indexes[line_index[0]]]
                corner1 = box[corner_indexes[line_index[1]]]
                dist = np.linalg.norm(corner0 - corner1)
                dot_num = int(dist / 0.005)
                delta = (corner1 - corner0) / dot_num
                for idot in range(dot_num):
                    plotdot = corner0 + idot * delta
                    f.write(
                        'v %f %f %f %f %f %f\n' % (plotdot[0], plotdot[1], plotdot[2], color[0], color[1], color[2]))


def output_color_point_cloud(data, seg, out_file):
    with open(out_file, 'w') as f:
        l = len(seg)
        for i in range(l):
            color = color_map[seg[i]]
            f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], color[0], color[1], color[2]))


def output_point_cloud_rgb(data, rgb, out_file):
    with open(out_file, 'w') as f:
        l = len(data)
        for i in range(l):
            f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], rgb[i][0],  rgb[i][1],  rgb[i][2]))


def output_color_point_cloud_red_blue(data, seg, out_file):
    with open(out_file, 'w') as f:
        l = len(seg)
        for i in range(l):
            if seg[i] == 1:
                color = [0, 0, 1]
            elif seg[i] == 0:
                color = [1, 0, 0]
            else:
                color = [0, 0, 0]
            f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], color[0], color[1], color[2]))


##define color heat map
norm = mpl.colors.Normalize(vmin=0, vmax=255)
magma_cmap = plt.cm.get_cmap('magma')
magma_rgb = []
for i in range(0, 255):
       k = mpl.colors.colorConverter.to_rgb(magma_cmap(norm(i)))
       magma_rgb.append(k)


def output_scale_point_cloud(data, scales, out_file):
    with open(out_file, 'w') as f:
        l = len(scales)
        for i in range(l):
            scale = int(scales[i]*254)
            if scale > 254:
                scale = 254
            if scale < 0:
                scale = 0
            color = magma_rgb[scale]
            f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], color[0], color[1], color[2]))
