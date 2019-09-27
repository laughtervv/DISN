import numpy as np
import mcubes

def writeOBJ(vertices, triangles):
    objpath = 'result.obj'
    with open(objpath, 'w') as f:
        for v in vertices:
            f.write('v %.4f %.4f %.4f\n'%(v[0], v[1], v[2]))
        #  If you want to add triangles to .obj file, you can uncomment these lines
        #  In fact vertices information is better.
        #  for p in triangles:
        #     f.write("f")
        #     for i in p:
        #         f.write(" %d"%(i+1))
        #     f.write("\n")

if __name__ == '__main__':
    # result.npy's shape is (1, 16974593, 1)
    path = 'result.npy'
    u = np.load(path).reshape(257,257,257)
    vertices, triangles = mcubes.marching_cubes(u, 0)
    writeOBJ(vertices, triangles)
    print('Done.')
