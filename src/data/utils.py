import numpy as np
import h5py


#read sdf files produced by SDFGen
def read_sdf_file_as_3d_array(name):
    fp = open(name, 'rb')
    line = fp.readline().strip()
    if not line.startswith(b'#sdf'):
        raise IOError('Not a sdf file')
    dims = list(map(int, fp.readline().strip().split(b' ')[1:]))
    line = fp.readline()
    data = np.frombuffer(fp.read(), dtype=np.float32)
    data = data.reshape(dims)
    fp.close()
    return data

#a simplified reader for binvox files
#mostly copied from binvox_rw.py
#https://github.com/dimatura/binvox-rw-py
def read_binvox_file_as_3d_array(name,fix_coords=True):
    fp = open(name, 'rb')
    line = fp.readline().strip()
    if not line.startswith(b'#binvox'):
        raise IOError('Not a binvox file')
    dims = list(map(int, fp.readline().strip().split(b' ')[1:]))
    translate = list(map(float, fp.readline().strip().split(b' ')[1:]))
    scale = list(map(float, fp.readline().strip().split(b' ')[1:]))[0]
    line = fp.readline()
    raw_data = np.frombuffer(fp.read(), dtype=np.uint8)
    # if just using reshape() on the raw data:
    # indexing the array as array[i,j,k], the indices map into the
    # coords as:
    # i -> x
    # j -> z
    # k -> y
    # if fix_coords is true, then data is rearranged so that
    # mapping is
    # i -> x
    # j -> y
    # k -> z
    values, counts = raw_data[::2], raw_data[1::2]
    data = np.repeat(values, counts).astype(np.bool)
    data = data.reshape(dims)
    if fix_coords:
        data = np.transpose(data, (0, 2, 1))
    data = np.ascontiguousarray(data, np.uint8)
    fp.close()
    return data


def read_data_input_only(hdf5_dir,grid_size,input_type,out_bool,out_float,is_undc):
    hdf5_file = h5py.File(hdf5_dir, 'r')
    if out_bool:
        if is_undc:
            LOD_gt_int = np.zeros([grid_size+1,grid_size+1,grid_size+1,3],np.int32)
        else:
            LOD_gt_int = np.zeros([grid_size+1,grid_size+1,grid_size+1,1],np.int32)
    else:
        LOD_gt_int = None
    if out_float:
        LOD_gt_float = np.zeros([grid_size+1,grid_size+1,grid_size+1,3],np.float32)
    else:
        LOD_gt_float = None
    if input_type=="sdf" or input_type=="udf":
        LOD_input = hdf5_file[str(grid_size)+"_sdf"][:]
        LOD_input = LOD_input*grid_size #denormalize
    elif input_type=="voxel":
        LOD_input = hdf5_file[str(grid_size)+"_voxel"][:]
    elif input_type=="pointcloud" or input_type=="noisypc":
        LOD_input = hdf5_file["pointcloud"][:].astype(np.float32)
        LOD_input = (LOD_input+0.5)*grid_size #denormalize
    hdf5_file.close()
    return LOD_gt_int, LOD_gt_float, LOD_input


def read_data(hdf5_dir, 
                grid_size,
                input_type, 
                out_bool, 
                out_float, 
                is_undc):
    hdf5_file = h5py.File(hdf5_dir, 'r')
    
    if out_bool:
        LOD_gt_int = hdf5_file[str(grid_size)+"_int"][:].astype(np.int32) # 这个 LOD 是个啥？
    else:
        LOD_gt_int = None
    
    if out_float:
        LOD_gt_float = hdf5_file[str(grid_size)+"_float"][:].astype(np.float32)
    else:
        LOD_gt_float = None
    
    if input_type=="pointcloud" or input_type=="noisypc":
        LOD_input = hdf5_file["pointcloud"][:].astype(np.float32)
        LOD_input = (LOD_input+0.5)*grid_size #denormalize
    hdf5_file.close()
    return LOD_gt_int, LOD_gt_float, LOD_input


def read_and_augment_data_undc(hdf5_dir, 
                                grid_size,
                                input_type,
                                out_bool,
                                out_float,
                                aug_permutation=True,
                                aug_reversal=True,
                                aug_inversion=True):
    grid_size_1 = grid_size + 1 # 这里为什么要+1呢？

    #read input hdf5
    LOD_gt_int, LOD_gt_float, LOD_input = read_data(hdf5_dir, grid_size, input_type, out_bool, out_float, is_undc=True)

    new_grid_dict = {} #store grid
    new_pc_dict = {} #store pointcloud

    if out_bool:
        new_grid_dict['int_edge_x_'] = LOD_gt_int[:-1, :  , :  , 0]
        new_grid_dict['int_edge_y_'] = LOD_gt_int[:  , :-1, :  , 1]
        new_grid_dict['int_edge_z_'] = LOD_gt_int[:  , :  , :-1, 2]

    if out_float:
        new_grid_dict['float_center_x_'] = LOD_gt_float[:-1,:-1,:-1,0]
        new_grid_dict['float_center_y_'] = LOD_gt_float[:-1,:-1,:-1,1]
        new_grid_dict['float_center_z_'] = LOD_gt_float[:-1,:-1,:-1,2]

    if input_type=="pointcloud" or input_type=="noisypc":
        new_pc_dict['input_x_'] = LOD_input[:,0]
        new_pc_dict['input_y_'] = LOD_input[:,1]
        new_pc_dict['input_z_'] = LOD_input[:,2]


    #augment data 接下来一大段都是为了对数据进行增强，包括旋转啦之类的
    permutation_list = [ [0,1,2],
                         [0,2,1], 
                         [1,0,2], 
                         [1,2,0], 
                         [2,0,1], 
                         [2,1,0] ]

    reversal_list = [ [0,0,0],
                      [0,0,1],
                      [0,1,0],
                      [0,1,1], 
                      [1,0,0],
                      [1,0,1],
                      [1,1,0],
                      [1,1,1] ]

    if aug_permutation:
        permutation = permutation_list[np.random.randint(len(permutation_list))]
    else:
        permutation = permutation_list[0]
    
    if aug_reversal:
        reversal = reversal_list[np.random.randint(len(reversal_list))]
    else:
        reversal = reversal_list[0]
    
    if aug_inversion:
        inversion_flag = np.random.randint(2)
    else:
        inversion_flag = 0


    if reversal[0]:
        for k in new_grid_dict: #reverse
            new_grid_dict[k] = new_grid_dict[k][::-1,:,:]
        if out_float:
            k = 'float_center_x_'
            mask = (new_grid_dict[k]>=0)
            new_grid_dict[k] = new_grid_dict[k]*(1-mask)+(1-new_grid_dict[k])*mask
        if input_type=="pointcloud" or input_type=="noisypc": 
            k = 'input_x_'
            new_pc_dict[k] = grid_size-new_pc_dict[k]
    if reversal[1]:
        for k in new_grid_dict: #reverse
            new_grid_dict[k] = new_grid_dict[k][:,::-1,:]
        if out_float:
            k = 'float_center_y_'
            mask = (new_grid_dict[k]>=0)
            new_grid_dict[k] = new_grid_dict[k]*(1-mask)+(1-new_grid_dict[k])*mask
        if input_type=="pointcloud" or input_type=="noisypc": 
            k = 'input_y_'
            new_pc_dict[k] = grid_size-new_pc_dict[k]
    if reversal[2]:
        for k in new_grid_dict: #reverse
            new_grid_dict[k] = new_grid_dict[k][:,:,::-1]
        if out_float:
            k = 'float_center_z_'
            mask = (new_grid_dict[k]>=0)
            new_grid_dict[k] = new_grid_dict[k]*(1-mask)+(1-new_grid_dict[k])*mask
        if input_type=="pointcloud" or input_type=="noisypc": 
            k = 'input_z_'
            new_pc_dict[k] = grid_size-new_pc_dict[k]


    if permutation == [0,1,2]:
        pass
    else:
        for k in new_grid_dict: #transpose
            new_grid_dict[k] = np.transpose(new_grid_dict[k], permutation)

        olddict = new_grid_dict
        new_grid_dict = {}
        oldpcdict = new_pc_dict
        new_pc_dict = {}
        if input_type=="sdf" or input_type=="udf":
            new_grid_dict['input_sdf'] = olddict['input_sdf']
        elif input_type=="voxel":
            new_grid_dict['input_voxel'] = olddict['input_voxel']

        if permutation == [0,2,1]:
            if out_bool: 
                new_grid_dict['int_edge_x_'] = olddict['int_edge_x_']
                new_grid_dict['int_edge_y_'] = olddict['int_edge_z_']
                new_grid_dict['int_edge_z_'] = olddict['int_edge_y_']
            if out_float: 
                new_grid_dict['float_center_x_'] = olddict['float_center_x_']
                new_grid_dict['float_center_y_'] = olddict['float_center_z_']
                new_grid_dict['float_center_z_'] = olddict['float_center_y_']
            if input_type=="pointcloud" or input_type=="noisypc":
                new_pc_dict['input_x_'] = oldpcdict['input_x_']
                new_pc_dict['input_y_'] = oldpcdict['input_z_']
                new_pc_dict['input_z_'] = oldpcdict['input_y_']
        elif permutation == [1,0,2]:
            if out_bool: 
                new_grid_dict['int_edge_x_'] = olddict['int_edge_y_']
                new_grid_dict['int_edge_y_'] = olddict['int_edge_x_']
                new_grid_dict['int_edge_z_'] = olddict['int_edge_z_']
            if out_float: 
                new_grid_dict['float_center_x_'] = olddict['float_center_y_']
                new_grid_dict['float_center_y_'] = olddict['float_center_x_']
                new_grid_dict['float_center_z_'] = olddict['float_center_z_']
            if input_type=="pointcloud" or input_type=="noisypc":
                new_pc_dict['input_x_'] = oldpcdict['input_y_']
                new_pc_dict['input_y_'] = oldpcdict['input_x_']
                new_pc_dict['input_z_'] = oldpcdict['input_z_']
        elif permutation == [2,1,0]:
            if out_bool: 
                new_grid_dict['int_edge_x_'] = olddict['int_edge_z_']
                new_grid_dict['int_edge_y_'] = olddict['int_edge_y_']
                new_grid_dict['int_edge_z_'] = olddict['int_edge_x_']
            if out_float: 
                new_grid_dict['float_center_x_'] = olddict['float_center_z_']
                new_grid_dict['float_center_y_'] = olddict['float_center_y_']
                new_grid_dict['float_center_z_'] = olddict['float_center_x_']
            if input_type=="pointcloud" or input_type=="noisypc":
                new_pc_dict['input_x_'] = oldpcdict['input_z_']
                new_pc_dict['input_y_'] = oldpcdict['input_y_']
                new_pc_dict['input_z_'] = oldpcdict['input_x_']
        elif permutation == [1,2,0]:
            if out_bool: 
                new_grid_dict['int_edge_x_'] = olddict['int_edge_y_']
                new_grid_dict['int_edge_y_'] = olddict['int_edge_z_']
                new_grid_dict['int_edge_z_'] = olddict['int_edge_x_']
            if out_float: 
                new_grid_dict['float_center_x_'] = olddict['float_center_y_']
                new_grid_dict['float_center_y_'] = olddict['float_center_z_']
                new_grid_dict['float_center_z_'] = olddict['float_center_x_']
            if input_type=="pointcloud" or input_type=="noisypc":
                new_pc_dict['input_x_'] = oldpcdict['input_y_']
                new_pc_dict['input_y_'] = oldpcdict['input_z_']
                new_pc_dict['input_z_'] = oldpcdict['input_x_']
        elif permutation == [2,0,1]:
            if out_bool: 
                new_grid_dict['int_edge_x_'] = olddict['int_edge_z_']
                new_grid_dict['int_edge_y_'] = olddict['int_edge_x_']
                new_grid_dict['int_edge_z_'] = olddict['int_edge_y_']
            if out_float: 
                new_grid_dict['float_center_x_'] = olddict['float_center_z_']
                new_grid_dict['float_center_y_'] = olddict['float_center_x_']
                new_grid_dict['float_center_z_'] = olddict['float_center_y_']
            if input_type=="pointcloud" or input_type=="noisypc":
                new_pc_dict['input_x_'] = oldpcdict['input_z_']
                new_pc_dict['input_y_'] = oldpcdict['input_x_']
                new_pc_dict['input_z_'] = oldpcdict['input_y_']


    #store outputs 然后将增强后的数据再次保存起来
    if out_bool:
        LOD_gt_int = np.zeros([grid_size_1, grid_size_1, grid_size_1, 3], np.int32)
        LOD_gt_int[:-1,:,:,0] = new_grid_dict['int_edge_x_']
        LOD_gt_int[:,:-1,:,1] = new_grid_dict['int_edge_y_']
        LOD_gt_int[:,:,:-1,2] = new_grid_dict['int_edge_z_']
    else:
        LOD_gt_int = None

    if out_float:
        LOD_gt_float = np.full([grid_size_1,grid_size_1,grid_size_1,3], -1, np.float32)
        LOD_gt_float[:-1,:-1,:-1,0] = new_grid_dict['float_center_x_']
        LOD_gt_float[:-1,:-1,:-1,1] = new_grid_dict['float_center_y_']
        LOD_gt_float[:-1,:-1,:-1,2] = new_grid_dict['float_center_z_']
    else:
        LOD_gt_float = None

    if input_type=="pointcloud" or input_type=="noisypc":
        LOD_input = np.zeros([len(LOD_input),3], np.float32)
        LOD_input[:,0] = new_pc_dict['input_x_']
        LOD_input[:,1] = new_pc_dict['input_y_']
        LOD_input[:,2] = new_pc_dict['input_z_']

    #vertices, triangles = dual_contouring_ndc_surface_test(LOD_gt_int, LOD_gt_float)
    #write_obj_triangle("samples/test_1.obj", vertices, triangles)
    #write_ply_point("samples/test_1.ply", LOD_input)
    #exit(0)

    return LOD_gt_int, LOD_gt_float, LOD_input


#this is not an efficient implementation. just for testing!
def dual_contouring_undc_test(int_grid, float_grid):
    all_vertices = []
    all_triangles = []

    dimx,dimy,dimz,_ = int_grid.shape
    vertices_grid = np.full([dimx,dimy,dimz], -1, np.int32)

    #all vertices
    for i in range(0,dimx-1):
        for j in range(0,dimy-1):
            for k in range(0,dimz-1):
            
                ex0 = int_grid[i,j,k,0]
                ex1 = int_grid[i,j+1,k,0]
                ex2 = int_grid[i,j+1,k+1,0]
                ex3 = int_grid[i,j,k+1,0]

                ey0 = int_grid[i,j,k,1]
                ey1 = int_grid[i+1,j,k,1]
                ey2 = int_grid[i+1,j,k+1,1]
                ey3 = int_grid[i,j,k+1,1]

                ez0 = int_grid[i,j,k,2]
                ez1 = int_grid[i+1,j,k,2]
                ez2 = int_grid[i+1,j+1,k,2]
                ez3 = int_grid[i,j+1,k,2]
                
                if ex0 or ex1 or ex2 or ex3 or ey0 or ey1 or ey2 or ey3 or ez0 or ez1 or ez2 or ez3:
                    #add a vertex
                    vertices_grid[i,j,k] = len(all_vertices)
                    pos = float_grid[i,j,k]+np.array([i,j,k], np.float32)
                    all_vertices.append(pos)

    all_vertices = np.array(all_vertices, np.float32)


    #all triangles

    #i-direction
    for i in range(0,dimx-1):
        for j in range(1,dimy-1):
            for k in range(1,dimz-1):
                if int_grid[i,j,k,0]:
                    all_triangles.append([vertices_grid[i,j-1,k-1],vertices_grid[i,j,k-1],vertices_grid[i,j,k]])
                    all_triangles.append([vertices_grid[i,j-1,k-1],vertices_grid[i,j,k],vertices_grid[i,j-1,k]])

    #j-direction
    for i in range(1,dimx-1):
        for j in range(0,dimy-1):
            for k in range(1,dimz-1):
                if int_grid[i,j,k,1]:
                    all_triangles.append([vertices_grid[i-1,j,k-1],vertices_grid[i,j,k],vertices_grid[i,j,k-1]])
                    all_triangles.append([vertices_grid[i-1,j,k-1],vertices_grid[i-1,j,k],vertices_grid[i,j,k]])

    #k-direction
    for i in range(1,dimx-1):
        for j in range(1,dimy-1):
            for k in range(0,dimz-1):
                if int_grid[i,j,k,2]:
                    all_triangles.append([vertices_grid[i-1,j-1,k],vertices_grid[i,j,k],vertices_grid[i-1,j,k]])
                    all_triangles.append([vertices_grid[i-1,j-1,k],vertices_grid[i,j-1,k],vertices_grid[i,j,k]])

    all_triangles = np.array(all_triangles, np.int32)

    return all_vertices, all_triangles




def write_obj_triangle(name, vertices, triangles):
    fout = open(name, 'w+')
    for ii in range(len(vertices)):
        fout.write("v "+str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+"\n")
    for ii in range(len(triangles)):
        fout.write("f "+str(int(triangles[ii,0]+1))+" "+str(int(triangles[ii,1]+1))+" "+str(int(triangles[ii,2]+1))+"\n")
    fout.close()

def write_ply_triangle(name, vertices, triangles):
    fout = open(name, 'w')
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex "+str(len(vertices))+"\n")
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    fout.write("element face "+str(len(triangles))+"\n")
    fout.write("property list uchar int vertex_index\n")
    fout.write("end_header\n")
    for ii in range(len(vertices)):
        fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+"\n")
    for ii in range(len(triangles)):
        fout.write("3 "+str(triangles[ii,0])+" "+str(triangles[ii,1])+" "+str(triangles[ii,2])+"\n")
    fout.close()

def write_ply_point(name, vertices):
    fout = open(name, 'w+')
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex "+str(len(vertices))+"\n")
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    fout.write("end_header\n")
    for ii in range(len(vertices)):
        fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+"\n")
    fout.close()

def write_ply_point_normal(name, vertices, normals=None):
    fout = open(name, 'w')
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex "+str(len(vertices))+"\n")
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    fout.write("property float nx\n")
    fout.write("property float ny\n")
    fout.write("property float nz\n")
    fout.write("end_header\n")
    if normals is None:
        for ii in range(len(vertices)):
            fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+" "+str(vertices[ii,3])+" "+str(vertices[ii,4])+" "+str(vertices[ii,5])+"\n")
    else:
        for ii in range(len(vertices)):
            fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+" "+str(normals[ii,0])+" "+str(normals[ii,1])+" "+str(normals[ii,2])+"\n")
    fout.close()
