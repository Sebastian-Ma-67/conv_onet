import numpy as np

def makeGrid(bb_min=[0,0,0], bb_max=[1,1,1], shape=[10,10,10], 
    mode='on', flatten=True, indexing="ij"):
    """ Make a grid of coordinates

    Args:
    bb_min (list or np.array): least coordinate for each dimension
    bb_max (list or np.array): maximum coordinate for each dimension
    shape (list or int): list of coordinate number along each dimension. If it is an int, the number
                same for all dimensions
    mode (str, optional): 'on' to have vertices lying on the boundary and 
                'in' for keeping vertices and its cell inside of the boundary
                same as align_corners=True and align_corners=False
    flatten (bool, optional): Return as list of points or as a grid. Defaults to True.
    indexing (["ij" or "xy"]): default to "xy", see https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html

    Returns:
    np.array: return coordinates (XxYxZxD if flatten==False, X*Y*ZxD if flatten==True.
    """
    coords=[]
    bb_min = np.array(bb_min)
    bb_max = np.array(bb_max)
    if type(shape) is int:
        shape = np.array([shape]*bb_min.shape[0])
    for i, si in enumerate(shape):
        if mode=='on':
            coord = np.linspace(bb_min[i], bb_max[i], si)
        elif mode=='in':
            offset = (bb_max[i]-bb_min[i])/2./si
            # 2*si*w=bmax-bmin
            # w = (bmax-bmin)/2/si
            # start, end = bmax+w, bmin-w
            coord = np.linspace(bb_min[i]+offset,bb_max[i]-offset, si)
        coords.append( coord )
    grid = np.stack(np.meshgrid(*coords,sparse=False, indexing=indexing), axis=-1)
    
    if flatten==True:
        grid = grid.reshape(-1,grid.shape[-1])
    return grid

my_grid =  makeGrid(shape=[3, 3, 3])
print(my_grid)