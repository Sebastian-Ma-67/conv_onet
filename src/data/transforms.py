import numpy as np


# Transforms
class PointcloudNoise(object):
    ''' Point cloud noise transformation class.

    It adds noise to point cloud data.

    Args:
        stddev (int): standard deviation
    '''

    def __init__(self, stddev):
        self.stddev = stddev

    def __call__(self, points_with_normals):
        ''' Calls the transformation.

        Args:
            data (dictionary): data dictionary
        '''
        points_with_normals_with_noise = points_with_normals.copy()
        points_tmp = points_with_normals['points']
        noise_tmp = self.stddev * np.random.randn(*points_tmp.shape)
        noise_tmp = noise_tmp.astype(np.float32)
        points_with_normals_with_noise['points'] = points_tmp + noise_tmp
        return points_with_normals_with_noise

class SubsamplePointcloud(object):
    ''' Point cloud subsampling transformation class.

    It subsamples the point cloud data.

    Args:
        N (int): number of points to be subsampled
    '''
    def __init__(self, N):
        self.N = N

    def __call__(self, points_with_normals):
        ''' Calls the transformation.

        Args:
            data (dict): data dictionary
        '''
        points_with_normals_out = points_with_normals.copy()
        points = points_with_normals['points']
        normals = points_with_normals['normals']


        np.random.seed(0) # 这里我们先让种子固定，方便测试
        indices = np.random.randint(points.shape[0], size=self.N)
        points_with_normals_out['points'] = points[indices, :]
        points_with_normals_out['normals'] = normals[indices, :]

        return points_with_normals_out


class SubsamplePoints(object):
    ''' Points subsampling transformation class.

    It subsamples the points data.

    Args:
        N (int): number of points to be subsampled
    '''
    def __init__(self, N):
        self.N = N

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dictionary): data dictionary
        '''
        points = data[None]
        occ = data['occ']

        data_out = data.copy()
        if isinstance(self.N, int):
            idx = np.random.randint(points.shape[0], size=self.N)
            data_out.update({
                None: points[idx, :],
                'occ':  occ[idx],
            })
        else:
            Nt_out, Nt_in = self.N
            occ_binary = (occ >= 0.5)
            points0 = points[~occ_binary]
            points1 = points[occ_binary]

            idx0 = np.random.randint(points0.shape[0], size=Nt_out)
            idx1 = np.random.randint(points1.shape[0], size=Nt_in)

            points0 = points0[idx0, :]
            points1 = points1[idx1, :]
            points = np.concatenate([points0, points1], axis=0)

            occ0 = np.zeros(Nt_out, dtype=np.float32)
            occ1 = np.ones(Nt_in, dtype=np.float32)
            occ = np.concatenate([occ0, occ1], axis=0)

            volume = occ_binary.sum() / len(occ_binary)
            volume = volume.astype(np.float32)

            data_out.update({
                None: points,
                'occ': occ,
                'volume': volume,
            })
        return data_out
