import nibabel
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler

# from Tancik et al.:
# https://github.com/tancik/fourier-feature-networks/blob/master/Experiments/3d_MRI.ipynb
# https://github.com/tancik/fourier-feature-networks/blob/master/Demo.ipynb
# Fourier feature mapping
def input_mapping(x, B):
    '''
    :param x: vector if input features
    :param B: matrix or None
    :return: 
    '''
    if B is None:
        return x
    else:
        x_proj = (2. * np.pi * x) @ B.T
        return np.concatenate([np.sin(x_proj), np.cos(x_proj)], axis=-1)


def norm_grid(grid, xmin, xmax, smin=-1, smax=1):
    def min_max_scale(X, x_min, x_max, s_min, s_max):
        return (X - x_min)/(x_max - x_min)*(s_max - s_min) + s_min

    return min_max_scale(X=grid, x_min=xmin, x_max=xmax, s_min=smin, s_max=smax)

def get_image_coordinate_grid_nib(image: nibabel.Nifti1Image):
    
    scaler = MinMaxScaler()
    img_header = image.header
    img_data = image.get_fdata()
    img_affine = image.affine
    (x, y, z) = image.shape
    label = []
    coordinates = []

    X = np.linspace(0,x-1,x)
    Y = np.linspace(0,y-1,y)
    Z = np.linspace(0,z-1,z)
    points = np.meshgrid(X,Y,Z, indexing='ij')
    points = np.stack(points).transpose(1,2,3,0).reshape(-1,3)
    coordinates = list(nib.affines.apply_affine(img_affine, points))
    label = list(img_data.flatten())

    # convert to numpy array
    coordinates_arr = np.array(coordinates, dtype=np.float32)
    label_arr = np.array(label, dtype=np.float32)

    def min_max_scale(X, s_min, s_max):
        x_min, x_max = X.min(), X.max()
        return (X - x_min)/(x_max - x_min)*(s_max - s_min) + s_min

    coordinates_arr_norm = min_max_scale(X=coordinates_arr, s_min=-1, s_max=1)
    label_arr_norm = scaler.fit_transform(label_arr.reshape(-1, 1))

    image_dict = {
        'affine': torch.tensor(img_affine),
        'origin': torch.tensor(np.array([0])),
        'spacing': torch.tensor(np.array(img_header["pixdim"][1:4])),
        'dim': torch.tensor(np.array([x, y, z])),
        'intensity': torch.tensor(label_arr, dtype=torch.float32).view(-1, 1),
        'intensity_norm': torch.tensor(label_arr_norm, dtype=torch.float32).view(-1, 1),
        'coordinates': torch.tensor(coordinates_arr, dtype=torch.float32),
        'coordinates_norm': torch.tensor(coordinates_arr_norm, dtype=torch.float32),
    }
    return image_dict
