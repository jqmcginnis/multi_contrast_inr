import nibabel
import numpy as np
import torch
from tqdm import tqdm
# import SimpleITK as sitk
import torch.nn.functional as F
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler

from scipy import ndimage
from skimage import filters

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


def calculate_sobel_filter(image: nibabel.Nifti1Image):
    img_data = image.get_fdata()

    sobelx = ndimage.sobel(img_data, axis=0)
    sobely = ndimage.sobel(img_data, axis=1)
    sobelz = ndimage.sobel(img_data, axis=2)

    sobel = np.sqrt(sobelx**2 + sobely**2, sobelz**2)

    scaler = MinMaxScaler()
    sobel = scaler.fit_transform(sobel.reshape(-1, 1)).reshape(img_data.shape)

    return sobel


def calculate_sobel_median_filter(image: nibabel.Nifti1Image, median_filter_size=(1,1,1)):
    img_data = image.get_fdata()

    sobelx = ndimage.sobel(img_data, axis=0)
    sobely = ndimage.sobel(img_data, axis=1)
    sobelz = ndimage.sobel(img_data, axis=2)

    sobel = np.sqrt(sobelx**2 + sobely**2, sobelz**2)

    sobel = ndimage.median_filter(sobel, size=median_filter_size)

    scaler = MinMaxScaler()
    sobel = scaler.fit_transform(sobel.reshape(-1, 1)).reshape(img_data.shape)

    return sobel


def calculate_laplacian(image: nibabel.Nifti1Image):

    img_data = image.get_fdata()
    # Apply a Gaussian filter to smooth the image
    img_smooth = filters.gaussian(img_data, sigma=1)
    # Calculate the Laplacian of the smoothed image using the filters.laplace function

    laplacian = np.abs(filters.laplace(img_smooth))
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(laplacian.reshape(-1, 1)).reshape(img_data.shape)

    return scaled


def get_image_coordinate_grid_nib(image: nibabel.Nifti1Image, slice=False):
    img_header = image.header
    img_data = image.get_fdata()
    img_affine = image.affine

    (x, y, z) = image.shape

    label = []
    coordinates = []

    for i in tqdm(range(x)):
        for j in range(y):
            for k in range(z):
                coordinates.append(nib.affines.apply_affine(img_affine, np.array(([i, j, k]))))
                label.append(img_data[i, j, k])

    # convert to numpy array
    coordinates_arr = np.array(coordinates, dtype=np.float32)
    label_arr = np.array(label, dtype=np.float32)

    # coordinates_arr_norm = scaler.fit_transform(coordinates_arr)

    def min_max_scale(X, s_min, s_max):
        x_min, x_max = X.min(), X.max()
        return (X - x_min)/(x_max - x_min)*(s_max - s_min) + s_min

    coordinates_arr_norm = min_max_scale(X=coordinates_arr, s_min=-1, s_max=1)

    scaler = MinMaxScaler()

    label_arr_norm = scaler.fit_transform(label_arr.reshape(-1, 1))

    if slice:

        coordinates_arr = coordinates_arr.reshape(x,y,z,3)
        label_arr = label_arr.reshape(x,y,z,1)

        coordinates_arr = coordinates_arr[:,:,::3,:].reshape(-1,3)
        label_arr = label_arr[:,:,::3,:].reshape(-1,1)
            
        coordinates_arr_norm = coordinates_arr_norm.reshape(x,y,z,3)
        label_arr_norm = label_arr_norm.reshape(x,y,z,1)

        coordinates_arr_norm = coordinates_arr_norm[:,:,::3,:].reshape(-1,3)
        label_arr_norm = label_arr_norm[:,:,::3,:].reshape(-1,1)
            
    # normalize intensities and coordinates
    # image_grid_norm = torch.tensor(image_grid, dtype=torch.float32)
    # image_data_norm = torch.tensor(label_arr_norm, dtype=torch.float32).view(-1,1)

    x_min, y_min, z_min = nib.affines.apply_affine(img_affine, np.array(([0, 0, 0])))
    x_max, y_max, z_max = nib.affines.apply_affine(img_affine, np.array(([x, y, z])))

    boundaries = dict()
    boundaries['xmin'] = x_min
    boundaries['ymin'] = y_min
    boundaries['zmin'] = z_min
    boundaries['xmax'] = x_max
    boundaries['ymax'] = y_max
    boundaries['zmax'] = z_max

    image_dict = {
        'boundaries': boundaries,
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
