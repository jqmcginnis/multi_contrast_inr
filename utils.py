import torch
import numpy as np
import json
from typing import List, Tuple, Optional
import torch.nn as nn
from math import log, sqrt
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import lpips
from loss_functions import MILossGaussian
import matplotlib.pyplot as plt 
import nibabel as nib
import nibabel.processing as nip
import nibabel.orientations as nio

class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)
        
def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)

def get_string(my_dict):
    return '_'.join([str(value) for value in my_dict.values()])


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def min_max_scale(X, s_min, s_max):
    x_min, x_max = X.min(), X.max()
    return torch.tensor((X - x_min) / (x_max - x_min) * (s_max - s_min) + s_min)


# from official FF repository
class input_mapping(nn.Module):
    def __init__(self, B=None, factor=1.0):
        super(input_mapping, self).__init__()
        self.B = factor * B
    
    def forward(self, x):

        x_proj = (2. * np.pi * x) @ self.B.T
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


def compute_metrics(gt, pred, mask, lpips_loss, device):

    if type(mask) == torch.Tensor:
        mask = mask.float().cpu().numpy()

    assert mask.max() == 1.0, 'Mask Format incorrect.'
    assert mask.min() == 0.0, 'Mask Format incorrect.'

    gt -= gt[mask == 1].min()
    gt /= gt.max()
    gt *= mask

    pred -= pred[mask == 1].min()
    pred /= pred.max()
    pred *= mask

    ssim = structural_similarity(gt, pred, data_range=1)
    psnr = peak_signal_noise_ratio(gt, pred, data_range=1)

    x, y, z = pred.shape

    lpips_val = 0

    for i in range(x):
        pred_t = torch.tensor(pred[i,:,:]).reshape(1, y, z).repeat(3,1,1).to(device)
        gt_t = torch.tensor(gt[i,:,:]).reshape(1, y, z).repeat(3,1,1).to(device)
        lpips_val += lpips_loss(gt_t, pred_t)

    for i in range(y):
        pred_t = torch.tensor(pred[:,i,:]).reshape(1, x, z).repeat(3,1,1).to(device)
        gt_t = torch.tensor(gt[:,i,:]).reshape(1, x, z).repeat(3,1,1).to(device)
        lpips_val += lpips_loss(gt_t, pred_t)

    for i in range(z):
        pred_t = torch.tensor(pred[:,:,i]).reshape(1, x, y).repeat(3,1,1).to(device)
        gt_t = torch.tensor(gt[:,:,i]).reshape(1, x, y).repeat(3,1,1).to(device)
        lpips_val += lpips_loss(gt_t, pred_t)

    lpips_val /= (x+y+z)

    vals = {}
    vals["ssim"]= ssim
    vals["psnr"]= psnr
    vals["lpips"] = lpips_val.item()

    return vals


def compute_mi(pred1, pred2, mask, device):
    if type(mask) == torch.Tensor:
        mask = mask.float()

    mi_metric = MILossGaussian(num_bins=32).to(device)
    pred1 = torch.tensor(pred1[mask==1]).to(device).unsqueeze(0).unsqueeze(0)
    pred2 = torch.tensor(pred2[mask==1]).to(device).unsqueeze(0).unsqueeze(0)
    
    vals = {}
    vals['mi'] = mi_metric(pred1, pred2)
    return vals


# from: https://matthew-brett.github.io/teaching/mutual_information.html
def mutual_information(hgram):
    """ Mutual information for joint histogram
    """
    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))


# from: https://matthew-brett.github.io/teaching/mutual_information.html
def compute_mi_hist(img1, img2, mask, bins=32):

    if type(mask) == torch.Tensor:
        mask = mask.float().cpu().numpy()

    if type(img1) == torch.Tensor():
        img1 = img1.cpu().numpy()

    if type(img2)== torch.Tensor():
        img2 = img2.cpu().numpy()  

    # only inside of the brain
    img1 = img1[mask==1]
    img2 = img2[mask==1]

    hist_2d, _, _= np.histogram2d(
        img1.ravel(),
        img2.ravel(),
        bins=bins)

    vals = {}
    vals['mi'] = mutual_information(hist_2d)
    return vals


def resample_nib(img, voxel_spacing=(1, 1, 1), order=3):
    """Resamples the nifti from its original spacing to another specified spacing
    
    Parameters:
    ----------
    img: nibabel image
    voxel_spacing: a tuple of 3 integers specifying the desired new spacing
    order: the order of interpolation
    
    Returns:
    ----------
    new_img: The resampled nibabel image 
    
    """
    # resample to new voxel spacing based on the current x-y-z-orientation
    aff = img.affine
    shp = img.shape
    zms = img.header.get_zooms()
    # Calculate new shape
    new_shp = tuple(np.rint([
        shp[0] * zms[0] / voxel_spacing[0],
        shp[1] * zms[1] / voxel_spacing[1],
        shp[2] * zms[2] / voxel_spacing[2]
        ]).astype(int))
    new_aff = nib.affines.rescale_affine(aff, shp, voxel_spacing, new_shp)
    new_img = nip.resample_from_to(img, (new_shp, new_aff), order=order, cval=-1024)
    print("[*] Image resampled to voxel size:", voxel_spacing)
    return new_img