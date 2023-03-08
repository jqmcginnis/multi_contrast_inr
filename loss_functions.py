import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch import Tensor
from typing import Optional, Tuple
from torch.nn.modules.loss import _Loss
import pdb


class StableStd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        assert tensor.numel() > 1
        ctx.tensor = tensor.detach()
        res = torch.std(tensor).detach()
        ctx.result = res.detach()
        return res

    @staticmethod
    def backward(ctx, grad_output):
        tensor = ctx.tensor.detach()
        result = ctx.result.detach()
        e = 1e-6
        assert tensor.numel() > 1
        return (
            (2.0 / (tensor.numel() - 1.0))
            * (grad_output.detach() / (result.detach() * 2 + e))
            * (tensor.detach() - tensor.mean().detach())
        )


stablestd = StableStd.apply

class _WeightedLoss(_Loss):
    def __init__(self, weight: Optional[Tensor] = None, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(_WeightedLoss, self).__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)
        self.weight: Optional[Tensor]
# TODO: add masks.
def ncc(x1, x2, e=1e-10):
    assert x1.shape == x2.shape, "Inputs are not of equal shape"
    cc = ((x1 - x1.mean()) * (x2 - x2.mean())).mean()
    std = stablestd(x1) * stablestd(x2)
    ncc = cc / (std + e)
    return ncc


def ncc_mask(x1, x2, mask, e=1e-10):  # TODO: calculate ncc per sample
    assert x1.shape == x2.shape, "Inputs are not of equal shape"
    x1 = torch.masked_select(x1, mask)
    x2 = torch.masked_select(x2, mask)
    cc = ((x1 - x1.mean()) * (x2 - x2.mean())).mean()
    std = stablestd(x1) * stablestd(x2)
    ncc = cc / (std + e)
    return ncc


class NCC(_Loss):
    def __init__(self, use_mask: bool = False):
        super().__init__()
        if use_mask:
            self.forward = self.masked_metric
        else:
            self.forward = self.metric

    def metric(self, fixed: Tensor, warped: Tensor) -> Tensor:
        return -ncc(fixed, warped)

    def masked_metric(self, fixed: Tensor, warped: Tensor, mask: Tensor) -> Tensor:
        return -ncc_mask(fixed, warped, mask)



class NMI(_Loss):
    """Normalized mutual information metric.

    As presented in the work by `De Vos 2020: <https://www.spiedigitallibrary.org/conference-proceedings-of-spie/11313/113130R/Mutual-information-for-unsupervised-deep-learning-image-registration/10.1117/12.2549729.full?SSO=1>`_

    """

    def __init__(
        self,
        intensity_range: Optional[Tuple[float, float]] = None,
        nbins: int = 32,
        sigma: float = 0.1,
        use_mask: bool = False,
    ):
        super().__init__()
        self.intensity_range = intensity_range
        self.nbins = nbins
        self.sigma = sigma
        if use_mask:
            self.forward = self.masked_metric
        else:
            self.forward = self.metric

    def metric(self, fixed: Tensor, warped: Tensor) -> Tensor:
        with torch.no_grad():
            if self.intensity_range:
                fixed_range = self.intensity_range
                warped_range = self.intensity_range
            else:
                fixed_range = fixed.min(), fixed.max()
                warped_range = warped.min(), warped.max()

        bins_fixed = torch.linspace(
            fixed_range[0],
            fixed_range[1],
            self.nbins,
            dtype=fixed.dtype,
            device=fixed.device,
        )
        bins_warped = torch.linspace(
            warped_range[0],
            warped_range[1],
            self.nbins,
            dtype=fixed.dtype,
            device=fixed.device,
        )

        return -nmi_gauss(
            fixed, warped, bins_fixed, bins_warped, sigma=self.sigma
        ).mean()

    def masked_metric(self, fixed: Tensor, warped: Tensor, mask: Tensor) -> Tensor:
        with torch.no_grad():
            if self.intensity_range:
                fixed_range = self.intensity_range
                warped_range = self.intensity_range
            else:
                fixed_range = fixed.min(), fixed.max()
                warped_range = warped.min(), warped.max()

        bins_fixed = torch.linspace(
            fixed_range[0],
            fixed_range[1],
            self.nbins,
            dtype=fixed.dtype,
            device=fixed.device,
        )
        bins_warped = torch.linspace(
            warped_range[0],
            warped_range[1],
            self.nbins,
            dtype=fixed.dtype,
            device=fixed.device,
        )

        return -nmi_gauss_mask(
            fixed, warped, bins_fixed, bins_warped, mask, sigma=self.sigma
        )


def nmi_gauss(x1, x2, x1_bins, x2_bins, sigma=1e-3, e=1e-10):
    assert x1.shape == x2.shape, "Inputs are not of similar shape"

    def gaussian_window(x, bins, sigma):
        assert x.ndim == 2, "Input tensor should be 2-dimensional."
        return torch.exp(
            -((x[:, None, :] - bins[None, :, None]) ** 2) / (2 * sigma ** 2)
        ) / (math.sqrt(2 * math.pi) * sigma)

    x1_windowed = gaussian_window(x1.flatten(1), x1_bins, sigma)
    x2_windowed = gaussian_window(x2.flatten(1), x2_bins, sigma)
    p_XY = torch.bmm(x1_windowed, x2_windowed.transpose(1, 2))
    p_XY = p_XY + e  # deal with numerical instability

    p_XY = p_XY / p_XY.sum((1, 2))[:, None, None]

    p_X = p_XY.sum(1)
    p_Y = p_XY.sum(2)

    I = (p_XY * torch.log(p_XY / (p_X[:, None] * p_Y[:, :, None]))).sum((1, 2))

    marg_ent_0 = (p_X * torch.log(p_X)).sum(1)
    marg_ent_1 = (p_Y * torch.log(p_Y)).sum(1)

    normalized = -1 * 2 * I / (marg_ent_0 + marg_ent_1)  # harmonic mean

    return normalized


def nmi_gauss_mask(x1, x2, x1_bins, x2_bins, mask, sigma=1e-3, e=1e-10):
    def gaussian_window_mask(x, bins, sigma):

        assert x.ndim == 1, "Input tensor should be 2-dimensional."
        return torch.exp(-((x[None, :] - bins[:, None]) ** 2) / (2 * sigma ** 2)) / (
            math.sqrt(2 * math.pi) * sigma
        )

    x1_windowed = gaussian_window_mask(torch.masked_select(x1, mask), x1_bins, sigma)
    x2_windowed = gaussian_window_mask(torch.masked_select(x2, mask), x2_bins, sigma)
    p_XY = torch.mm(x1_windowed, x2_windowed.transpose(0, 1))
    p_XY = p_XY + e  # deal with numerical instability

    p_XY = p_XY / p_XY.sum()

    p_X = p_XY.sum(0)
    p_Y = p_XY.sum(1)

    I = (p_XY * torch.log(p_XY / (p_X[None] * p_Y[:, None]))).sum()

    marg_ent_0 = (p_X * torch.log(p_X)).sum()
    marg_ent_1 = (p_Y * torch.log(p_Y)).sum()

    normalized = -1 * 2 * I / (marg_ent_0 + marg_ent_1)  # harmonic mean

    return normalized

class MILossGaussian(nn.Module):
    """
    Mutual information loss using Gaussian kernel in KDE
    """
    def __init__(self,
                 vmin=0.0,
                 vmax=1.0,
                 num_bins=16, # 32 or 64 
                 sample_ratio=1.0, #0.7 for t1 and t2
                 normalised=True,
                 gt_val=None,
                 ):
        super(MILossGaussian, self).__init__()

        self.vmin = vmin
        self.vmax = vmax
        self.sample_ratio = sample_ratio
        self.normalised = normalised
        self.gt_val = gt_val

        # set the std of Gaussian kernel so that FWHM is one bin width
        bin_width = (vmax - vmin) / num_bins
        self.sigma = bin_width * (1/(2 * math.sqrt(2 * math.log(2))))

        # set bin edges
        self.num_bins = num_bins
        self.bins = torch.linspace(self.vmin, self.vmax, self.num_bins, requires_grad=False).unsqueeze(1)

    def _compute_joint_prob(self, x, y):
        """
        Compute joint distribution and entropy
        Input shapes (N, 1, prod(sizes))
        """
        # cast bins
        self.bins = self.bins.type_as(x)

        # calculate Parzen window function response (N, #bins, H*W*D)
        win_x = torch.exp(-(x - self.bins) ** 2 / (2 * self.sigma ** 2))
        win_x = win_x / (math.sqrt(2 * math.pi) * self.sigma)
        win_y = torch.exp(-(y - self.bins) ** 2 / (2 * self.sigma ** 2))
        win_y = win_y / (math.sqrt(2 * math.pi) * self.sigma)

        # calculate joint histogram batch
        hist_joint = win_x.bmm(win_y.transpose(1, 2))  # (N, #bins, #bins)

        # normalise joint histogram to get joint distribution
        hist_norm = hist_joint.flatten(start_dim=1, end_dim=-1).sum(dim=1) + 1e-5
        p_joint = hist_joint / hist_norm.view(-1, 1, 1)  # (N, #bins, #bins) / (N, 1, 1)

        return p_joint

    def forward(self, x, y):
        """
        Calculate (Normalised) Mutual Information Loss.
        Args:
            x: (torch.Tensor, size (N, 1, *sizes))
            y: (torch.Tensor, size (N, 1, *sizes))
        Returns:
            (Normalise)MI: (scalar)
        """
        if self.sample_ratio < 1.:
            # random spatial sampling with the same number of pixels/voxels
            # chosen for every sample in the batch
            numel_ = np.prod(x.size()[2:])
            idx_th = int(self.sample_ratio * numel_)
            idx_choice = torch.randperm(int(numel_))[:idx_th]

            x = x.view(x.size()[0], 1, -1)[:, :, idx_choice]
            y = y.view(y.size()[0], 1, -1)[:, :, idx_choice]

        # make sure the sizes are (N, 1, prod(sizes))
        x = x.flatten(start_dim=2, end_dim=-1)
        y = y.flatten(start_dim=2, end_dim=-1)


        # compute joint distribution
        p_joint = self._compute_joint_prob(x, y)
        p_joint = p_joint/ p_joint.sum()

        # marginalise the joint distribution to get marginal distributions
        # batch size in dim0, x bins in dim1, y bins in dim2
        p_x = torch.sum(p_joint, dim=2)
        p_y = torch.sum(p_joint, dim=1)

        #px_py = p_x[:, None] * p_y[None, :]
        # px_py = (p_y.T @ p_x).unsqueeze(0)

        px_py = torch.outer(p_x.squeeze(), p_y.squeeze()).unsqueeze(0)
        nzs = p_joint >0 #>0

        # calculate entropy
        #ent_x = - torch.sum(p_x * torch.log(p_x + 1e-5), dim=1)  # (N,1)
        #ent_y = - torch.sum(p_y * torch.log(p_y + 1e-5), dim=1)  # (N,1)
        #ent_joint = - torch.sum(p_joint * torch.log(p_joint + 1e-5), dim=(1, 2))  # (N,1)

        '''

        if self.normalised:
            if self.gt_val:
                return (torch.mean((ent_x + ent_y) / ent_joint)-self.gt_val)**2
            else:
                return torch.mean((ent_x + ent_y) / ent_joint)
        else:
            if self.gt_val:
                return (torch.mean(ent_x + ent_y - ent_joint)-self.gt_val)**2
            else:
                return torch.mean(ent_x + ent_y - ent_joint)
        '''
        # pdb.set_trace()

        if self.gt_val:
            return ((p_joint[nzs] * torch.log(p_joint[nzs]/ (px_py[nzs]+ 1e-12))).sum()-self.gt_val)**2
        else:
            return (p_joint[nzs] * torch.log(p_joint[nzs]/ (px_py[nzs]+ 1e-12))).sum()




class LNCCLoss(nn.Module):
    """
    Local Normalized Cross Correlation loss
    Adapted from VoxelMorph implementation:
    https://github.com/voxelmorph/voxelmorph/blob/5273132227c4a41f793903f1ae7e27c5829485c8/voxelmorph/torch/losses.py#L7
    """
    def __init__(self, window_size=7):
        super(LNCCLoss, self).__init__()
        self.window_size = window_size

    def forward(self, x, y):
        # products and squares
        xsq = x * x
        ysq = y * y
        xy = x * y

        # set window size
        ndim = x.dim() - 2
        window_size = param_ndim_setup(self.window_size, ndim)

        # summation filter for convolution
        sum_filt = torch.ones(1, 1, *window_size).type_as(x)

        # set stride and padding
        stride = (1,) * ndim
        padding = tuple([math.floor(window_size[i]/2) for i in range(ndim)])

        # get convolution function of the correct dimension
        conv_fn = getattr(F, f'conv{ndim}d')

        # summing over window by convolution
        x_sum = conv_fn(x, sum_filt, stride=stride, padding=padding)
        y_sum = conv_fn(y, sum_filt, stride=stride, padding=padding)
        xsq_sum = conv_fn(xsq, sum_filt, stride=stride, padding=padding)
        ysq_sum = conv_fn(ysq, sum_filt, stride=stride, padding=padding)
        xy_sum = conv_fn(xy, sum_filt, stride=stride, padding=padding)

        window_num_points = np.prod(window_size)
        x_mu = x_sum / window_num_points
        y_mu = y_sum / window_num_points

        cov = xy_sum - y_mu * x_sum - x_mu * y_sum + x_mu * y_mu * window_num_points
        x_var = xsq_sum - 2 * x_mu * x_sum + x_mu * x_mu * window_num_points
        y_var = ysq_sum - 2 * y_mu * y_sum + y_mu * y_mu * window_num_points

        lncc = cov * cov / (x_var * y_var + 1e-5)

        return -torch.mean(lncc)


def l2reg_loss(u):
    """L2 regularisation loss"""
    derives = []
    ndim = u.size()[1]
    for i in range(ndim):
        derives += [finite_diff(u, dim=i)]
    loss = torch.cat(derives, dim=1).pow(2).sum(dim=1).mean()
    return loss


def bending_energy_loss(u):
    """Bending energy regularisation loss"""
    derives = []
    ndim = u.size()[1]
    # 1st order
    for i in range(ndim):
        derives += [finite_diff(u, dim=i)]
    # 2nd order
    derives2 = []
    for i in range(ndim):
        derives2 += [finite_diff(derives[i], dim=i)]  # du2xx, du2yy, (du2zz)
    derives2 += [math.sqrt(2) * finite_diff(derives[0], dim=1)]  # du2dxy
    if ndim == 3:
        derives2 += [math.sqrt(2) * finite_diff(derives[0], dim=2)]  # du2dxz
        derives2 += [math.sqrt(2) * finite_diff(derives[1], dim=2)]  # du2dyz

    assert len(derives2) == 2 * ndim
    loss = torch.cat(derives2, dim=1).pow(2).sum(dim=1).mean()
    return loss


def finite_diff(x, dim, mode="forward", boundary="Neumann"):
    """Input shape (N, ndim, *sizes), mode='foward', 'backward' or 'central'"""
    assert type(x) is torch.Tensor
    ndim = x.ndim - 2
    sizes = x.shape[2:]

    if mode == "central":
        # TODO: implement central difference by 1d conv or dialated slicing
        raise NotImplementedError("Finite difference central difference mode")
    else:  # "forward" or "backward"
        # configure padding of this dimension
        paddings = [[0, 0] for _ in range(ndim)]
        if mode == "forward":
            # forward difference: pad after
            paddings[dim][1] = 1
        elif mode == "backward":
            # backward difference: pad before
            paddings[dim][0] = 1
        else:
            raise ValueError(f'Mode {mode} not recognised')

        # reverse and join sublists into a flat list (Pytorch uses last -> first dim order)
        paddings.reverse()
        paddings = [p for ppair in paddings for p in ppair]

        # pad data
        if boundary == "Neumann":
            # Neumann boundary condition
            x_pad = F.pad(x, paddings, mode='replicate')
        elif boundary == "Dirichlet":
            # Dirichlet boundary condition
            x_pad = F.pad(x, paddings, mode='constant')
        else:
            raise ValueError("Boundary condition not recognised.")

        # slice and subtract
        x_diff = x_pad.index_select(dim + 2, torch.arange(1, sizes[dim] + 1).to(device=x.device)) \
                 - x_pad.index_select(dim + 2, torch.arange(0, sizes[dim]).to(device=x.device))

        return x_diff

def param_ndim_setup(param, ndim):
    """
    Check dimensions of paramters and extend dimension if needed.
    Args:
        param: (int/float, tuple or list) check dimension match if tuple or list is given,
                expand to `dim` by repeating if a single integer/float number is given.
        ndim: (int) data/model dimension
    Returns:
        param: (tuple)
    """
    if isinstance(param, (int, float)):
        param = (param,) * ndim
    elif isinstance(param, (tuple, list, omegaconf.listconfig.ListConfig)):
        assert len(param) == ndim, \
            f"Dimension ({ndim}) mismatch with data"
        param = tuple(param)
    else:
        raise TypeError("Parameter type not int, tuple or list")
    return param

def total_variation_loss(img, edge_map=None):
     if edge_map != None:
          edge_map = 1.0 -edge_map.permute(0,2,3,1)
     else:
          edge_map = torch.ones(img.shape).cuda()
     tv_h1 = (torch.pow(img[:,:,:,1:,:]-img[:,:,:,:-1,:], 2)*edge_map[:,:,:,1:,:]).sum()
     tv_w1 = (torch.pow(img[:,:,1:,:,:]-img[:,:,:-1,:,:], 2)*edge_map[:,:,1:,:,:]).sum()
     tv_d1 = (torch.pow(img[:,1:,:,:,:]-img[:,-1:,:,:,:], 2)*edge_map[:,1:,:,:,:]).sum()
     
     tv_h2 = (torch.pow(img[:,:,:,1:,:]-img[:,:,:,:-1,:], 2)*edge_map[:,:,:,:-1,:]).sum()
     tv_w2 = (torch.pow(img[:,:,1:,:,:]-img[:,:,:-1,:,:], 2)*edge_map[:,:,:-1,:,:]).sum()
     tv_d2 = (torch.pow(img[:,1:,:,:,:]-img[:,:-1,:,:,:], 2)*edge_map[:,:-1,:,:,:]).sum()

     return 0.163*(tv_h1+tv_w1+tv_d1+tv_h2+tv_w2+tv_d2)/edge_map.sum()