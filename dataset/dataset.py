from pathlib import Path
import os
import torch
from torch.utils.data import Dataset
from typing import Tuple
from dataset.dataset_utils import get_image_coordinate_grid_nib, norm_grid
import nibabel as nib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class _BaseDataset(Dataset):
    """Base dataset class"""

    def __init__(self, image_dir):
        super(_BaseDataset, self).__init__()
        self.image_dir = image_dir
        assert os.path.exists(image_dir), f"Image Directory does not exist: {image_dir}!"

    def __getitem__(self, index):
        """ Load data and pre-process """
        raise NotImplementedError

    def __len__(self) -> int:
        r"""Returns the number of coordinates stored in the dataset."""
        raise NotImplementedError

    def process(self):
        r"""Processes the dataset to the :obj:`self.processed_dir` folder."""
        raise NotImplementedError

class MultiModalDataset(_BaseDataset):
    r""" Dataset of view1 and view2 T2w image sequence of the same patient.
    These could be e.g. an view1 and view2 T2w brain image,
    an view1 and view2 spine image, etc.
    However, both images must be registered to one another - the Dataset does not do this.
    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
    """

    def __init__(self, image_dir: str="", name = "BrainLesionDataset",
                subject_id: str = "123456", 
                contrast1_LR_str: str= 'flair3d_LR', 
                contrast2_LR_str: str='dir_LR',
  
                transform = None, target_transform = None):
        super(MultiModalDataset, self).__init__(image_dir)
        self.dataset_name = name
        self.subject_id = subject_id
        self.contrast1_LR_str = contrast1_LR_str
        self.contrast2_LR_str = contrast2_LR_str
        self.contrast1_LR_mask_str = contrast1_LR_str.replace("LR", "mask_LR")
        self.contrast2_LR_mask_str = contrast2_LR_str.replace("LR", "mask_LR")
        self.contrast1_GT_str = contrast1_LR_str.replace("_LR", "")
        self.contrast2_GT_str = contrast2_LR_str.replace("_LR", "")
        self.contrast1_GT_mask_str = "brainmask"
        self.contrast2_GT_mask_str = "brainmask"


        self.dataset_name = (
            f'preprocessed_data/{self.dataset_name}_'
            f'{self.subject_id}_'
            f'{self.contrast1_LR_str}_{self.contrast1_GT_str}_'
            f'{self.contrast2_LR_str}_{self.contrast2_GT_str}_'
            f'{self.contrast1_LR_mask_str}_{self.contrast2_LR_mask_str}_'
            f'{self.contrast1_GT_mask_str}_{self.contrast2_GT_mask_str}_'
            f'.pt'
        )

        print(self.dataset_name)

        files = sorted(list(Path(self.image_dir).rglob('*.nii.gz'))) 
        files = [str(x) for x in files]


        # only keep NIFTIs that follow specific subject 
        files = [k for k in files if self.subject_id in k]
        print(files)

        # flair3 and flair3d_LR or t1 and t1_LR
        gt_contrast1 = [x for x in files if self.contrast1_GT_str in x and self.contrast1_LR_str not in x and 'mask' not in x][0]
        gt_contrast2 = [x for x in files if self.contrast2_GT_str in x and self.contrast2_LR_str not in x and 'mask' not in x][0]

        lr_contrast1 = [x for x in files if self.contrast1_LR_str in x and 'mask' not in x][0]
        lr_contrast2 = [x for x in files if self.contrast2_LR_str in x and 'mask' not in x][0]

        lr_contrast1_mask = [x for x in files if self.contrast1_LR_mask_str in x and 'mask' in x][0]
        lr_contrast2_mask = [x for x in files if self.contrast2_LR_mask_str in x and 'mask' in x][0]

        gt_contrast1_mask = [x for x in files if self.contrast1_GT_mask_str in x and 'mask' in x][0]
        gt_contrast2_mask = [x for x in files if self.contrast2_GT_mask_str in x and 'mask' in x][0]

        self.lr_contrast1 = lr_contrast1
        self.lr_contrast2 = lr_contrast2
        self.lr_contrast1_mask = lr_contrast1_mask
        self.lr_contrast2_mask = lr_contrast2_mask
        self.gt_contrast1 = gt_contrast1
        self.gt_contrast2 = gt_contrast2
        self.gt_contrast1_mask = gt_contrast1_mask
        self.gt_contrast2_mask = gt_contrast2_mask

        if os.path.isfile(self.dataset_name):
            print("Dataset available.")
            dataset = torch.load(self.dataset_name)
            self.data = dataset["data"]
            self.label = dataset["label"]
            self.mask = dataset["mask"]
            self.affine = dataset["affine"]
            self.dim = dataset["dim"]
            self.len = dataset["len"]
            self.gt_contrast1 = dataset["gt_contrast1"]
            self.gt_contrast2 = dataset["gt_contrast2"]
            self.gt_contrast1_mask = dataset["gt_contrast1_mask"]
            self.gt_contrast2_mask = dataset["gt_contrast2_mask"]
            self.coordinates = dataset["coordinates"]
            print("skipping preprocessing.")

        else:
            self.len = 0
            self.data = []
            self.label = []
            self._process()

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.len

    def __getitem__(self, idx) -> Tuple[dict, dict]:
        data = self.data[idx]
        label = self.label[idx]
        mask = self.mask[idx]
        return data, label, mask

    def get_intensities(self):
        return self.label
    
    def get_mask(self):
        return self.mask

    def get_coordinates(self):
        return self.coordinates

    def get_affine(self):
        return self.affine
    
    def get_dim(self):
        return self.dim
       
    def get_contrast2_gt(self):
        return self.gt_contrast2
    
    def get_contrast1_gt(self):
        return self.gt_contrast1
        
    def get_contrast2_gt_mask(self):
        return self.gt_contrast2_mask
    
    def get_contrast1_gt_mask(self):
        return self.gt_contrast1_mask

    def _process(self):

        print(f"Using {self.lr_contrast1} as contrast1.")
        print(f"Using {self.lr_contrast2} as contrast2.")

        print(f"Using {self.lr_contrast1_mask} as contrast1 mask.")
        print(f"Using {self.lr_contrast2_mask} as contrast2 mask.")

        print(f"Using {self.gt_contrast1} as gt contrast1.")
        print(f"Using {self.gt_contrast2} as gt contrast2.")

        print(f"Using {self.gt_contrast1_mask} as gt contrast1 mask.")
        print(f"Using {self.gt_contrast2_mask} as gt contrast2 mask.")

        contrast1_dict = get_image_coordinate_grid_nib(nib.load(str(self.lr_contrast1)))
        contrast2_dict = get_image_coordinate_grid_nib(nib.load(str(self.lr_contrast2)))
        
        contrast1_mask_dict = get_image_coordinate_grid_nib(nib.load(str(self.lr_contrast1_mask)))
        contrast2_mask_dict = get_image_coordinate_grid_nib(nib.load(str(self.lr_contrast2_mask)))

        data_contrast2 = contrast2_dict["coordinates"]
        data_contrast1 = contrast1_dict["coordinates"]

        min1, max1 = data_contrast1.min(), data_contrast1.max()
        min2, max2 = data_contrast2.min(), data_contrast2.max()

        print(min1, max1)
        print(min2, max2)

        min_c, max_c = np.min(np.array([min1, min2])), np.max(np.array([max1, max2]))

        print(min_c, max_c)

        data_contrast1 = norm_grid(data_contrast1, xmin=min_c, xmax=max_c)
        data_contrast2 = norm_grid(data_contrast2, xmin=min_c, xmax=max_c)

        labels_contrast2 = contrast2_dict["intensity_norm"]
        labels_contrast1 = contrast1_dict["intensity_norm"]
        
        mask_contrast2 = contrast2_mask_dict["intensity_norm"].bool()
        mask_contrast1 = contrast1_mask_dict["intensity_norm"].bool()

        labels_contrast2_stack = torch.cat((labels_contrast2, torch.ones(labels_contrast2.shape)*-1), dim=1)
        labels_contrast1_stack = torch.cat((torch.ones(labels_contrast1.shape)*-1, labels_contrast1), dim=1)
        
        # assemble the data and labels
        self.data = torch.cat((data_contrast1, data_contrast2), dim=0)
        self.label = torch.cat((labels_contrast1_stack, labels_contrast2_stack), dim=0)
        self.mask = torch.cat((mask_contrast1, mask_contrast2), dim=0)
        self.len = len(self.label)

        # store the GT images to compute SSIM and other metrics!
        gt_contrast1_dict = get_image_coordinate_grid_nib(nib.load(str(self.gt_contrast1)))
        gt_contrast2_dict = get_image_coordinate_grid_nib(nib.load(str(self.gt_contrast2)))

        self.gt_contrast2 = gt_contrast2_dict["intensity_norm"]
        self.gt_contrast1 = gt_contrast1_dict["intensity_norm"]

        self.gt_contrast2_mask = torch.tensor(nib.load(self.gt_contrast2_mask).get_fdata()).bool()
        self.gt_contrast1_mask = torch.tensor(nib.load(self.gt_contrast1_mask).get_fdata()).bool()

        self.coordinates = gt_contrast1_dict["coordinates_norm"]

        self.affine = gt_contrast1_dict["affine"]
        self.dim = gt_contrast1_dict["dim"]

        print(self.data.shape)
        print(self.label.shape)

        # store to avoid preprocessing
        dataset = {
            'len': self.len,
            'data': self.data,
            'mask': self.mask,
            'label': self.label,
            'affine': self.affine,
            'gt_contrast1': self.gt_contrast1,
            'gt_contrast2': self.gt_contrast2,
            'gt_contrast1_mask': self.gt_contrast1_mask,
            'gt_contrast2_mask': self.gt_contrast2_mask,
            'dim': self.dim,
            'coordinates': self.coordinates,
        }
        if not os.path.exists(os.path.join(os.getcwd(), os.path.split(self.dataset_name)[0])):
            os.makedirs(os.path.join(os.getcwd(), os.path.split(self.dataset_name)[0]))
        torch.save(dataset, self.dataset_name)

class InferDataset(Dataset):
    def __init__(self, grid):
        super(InferDataset, self,).__init__()
        self.grid = grid

    def __len__(self):
        return len(self.grid)

    def __getitem__(self, idx):
        data = self.grid[idx]
        return data

if __name__ == '__main__':

    dataset = MultiModalDataset(
                image_dir='miccai',
                name='miccai_dataset',          
                )

    print("Passed.")
    
