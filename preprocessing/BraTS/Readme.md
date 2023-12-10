### How to preprocess the BRATS data

If you want to work with BRATS, you need to apply for the data on the [website](https://www.med.upenn.edu/cbica/brats2020/registration.html). While the datasets are open source/public, they still require registration to obtain the data. The incoming data structure when we download the BRATS data looks like this:

```
(base) jqm@latitude:~/Downloads$ tree MICCAI_BraTS_2019_Data_Training/ -L 1
MICCAI_BraTS_2019_Data_Training/
├── HGG
├── LGG
├── name_mapping.csv
├── Readme.md
└── survival_data.csv

2 directories, 3 files
```

And for the scans in HGG, e.g. like this:

```
...
├── BraTS19_CBICA_AAB_1
│   ├── BraTS19_CBICA_AAB_1_flair.nii.gz (240,240,155)
│   ├── BraTS19_CBICA_AAB_1_seg.nii.gz (240,240,155)
│   ├── BraTS19_CBICA_AAB_1_t1ce.nii.gz (240,240,155)
│   ├── BraTS19_CBICA_AAB_1_t1.nii.gz (240,240,155)
│   └── BraTS19_CBICA_AAB_1_t2.nii.gz (240,240,155)
├── BraTS19_CBICA_AAG_1
│   ├── BraTS19_CBICA_AAG_1_flair.nii.gz (240,240,155)
│   ├── BraTS19_CBICA_AAG_1_seg.nii.gz (240,240,155)
│   ├── BraTS19_CBICA_AAG_1_t1ce.nii.gz (240,240,155)
│   ├── BraTS19_CBICA_AAG_1_t1.nii.gz (240,240,155)
│   └── BraTS19_CBICA_AAG_1_t2.nii.gz (240,240,155)
├── BraTS19_CBICA_AAL_1
```

### Step 0: Reduce the dataset

As we had originally planned to do some testing with BRATS and the FLAIR modality, we decided to only use scans
that visually had a 3D FLAIR (and not an upsampled 2D FLAIR). Thus, we decided to include the following patients in our paper:

```
(base) jqm@latitude:~/Downloads/NeRF_BRATS_data$ tree -L 1
.
├── BraTS19_CBICA_AAB_1
├── BraTS19_CBICA_AAG_1
├── BraTS19_CBICA_AAL_1
├── BraTS19_CBICA_AAM_1
├── BraTS19_CBICA_ANI_1
├── BraTS19_CBICA_ANK_1
├── BraTS19_CBICA_ANP_1
├── BraTS19_CBICA_ANV_1
├── BraTS19_CBICA_ANZ_1
├── BraTS19_CBICA_ATW_1
├── BraTS19_CBICA_AUC_1
├── BraTS19_CBICA_AUE_1
├── BraTS19_CBICA_AUN_1
├── BraTS19_CBICA_AUQ_1
├── BraTS19_CBICA_AUR_1
├── BraTS19_CBICA_AXJ_1
├── BraTS19_CBICA_AXL_1
├── BraTS19_CBICA_AXM_1
├── BraTS19_CBICA_AXN_1
├── BraTS19_CBICA_AXO_1
├── BraTS19_CBICA_AXQ_1
├── BraTS19_CBICA_AXW_1
├── BraTS19_CBICA_AZA_1
├── BraTS19_CBICA_AZD_1
└── BraTS19_CBICA_AZH_1
```

### Step 1: Obtaining the brainmask

All images in BratTS19 are fully co-registered and skull-stripped already. However, the brainmask has not been exported.
As all background voxels are zero,we can obtain the mask by using non-zero values as a threshold. We do this in `get_mask.py`.
To obtain the mask we use all "*t1.nii.gz" as references and run:

`find /path/to/reduced/data -type f -name "*t1.nii.gz" -exec python3 get_mask.py --image {} --identifier_in t1.nii.gz  --identifier_out brainmask.nii.gz \;`

Now your directory should look like this:
```
.
├── BraTS19_CBICA_AAB_1_brainmask.nii.gz Shape:(240,240,155)
├── BraTS19_CBICA_AAB_1_flair.nii.gz Shape:(240,240,155)
├── BraTS19_CBICA_AAB_1_seg.nii.gz Shape:(240,240,155)
├── BraTS19_CBICA_AAB_1_t1ce.nii.gz Shape:(240,240,155)
├── BraTS19_CBICA_AAB_1_t1.nii.gz Shape:(240,240,155)
└── BraTS19_CBICA_AAB_1_t2.nii.gz Shape:(240,240,155)
0 directories, 6 files
```

### Step 2: Padding the images

To have nicer resulting batch sizes, we pad all images to (240, 240, 160). We can also additionally reduce the images sizes, as "the content" (i.e. brain) is vastly smaller, however, we are skipping this for now to avoid another point of failure.

`find /path/to/reduced/data -type f -name "*.nii.gz" -exec python3 pad.py --image {} \;`

Now your directory should look like this, note the new shapes:
```
.
├── BraTS19_CBICA_AAB_1_brainmask.nii.gz Shape:(240,240,160)
├── BraTS19_CBICA_AAB_1_flair.nii.gz Shape:(240,240,160)
├── BraTS19_CBICA_AAB_1_seg.nii.gz Shape:(240,240,160)
├── BraTS19_CBICA_AAB_1_t1ce.nii.gz Shape:(240,240,160)
├── BraTS19_CBICA_AAB_1_t1.nii.gz Shape:(240,240,160)
└── BraTS19_CBICA_AAB_1_t2.nii.gz Shape:(240,240,160)
0 directories, 6 files
```

### Step 3: Downsampling to obtain LR images

For BRATS we use high in-plane resolution for T1w  to obtain axial T1w, and for T2w we decide to use cor T2w.
You may experiment with other combinations of contrasts and resolutions as well.

To obtain the images, just run the following two commands:

```
find /path/to/reduced/data -type f -name "*t1.nii.gz" -exec python3 downsample.py --image {} --view axial \;
find /path/to/reduced/data -type f -name "*t2.nii.gz" -exec python3 downsample.py --image {} --view coronal \;
```

Now your dataset directory should look like the following:
```
.
├── BraTS19_CBICA_AAB_1_brainmask.nii.gz (240,240,160)
├── BraTS19_CBICA_AAB_1_flair.nii.gz (240,240,160)
├── BraTS19_CBICA_AAB_1_seg.nii.gz (240,240,160)
├── BraTS19_CBICA_AAB_1_t1ce.nii.gz (240,240,160)
├── BraTS19_CBICA_AAB_1_t1.nii.gz (240,240,160)
├── BraTS19_CBICA_AAB_1_t1_LR.nii.gz (240,240,40)
└── BraTS19_CBICA_AAB_1_t2.nii.gz (240,240,160)
└── BraTS19_CBICA_AAB_1_t2_LR.nii.gz (240,60,160)
0 directories, 6 files
```

### Step 4: Obtaining the brain masks for the LR scans

To obtain the brain masks, use:

`find /path/to/reduced/data -type f -name "*t1_LR.nii.gz" -exec python3 get_mask.py --image {} --identifier_in t1_LR.nii.gz  --identifier_out t1_mask_LR.nii.gz \;`

`find /path/to/reduced/data -type f -name "*t2_LR.nii.gz" -exec python3 get_mask.py --image {} --identifier_in t2_LR.nii.gz  --identifier_out t2_mask_LR.nii.gz \;`

Tadaaa. Finally, you should have arrived at the resulting structure:

```
(base) jqm@latitude:~/Downloads/MICCAI_BraTS_2019_Data_Training/instructions/BraTS19_CBICA_AAB_1$ tree
.
├── BraTS19_CBICA_AAB_1_brainmask.nii.gz (240,240,160)
├── BraTS19_CBICA_AAB_1_flair.nii.gz (240,240,160) -> we don't use this file
├── BraTS19_CBICA_AAB_1_seg.nii.gz (240,240,160) -> we don't use this file
├── BraTS19_CBICA_AAB_1_t1ce.nii.gz (240,240,160) -> we don't use this file
├── BraTS19_CBICA_AAB_1_t1_LR.nii.gz (240,240,40) -> we use this as LR Contrast 1
├── BraTS19_CBICA_AAB_1_t1_mask_LR.nii.gz (240,240,40) -> we use this as LR Contrast Mask 1
├── BraTS19_CBICA_AAB_1_t1.nii.gz (240,240,160) -> we use this as GT Contrast 1
├── BraTS19_CBICA_AAB_1_t2_LR.nii.gz (240,60,160) -> we use this as LR Contrast 2
├── BraTS19_CBICA_AAB_1_t2_mask_LR.nii.gz (240,60,160) -> we use this as LR Contrast Mask 2
└── BraTS19_CBICA_AAB_1_t2.nii.gz (240,240,160) -> we use this as GT Contrast 2
0 directories, 10 files
```
