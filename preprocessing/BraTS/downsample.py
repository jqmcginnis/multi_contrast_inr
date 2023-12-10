import nibabel as nib
import nibabel.processing as nip
import numpy as np
import argparse

# Define the resampling factors for different views
resampling_factors = {
    'axial': (1, 1, 4),
    'coronal': (1, 4, 1),
    'sagittal': (4, 1, 1),
}

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

    print(new_shp)
    new_aff = nib.affines.rescale_affine(aff, shp, voxel_spacing, new_shp)
    new_img = nip.resample_from_to(img, (new_shp, new_aff), order=order, cval=-1024)
    print("[*] Image resampled to voxel size:", voxel_spacing)
    return new_img

def resample_and_save(input_file, output_file, view):
    # Load the NIfTI image
    img = nib.load(input_file)

    # Get the desired voxel spacing based on the view
    voxel_spacing = resampling_factors.get(view.lower())

    if voxel_spacing is None:
        print("Invalid view specified. Use '--view axial', '--view sagittal', or '--view coronal'")
        return

    # Resample the image
    new_img = resample_nib(img, voxel_spacing=voxel_spacing)

    nib.save(new_img, output_file)
    print(f"Resampled image saved as {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resample a NIfTI image based on the specified view.")
    parser.add_argument("--image", help="Input NIfTI image file (e.g., t1.nii.gz)")
    parser.add_argument("--view", choices=['axial', 'sagittal', 'coronal'], required=True, help="Specify the view for downsampling")

    args = parser.parse_args()

    output_file = str(args.image).replace(".nii.gz", "_LR.nii.gz")
    resample_and_save(input_file=args.image, output_file=output_file, view=args.view)
