import argparse
import nibabel as nib
import numpy as np
import os

def zero_pad_nifti_image(input_image_path, output_image_path):
    # Load the NIfTI image
    img = nib.load(input_image_path)

    # Check if the image has the initial dimensions (240, 240, 155)
    assert img.shape == (240, 240, 155), f"Image {input_image_path} does not have dimensions (240, 240, 155)"

    # Get the current data and affine
    data = img.get_fdata()
    affine = img.affine

    # Calculate the padding size
    target_shape = (240, 240, 160)
    pad_size = [(0, 0) if data.shape[i] >= target_shape[i] else (0, target_shape[i] - data.shape[i]) for i in range(3)]

    # Zero-pad the data
    padded_data = np.pad(data, pad_size, mode='constant')

    # Create a new NIfTI image with the padded data
    padded_img = nib.Nifti1Image(padded_data, affine)

    # Save the padded NIfTI image with the same filename
    nib.save(padded_img, output_image_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-pad a NIfTI image to (240, 240, 160) dimensions.")
    parser.add_argument("--image", required=True, help="Path to the input NIfTI image.")
    args = parser.parse_args()

    input_image_path = args.image
    output_image_path = input_image_path

    zero_pad_nifti_image(input_image_path, output_image_path)
    print(f"Image {input_image_path} zero-padded and saved as {output_image_path}.")
