import nibabel as nib
import numpy as np
import argparse

def threshold_nifti(input_path, output_path):
    # Load the NIfTI file
    nifti_image = nib.load(input_path)
    image_data = nifti_image.get_fdata()
    # Apply thresholding
    thresholded_data = np.where(image_data >= 1, 1, 0).astype(np.int8)

    # Create a new NIfTI image
    new_nifti_image = nib.Nifti1Image(thresholded_data, nifti_image.affine, nifti_image.header)

    # Save the new NIfTI file
    nib.save(new_nifti_image, output_path)

def main():
    parser = argparse.ArgumentParser(description="Threshold a NIfTI image.")
    parser.add_argument("--image", required=True, help="Path to the input NIfTI image")
    parser.add_argument("--identifier_in", required=True, default="t1.nii.gz", help="Identifier for the input file")
    parser.add_argument("--identifier_out", required=True, default="brainmask.nii.gz", help="Identifier for the output file")

    args = parser.parse_args()

    input_path = args.image
    output_path = input_path.replace(args.identifier_in, args.identifier_out)

    print(f"Thresholding {input_path} and saving as {output_path}")

    threshold_nifti(input_path, output_path)

if __name__ == "__main__":
    main()
