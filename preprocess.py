import nibabel as nib
import numpy as np
from PIL import Image
import os


def convert_nii_to_png(nii_file, output_folder):
    # Load the NIfTI file
    image_data = nib.load(nii_file).get_fdata()

    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Normalize and save each slice as a PNG file
    for i in range(image_data.shape[2]):
        slice = image_data[:, :, i]
        # Normalize the slice to 0-255
        slice_255 = ((slice - np.min(slice)) / (np.max(slice) - np.min(slice))) * 255.0
        slice_255 = slice_255.astype(np.uint8)
        # Convert to PIL image and save
        Image.fromarray(slice_255).save(os.path.join(output_folder, f'slice_{i:03d}.png'))


def batch_convert_nii_to_png(input_folder, output_folder_base):
    # Walk through the input folder
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.nii.gz') and not file.startswith('._'):
                print(f'Processing {file}...')
                file_path = os.path.join(root, file)
                # Construct a directory path for the outputs
                relative_path = os.path.relpath(root, input_folder)
                file_output_folder = os.path.join(output_folder_base, relative_path,
                                                  os.path.splitext(os.path.splitext(file)[0])[0])
                convert_nii_to_png(file_path, file_output_folder)




input_folder = r'C:\Users\lee\Desktop\Task06_Lung\Task06_Lung\labelsTr'
output_folder_base = r'C:\Users\lee\Desktop\wangzhengshuaitask\masktr'

batch_convert_nii_to_png(input_folder, output_folder_base)
