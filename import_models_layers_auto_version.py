"""Volumetric map
"""

from __future__ import annotations
from datetime import datetime
import torch
import os
import re
import numpy as np
import torchvision

from rsatoolbox.rdm import calc_rdm
from rsatoolbox.data import Dataset
from torchvision import transforms
from PIL import Image
from surgeon_pytorch import get_layers, Inspect
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jobcontext import JobContext
import nibabel as nib


# from nilearn import plotting


# image loader
def image_loader(image_address):
    img = Image.open(image_address)
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = transform(img)

    return img


# Function to extract numeric parts from the filename
def extract_numbers(filename):
    # Use regular expression to find all numbers in the filename
    numbers = re.findall(r'\d+', filename)
    # Convert the found numbers to integers
    return [int(number) for number in numbers]


def main(job: JobContext):
    # add the desired model
    model_name = job.data['model']['title']
    git_id = job.data['model']['gitID']
    model = torch.hub.load(git_id, model_name, pretrained=True)

    # use eval mode for feeding the model
    model.eval()
    layers = [job.data['layer']['key']]

    # defining wrapper for hooking activations
    model_wrapped = Inspect(model, layer=layers, keep_output=False)
    model_wrapped.eval()

    # generate a random tensor to represent an image
    random_tensor = torch.rand(3, 224, 224)  # (batch_size, channels, height, width)

    # preprocess the tensor to match the input requirements of the model
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Apply the preprocessing to the random tensor
    input_tensor = preprocess(random_tensor)

    with torch.no_grad():
        output = model_wrapped(torch.unsqueeze(input_tensor, 0))

    # store the dimensions that are required for hooking activations
    list_of_dims = []
    for i, e in enumerate(output):
        list_of_dims.append(torch.squeeze(e).numpy().shape)
        print('Layer ', i, ' shape is ', torch.squeeze(e).numpy().shape)

    # define arrays dimemsions
    num_arrays = len(output)
    first_dimension = len(job.data['images'])
    print('Size of stimulus set is ', first_dimension)

    job.outputPath.joinpath(str(model_name))
    path = job.outputPath / str(model_name) / 'activations'

    try:
        os.makedirs(path)
    except OSError as e:
        print(f"Creation of the directory {path} failed")
    else:
        print(f"Successfully created the directory {path}")

    # initialize the memmap arrays for storing activations locally
    for i in range(num_arrays):
        mmapped_array = np.memmap(f'{path}/{layers[i]}.dat', dtype=np.float32,
                                  mode='w+',
                                  shape=(first_dimension, *list_of_dims[i]))
        mmapped_array.flush()
        del mmapped_array

    # define the path to NSD100 images
    path_to_images = str(job.outputPath / 'images')

    # get a list of all image files in the directory
    image_files = job.data['images']

    # Sort filenames using the extracted numeric parts
    sorted_filenames = sorted(image_files, key=extract_numbers)
    count = 0

    # feeding the model with images and hook the activations
    with torch.no_grad():
        print('Layers activations are hooking ...')
        for c, img in enumerate(sorted_filenames):
            image = image_loader(path_to_images + '/' + img)
            image_batch = torch.unsqueeze(image, 0)
            layer_act = model_wrapped(image_batch)

            for j in range(num_arrays):
                # Load the memory-mapped array for this iteration
                mmapped_array = np.memmap(
                    f'{path}/{layers[j]}.dat', dtype=np.float32,
                    mode='r+', shape=(first_dimension, *list_of_dims[j]))
                mmapped_array[c, ...] = np.squeeze(layer_act[j])
                print(mmapped_array[c, ...].shape)
                print('sec', np.squeeze(layer_act[j]).shape)
                mmapped_array.flush()
                del mmapped_array
            del layer_act

            if c % 10 == 0:
                print(c, ' images passed to Alexnet')
        print('Layers activations are hooked.')

    # free up the RAM
    for i in range(num_arrays):
        mmapped_array = np.memmap(f"{path}/{layers[i]}.dat",
                                  dtype=np.float32, mode='r+', shape=(first_dimension, *list_of_dims[i]))
        del mmapped_array

    rdms_path = job.outputPath / str(model_name) / 'rdms'

    try:
        os.makedirs(rdms_path)
    except OSError as e:
        print(f"Creation of the directory {rdms_path} failed")
    else:
        print(f"Successfully created the directory {rdms_path}")

    # Calculate the RDMs of selected layers
    for i in range(num_arrays):
        # Load the activation array for the current layer
        act_arr = np.memmap(
            f'{path}/{layers[i]}.dat',
            dtype=np.float32, mode='r', shape=(first_dimension, *list_of_dims[i])
        )

        print(f"Processing layer {i}: shape {act_arr.shape}")

        # Convert the numpy array to rsatoolbox Dataset
        data = act_arr.reshape(first_dimension, -1)  # Flatten the spatial dimensions
        dataset = Dataset(data)

        # Calculate the RDM using rsatoolbox
        rdms = calc_rdm(dataset)

        # Save the RDM to a .npy file
        np.save(f'{rdms_path}/{layers[i]}.npy', rdms.get_matrices())

        # Flush and delete the memmap object
        act_arr._mmap.close()
        del act_arr
        del rdms

    # free up the RAM
    for i in range(num_arrays):
        mmapped_array = np.memmap(f"{path}/{layers[i]}.dat",
                                  dtype=np.float32, mode='r+', shape=(first_dimension, *list_of_dims[i]))
        del mmapped_array

    # Loading brain mask
    mask_path = job.data['local_rdm_data']['mask']
    brain_mask_nii = nib.load(mask_path)
    brain_mask_data = brain_mask_nii.get_fdata()

    # Loading searchlight centers
    center_path = job.data['local_rdm_data']['centers']
    centers_linear = np.load(center_path)  # Changed to np.load
    centers = np.array(np.unravel_index(centers_linear, brain_mask_data.shape)).T

    # Loading NSD100 RDMS
    rdms_path = job.data['local_rdm_data']['rdms']
    nsd_rdms = np.load(rdms_path)  # Changed to np.load

    alexnet_rdms_dir = job.outputPath.joinpath(f'{model_name}/rdms')
    alexnet_rdm_files = sorted([f for f in os.listdir(alexnet_rdms_dir) if f.endswith('.npy')])

    for idx, rdm_file in enumerate(alexnet_rdm_files):
        alexnet_rdm_path = os.path.join(alexnet_rdms_dir, rdm_file)
        alexnet_rdm = np.load(alexnet_rdm_path)

        # Check if alexnet_rdm is 100x100 and convert to upper triangle flattened
        alexnet_rdm = np.squeeze(alexnet_rdm)
        if alexnet_rdm.shape == (100, 100):
            alexnet_rdm = alexnet_rdm[np.triu_indices(100, k=1)]

        # Calculate correlation between AlexNet RDM and NSD RDMs
        correlations = np.zeros(len(centers))
        for i, center in enumerate(centers):
            print('Correlation of center number: ', i)
            nsd_rdm = nsd_rdms[i]
            correlations[i] = np.corrcoef(alexnet_rdm.flatten(), nsd_rdm.flatten())[0, 1]

        volume = np.zeros(brain_mask_data.shape)

        # Assign correlation values to corresponding centers
        for i, center in enumerate(centers):
            print('Assigning corr val to center number: ', i)
            volume[tuple(center)] = correlations[i]

        # Create a new NIfTI image from the volume
        volume_nii = nib.Nifti1Image(volume, brain_mask_nii.affine, brain_mask_nii.header)

        result_path = job.outputPath.joinpath(f'{model_name}/sub04')
        try:
            os.makedirs(result_path)
        except OSError as e:
            print(f"Creation of the directory {result_path} failed")
        else:
            print(f"Successfully created the directory {result_path}")
        output_path = job.outputPath.joinpath(f'{model_name}/sub04/rdm_layer_{rdm_file}.nii.gz')
        job.files.append(output_path)

        nib.save(volume_nii, output_path)

