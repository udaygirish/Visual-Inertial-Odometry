import numpy as np 
import cv2
import os 
import sys 
from tqdm import tqdm
import torch

def normalize_image(image):
    # Normalize the image
    image = image/255.0
    return image

def LoadImages(folder, no_of_images=None):
    images = []

    # Get a list of files look at sorting at later
    #files = sorted(os.listdir(folder), key=lambda x: int(x.split("_")[1].split(".")[0]))
    print("First index", int(os.listdir(folder)[0].split(".")[0]))
    list_files = os.listdir(folder)
    # Keep only .png or .jpg files
    list_files = [f for f in list_files if f.endswith('.png') or f.endswith('.jpg')]
    files = sorted(list_files, key=lambda x: int(x.split(".")[0]))
    # Iterate and Load Image files 
    for i_no, img_path  in tqdm(enumerate(files)):
        if no_of_images is not None:
            if i_no >= no_of_images:
                break
        # Read the image
        tmp = cv2.imread(os.path.join(folder, img_path))
        if tmp is not None:
            # Convert the image to black and white
            #tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
            tmp = cv2.resize(tmp, (512, 512))
            # Normalize the image
            tmp = normalize_image(tmp)
            images.append(tmp)
    return images

def BatchGen(images, IMU, pose, minibatch):
    # Generate batches of Data
    batches = []
    total_batches = int(len(images)/minibatch)
    for i in range(total_batches):
        batched_images = []
        batched_IMU = []
        batched_pos_q = []
        batched_pos_t = []
        rand_ind = np.random.choice(len(images) - 1, minibatch, replace=False)
        for idx in rand_ind:

            # PREPARE FOR IMAGES
            current_image = images[idx]
            next_image = images[idx + 1]
            merged_image = np.stack((current_image, next_image), axis=2)
            torch_merged_image = torch.from_numpy(merged_image).float()
            torch_merged_image = torch_merged_image.view(6,512,512)
            batched_images.append(torch_merged_image.unsqueeze(0))

            # PREPARE FOR IMU
            current_IMU = IMU[idx*10:(idx+1)*10, 1:] #100 -> 10 changed
            torch_current_IMU = torch.from_numpy(current_IMU).float()
            torch_current_IMU = torch_current_IMU.reshape(10,6)  # 100,6 --> 10,6
            batched_IMU.append(torch_current_IMU.unsqueeze(0))

            # LABELS PREPARATION
            current_pose_t = pose[idx, 1:4]
            current_pose_q = pose[idx, 4:]
            torch_current_pose_t = torch.from_numpy(current_pose_t).float()
            torch_current_pose_q = torch.from_numpy(current_pose_q).float()

            batched_pos_t.append(torch_current_pose_t.unsqueeze(0))
            batched_pos_q.append(torch_current_pose_q.unsqueeze(0))

        batched_images = torch.cat(batched_images, dim=0)  # Concatenating all the images in the Batch
        batched_IMU = torch.cat(batched_IMU, dim=0) # Concatenating all the IMU in the Batch

        batched_pos_t = torch.cat(batched_pos_t, dim=0)  # Concatenating all the Translation in the Batch
        batched_pos_q = torch.cat(batched_pos_q, dim=0)  # Concatenating all the Quaternion in the Batch

        batch = [batched_images, batched_IMU, batched_pos_t, batched_pos_q] 
        batches.append(batch)

    return batches