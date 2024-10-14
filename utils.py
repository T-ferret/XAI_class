import os
from PIL import Image
import numpy as np
import torch
from matplotlib import pyplot as plt


def get_test_imgs(test_dir, preprocess):
    dir_list = os.listdir(test_dir)
    y_test = []

    for i, img_name in enumerate(dir_list):
        split_name = img_name.split('.')
        label = split_name[0]
        y_test.append(label)
        img_loc = os.path.join(test_dir, img_name)
        img = Image.open(img_loc).convert('RGB')

        if i == 0:
            X_test = preprocess(img).unsqueeze(0)
        else:
            X_test = torch.cat((X_test, preprocess(img).unsqueeze(0)), 0)

    return X_test, np.array(y_test)


def preprocess_orgimgs(org_images, input_size):
    org_imgs = []

    for img in org_images:
        img = np.array(img.resize((232, 232)))
        crop_size = int((img.shape[0] - input_size) / 2)

        # center crop
        org_imgs.append(img[crop_size:232-crop_size, crop_size:232-crop_size])

    return np.array(org_imgs)


def denomrmalize(images, mean, std):
    denorm_imgs = np.array(images)
    denorm_imgs = denorm_imgs * std + mean
    denorm_imgs = denorm_imgs.transpose(0, 2, 3, 1).clip(0, 1)

    return denorm_imgs


def get_class_name(weights, labels):
    label_names = []

    for label in labels:
        label_name = weights.meta['categories'][label]
        label_names.append(label_name)

    return label_names


def plot_images(images, labels):
    fig, axes = plt.subplots(nrows=1, ncols=len(images), figsize=(20, 5), sharex=True, sharey=True)

    for i in range(len(images)):
        axes[i].imshow(images[i])
        axes[i].axis('off')
        axes[i].set_title(labels[i])
    plt.show()


def visualize_gradients(org_imgs, pred_names, gradients):

    fig, axes = plt.subplots(nrows=2, ncols=len(org_imgs), figsize=(20, 8), sharex=True, sharey=True)
    for i in range(len(org_imgs)):
        axes[0, i].imshow(org_imgs[i])
        axes[0, i].set_title(pred_names[i])
        axes[0, i].axis('off')
        axes[1, i].imshow(gradients[i], cmap='jet', alpha=1)
        axes[1, i].axis('off')

    plt.show()