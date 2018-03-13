# -*- coding: utf-8 -*-
import numpy as np
import cv2
from keras.preprocessing.image import img_to_array
from random import randint


def binarylab(labels, dims, n_labels):
    x = np.zeros([dims[0], dims[1], n_labels])
    for i in range(dims[0]):
        for j in range(dims[1]):
            x[i, j, labels[i][j]]=1
    x = x.reshape(dims[0] * dims[1], n_labels)
    return x


# generator that we will use to read the data from the directory
def data_gen_small(img_dir, mask_dir, lists, batch_size, dims, n_labels):
        while True:
            ix = np.random.choice(np.arange(len(lists)), batch_size)
            imgs = []
            labels = []
            for i in ix:
                # images
                original_img = cv2.imread(img_dir + str(lists.iloc[i, 0]) + ".jpg")[:, :, ::-1]
                resized_img = cv2.resize(original_img, dims)
                array_img = img_to_array(resized_img)
                
                # masks
                original_mask = cv2.imread(mask_dir + str(lists.iloc[i, 0]) + '.png')
                resized_mask = cv2.resize(original_mask, (dims[0], dims[1]))
                
                #Flip randomly images and masks
                orientation = randint(0, 4)
                if orientation == 0: #horizontal
                    array_img = cv2.flip(array_img, 0)
                    resized_mask = cv2.flip(resized_mask, 0)
                elif orientation == 1: #vertical
                    array_img = cv2.flip(array_img, 1)
                    resized_mask = cv2.flip(resized_mask, 1)
                elif orientation == 2: #horizontal and vertical
                    array_img = cv2.flip(array_img, -1)
                    resized_mask = cv2.flip(resized_mask, -1)
                elif orientation == 3: #none
                    pass
                
                # Convert mask to labels
                array_mask = binarylab(resized_mask[:, :, 0], dims, n_labels)
                # append image and mask to others
                imgs.append(array_img)
                labels.append(array_mask)
                
            imgs = np.array(imgs)
            labels = np.array(labels)
            yield imgs, labels