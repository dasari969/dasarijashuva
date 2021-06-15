# -*- coding: utf-8 -*-
"""
Created on Mon May 10 22:25:38 2021

@author: vinod
"""
import numpy as np
import pandas as pd
import cv2
from skimage import io
from PIL import Image 
import matplotlib.pylab as plt


## A)) TOO Visuali
img = "723604_photo.jpg"
my_img = cv2.imread(img)

plt.imshow(cv2.cvtColor(my_img, cv2.COLOR_BGR2RGB))
plt.title(" COLOR IMAGE OF MINE")

## B))  Convert it into a gray-level image 
gray_img = cv2.cvtColor(my_img , cv2.COLOR_BGR2GRAY)
plt.imshow(gray_img, cmap='gray')
plt.title(" GRAY IMAGE OF MINE")

## Convert gray-level image into nparray

img_mat = np.array(list(gray_img), float) 
print(img_mat)
img_mat.shape

plt.imshow(img_mat)

## C))  Perform the singular value decomposition of the 2D array

# scale the image matrix befor SVD
img_mat_scaled= (img_mat-img_mat.mean())/img_mat.std()

# Perform SVD using np.linalg.svd
U, S, V = np.linalg.svd(img_mat_scaled)

# Compute Variance explained by each singular vector
var_explained = np.round(S**2/np.sum(S**2), decimals=4)

import seaborn as sns
from matplotlib import style
style.use("classic")
sns.barplot(x=list(range(1,31)), y=var_explained[0:30], color="dodgerblue")
plt.xlabel('Singular Vector', fontsize=10)
plt.ylabel('Variance Explained', fontsize=10)
plt.tight_layout()
plt.title(" Singular value decompsition")
style.available

### D))	Then perform the image compression (low rank approximation) by choosing 10% top SVDs, 
###               20% top SVDs, 50% SVDs. Visualize all these results.

#Reconstruction with top 10 singular values
num_components = 10
reconst_img_10 = np.array(U[:, :num_components]) .dot(np.diag(S[:num_components]).dot(np.array(V[:num_components, :])))
plt.imshow(reconst_img_10)

#Reconstruction with top 20 singular values
num_components = 20
reconst_img_20 = np.array(U[:, :num_components]) .dot(np.diag(S[:num_components]).dot(np.array(V[:num_components, :])))
plt.imshow(reconst_img_20)

#Reconstruction with top 50 singular values
num_components = 50
reconst_img_50 = np.array(U[:, :num_components]) .dot(np.diag(S[:num_components]).dot(np.array(V[:num_components, :])))
plt.imshow(reconst_img_50)

#Reconstruction with top 100 singular values
num_components = 100
reconst_img_100 = np.array(U[:, :num_components]) .dot(np.diag(S[:num_components]).dot(np.array(V[:num_components, :])))
plt.imshow(reconst_img_100)


fig, axs = plt.subplots(2, 2,figsize=(10,10))
axs[0, 0].imshow(reconst_img_10)
axs[0, 0].set_title('Reconstructed Image: 10 SVs', size=16)
axs[0, 1].imshow(reconst_img_20)
axs[0, 1].set_title('Reconstructed Image: 20 SVs', size=16)
axs[1, 0].imshow(reconst_img_50)
axs[0, 1].set_title('Reconstructed Image: 50 SVs', size=16)
axs[1, 0].imshow(reconst_img_100)
axs[0, 1].set_title('Reconstructed Image: 100 SVs', size=16)
plt.tight_layout()


