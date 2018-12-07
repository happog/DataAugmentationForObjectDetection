#%% [markdown]
# # Data Augmentation For Object Detection
# 
# This notebook serves as general manual to using this codebase. We cover all the major augmentations, as well as ways to combine them. 

#%%
from data_aug.data_aug import *
from data_aug.bbox_util import *
import numpy as np 
import cv2 
import matplotlib.pyplot as plt 
import pickle as pkl
get_ipython().run_line_magic('matplotlib', 'inline')

#%% [markdown]
# ## Storage Format 
# 
# First things first, we define how the storage formats required for images to work. 
# 1. **The Image**: A OpenCV numpy array, of shape *(H x W x C)*. 
# 2. **Annotations**: A numpy array of shape *N x 5* where *N* is the number of objects, one represented by each row. 5 columns represent the top-left x-coordinate, top-left y-coordinate, bottom-right x-coordinate, bottom-right y-coordinate, and the class of the object. 
# 
# Here is an image to aid your imagination. 
# 
# ![Annotation Format](ann_form.jpg)
# 
# Whatever format your annotations are present in make sure, you convert them to this format.
# 
# For demonstration purposes, we will be using the image above to show the transformations. The image as well as it's annotation has been provided. The annotation is a numpy array in a pickled format. 

#%%
img = cv2.imread("messi.jpg")[:,:,::-1]   #opencv loads images in bgr. the [:,:,::-1] does bgr -> rgb
bboxes = pkl.load(open("messi_ann.pkl", "rb"))

#inspect the bounding boxes
print(bboxes)

#%% [markdown]
# You can use the function `draw_rect` to plot the bounding boxes on an image. 

#%%
plotted_img = draw_rect(img, bboxes)
plt.imshow(plotted_img)
plt.show()

#%% [markdown]
# Now, we can get started with our image augmentations. The first one is **Horizontal Flipping**. The function takes one arguement, *p* which is the probability that the image will be flipped. 

#%%
img_, bboxes_ = RandomHorizontalFlip(1)(img.copy(), bboxes.copy())
plotted_img = draw_rect(img_, bboxes_)
plt.imshow(plotted_img)
plt.show()

#%% [markdown]
# **Scaling**. Scales the image. If the argument *diff* is True, then the image is scaled with different values in the vertical and the horizontal directions, i.e. aspect ratio is not maintained. 
# 
# If the first argument is a float, then the scaling factors for both x and y directions are randomly sampled from *(- arg, arg)*. Otherwise, you can specify a tuple for this range.

#%%
img_, bboxes_ = RandomScale(0.3, diff = True)(img.copy(), bboxes.copy())
plotted_img = draw_rect(img_, bboxes_)
plt.imshow(plotted_img)
plt.show()

#%% [markdown]
# **Translation**. Translates the image. If the argument *diff* is True, then the image is translated with different values in the vertical and the horizontal directions.
# 
# If the first argument is a float, then the translating factors for both x and y directions are randomly sampled from *(- arg, arg)*. Otherwise, you can specify a tuple for this range.

#%%
img_, bboxes_ = RandomTranslate(0.3, diff = True)(img.copy(), bboxes.copy())
plotted_img = draw_rect(img_, bboxes_)
plt.imshow(plotted_img)
plt.show()

#%% [markdown]
# **Rotation**. Rotates the image. 
# 
# If the first argument is a int, then the rotating angle, in degrees, is sampled from *(- arg, arg)*. Otherwise, you can specify a tuple for this range.

#%%
img_, bboxes_ = RandomRotate(20)(img.copy(), bboxes.copy())
plotted_img = draw_rect(img_, bboxes_)
plt.imshow(plotted_img)
plt.show()

#%% [markdown]
# **Shearing**. Sheares the image horizontally
# 
# If the first argument is a float, then the shearing factor is sampled from *(- arg, arg)*. Otherwise, you can specify a tuple for this range.

#%%
img_, bboxes_ = RandomShear(0.2)(img.copy(), bboxes.copy())
plotted_img = draw_rect(img_, bboxes_)
plt.imshow(plotted_img)
plt.show()

#%% [markdown]
# **Resizing**.  Resizes the image to square dimensions while keeping the aspect ratio constant.
# 
# The argument to this augmentation is the side of the square.

#%%
img_, bboxes_ = Resize(608)(img.copy(), bboxes.copy())
plotted_img = draw_rect(img_, bboxes_)
plt.imshow(plotted_img)
plt.show()

#%% [markdown]
# HSV transforms are supported as well. 

#%%
img_, bboxes_ = RandomHSV(100, 100, 100)(img.copy(), bboxes.copy())
plotted_img = draw_rect(img_, bboxes_)
plt.imshow(plotted_img)
plt.show()

#%% [markdown]
# You can combine multiple transforms together by using the Sequence class as follows. 

#%%
seq = Sequence([RandomHSV(40, 40, 30),RandomHorizontalFlip(), RandomScale(), RandomTranslate(), RandomRotate(10), RandomShear()])
img_, bboxes_ = seq(img.copy(), bboxes.copy())

plotted_img = draw_rect(img_, bboxes_)
plt.imshow(plotted_img)
plt.show()

#%% [markdown]
# A list of all possible transforms can be found in the `docs` folder.
# 
# 
# 
# 
# 

