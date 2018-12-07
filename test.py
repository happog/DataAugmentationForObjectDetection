from data_aug.data_aug import *
from data_aug.bbox_util import *
import cv2 
import pickle as pkl
import numpy as np 
import matplotlib.pyplot as plt


img = cv2.imread("messi.jpg")[:,:,::-1] #OpenCV uses BGR channels
bboxes = pkl.load(open("messi_ann.pkl", "rb"))
print(bboxes)


# transforms = Sequence([RandomHorizontalFlip(1), RandomScale(0.08, diff = True), RandomRotate(2)])
transforms = Sequence([RandomTranslate(0.1)])
# transforms = Sequence([RandomScale(0.08, diff = True)])

img, bboxes = transforms(img, bboxes)

plt.imshow(draw_rect(img, bboxes))
plt.show()
