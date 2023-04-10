import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils

img=cv2.imread('image1.png')
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(cv2.cvtColor(gray,cv2.COLOR_BGR2RGB))
# plt.savefig('my_figure.png')
bfilter= cv2.bilateralFilter(gray,11,17,17) #noise reduction
edged = cv2.Canny(bfilter, 30, 200) #egde detection
plt.imshow(cv2.cvtColor(edged,cv2.COLOR_BGR2RGB))
# plt.savefig('my_figure.png')
keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx)==4:
        location = approx
        break
# print(location) 
mask = np.zeros(gray.shape, np.uint8)
new_image=cv2.drawContours(mask, [location], 0,225,-1)
new_image = cv2.bitwise_and(img, img,mask=mask)
plt.imshow(cv2.cvtColor(new_image,cv2.COLOR_BGR2RGB))
plt.savefig('lp.png')

