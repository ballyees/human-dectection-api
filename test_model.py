from model import detector
import cv2
import matplotlib.pyplot as plt
img = cv2.imread('test_img.jpg')
response = detector.run(img, isRGB=False, thresh=0.2)
img_res = response[-1]
plt.imsave('res.png', img_res)