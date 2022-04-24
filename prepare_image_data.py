import pandas as pd 
import glob
import cv2

img_path = './img_noisy.png'

xcords = []
ycords = []

patch = 256
frame1 = cv2.imread(img_path)

for i in range(0,7168-patch,patch):
	for j in range(0,4608-patch,patch):
		img_nm = img_path.split('/')[-1]
		cropImg1 = frame1[j:j+patch,i:i+patch]
		cv2.imwrite('/data/Nam/input/'+str(i)+'_'+str(j)+'.png',cropImg1)
		print('/data/Nam/input/'+str(i)+'_'+str(j)+'.png')