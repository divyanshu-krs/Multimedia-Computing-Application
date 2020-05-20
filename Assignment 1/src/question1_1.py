import numpy as np
import cv2
import sys
from scipy import sqrt, pi, arctan2, cos, sin
from scipy.ndimage import uniform_filter
import os

#reference - https://github.com/New-generation-hsc/PPractice/blob/641950c27303820c2441ea87ae184458cfdf0dd1/features/getColorFeature.py
#reference - Content Based Image Retrieval System Using Auto Color Correlogram (Journal of Computer Applications (JCA,2013))

def point_exist(X, Y, point):

    if point[0] < 0 or point[0] >= X:
        return 0
    if point[1] < 0 or point[1] >= Y:
        return 0
    return 1
 

def correlogram(image, num_clr, K):

    X, Y,t= image.shape
 
    colorsPercent = []

    for k in K:
        countColor = 0
        color = []
        for i in num_clr:
           color.append(0)
        jump_x = int(round(X/10))  
        jump_y = int(round(Y / 10)) 
        for x in range(0, X, jump_x):

            for y in range(0, Y, jump_y):
                Ci = image[x][y]
                up_right_corner = (x + k, y + k)
                right = (x + k, y)
                low_right_corner = (x + k, y - k)
                low = (x, y - k)
                low_left_corner = (x - k, y - k)
                left = (x - k, y)
                up_left_corner = (x - k, y + k)
                up = (x, y + k)
                points = (up_right_corner, right, low_right_corner, low , low_left_corner, left, up_left_corner, up)
                total_points = []
                for i in points:
                  if point_exist(X, Y, i) == 1:
                    total_points.append(i)
                    
                for t in total_points:
                    pt = image[t[0]][t[1]]
                    for m in range(len(num_clr)):
                        if np.array_equal(num_clr[m], Ci) and np.array_equal(num_clr[m], pt):
                            countColor = countColor + 1
                            color[m] = color[m] + 1

        for i in range(len(color)):
            color[i] = float(color[i]) / countColor
        
        colorsPercent.append(color)

    return colorsPercent


def autoCorrelogram(img):

    Z = img.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    num_clusters = 32
    ret, label, center = cv2.kmeans(Z, num_clusters, None,criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    original_img = np.uint8(center)
    res = original_img[label.flatten()]
    res2 = res.reshape((img.shape))
    if_uni = np.array(res)

    K = [ ]
    for i in range(1, 9, 2):
      K.append(i)

    order = np.lexsort(if_uni.T)
    if_uni = if_uni[order]
    diff = np.diff(if_uni, axis = 0)
    ui = np.ones(len(if_uni), 'bool')
    ui[1:] = (diff != 0).any(axis = 1)
    colors32 = if_uni[ui]

    result = correlogram(res2, colors32, K)
    return result


def check_distance(x, y):
    
    x = np.array(x)
    y = np.array(y)
    return np.sqrt(np.sum((x - y) ** 2))

def list_image_from_query(direc):
    image_query = []
    for filename in os.listdir(direc):
        if filename.endswith('.txt'):
            
            file = open('/home/div/32/MCA/Assignment 1/HW-1/train/query/' + filename,'r')
            read = file.read()
            read = read.split(' ')
            image = read[0][5:]
            image_query.append(image)
    return (image_query)             

        
    
def find_corr_query_image(image_q_list):
    image_corr = []
    for i in image_q_list:
        img = cv2.imread("/home/div/32/MCA/Assignment 1/HW-1/images/"+ i+".jpg")
        print('home/div/32/MCA/Assignment 1/HW-1/images/'+ i+'.jpg')
        #img.show()
        img_crr = autoCorrelogram(img)
        image_corr.append(img_crr)
    return image_corr


def list_of_images(direc):
    images = []

    for filename in os.listdir(direc):
        images.append(filename)
    return images    

def find_corr_images(images):
    corr = []
    for i in images:
        img = cv2.imread("/home/div/32/MCA/Assignment 1/img/"+ i)
        print('home/div/32/MCA/Assignment 1/HW-1/images/'+ i+'.jpg')
        #img.show()
        img_crr = autoCorrelogram(img)
        corr.append(img_crr)
    with open("correlogram_data.txt", "wb") as fp:
        pickle.dump(corr, fp)    
    return corr
    

import pickle
im = cv2.imread("t.jpg")
im2 = cv2.imread("/home/div/32/MCA/Assignment 1/HW-1/images/ashmolean_000269.jpg")
img_from_q_list = list_image_from_query('/home/div/32/MCA/Assignment 1/HW-1/train/query')
img_corr = find_corr_query_image(img_from_q_list)

#with open("query_correlogram.txt", "wb") as fp:
#    pickle.dump(img_corr, fp)

image_list = list_of_images('/home/div/32/MCA/Assignment 1/img')
corr = find_corr_images(image_list)




#a = autoCorrelogram(im)
#b = autoCorrelogram(im2)
#print(check_distance(a,b))
