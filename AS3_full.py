# QUESTION 1
from matplotlib import pyplot as plt
import numpy as np
import cv2

def sobelY(img):
    # set the matrix of Gy
    G_y = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])
    # get the value of rows and columns of the input image
    row, column = img.shape
    # set an empty matrix with the same shape of input image to store result
    result_y = np.zeros(img.shape)
    
    for r in range(row-2): # range: [0, row-3]
        for c in range(column-2): # range: [0, column-3]
            # iterate the image with the size 3*3
            result_y[r+1, c+1] = abs(np.sum(G_y * img[r:r+3, c:c+3]))
            '''Because some value is greater than 255, 
               which will lead to black after setting the type np.uint8.
               If use the value weight to arrange the pixel value, 
               the image will become darker.
               Hence, set them to 255 '''
            if result_y[r+1, c+1] >= 255:
                result_y[r+1, c+1] = 255
    # Otherwise the pixel will become white becasue of data loss
    return np.uint8(result_y) 

def sobelX(img):
    # set the matrix of Gx
    G_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    # get the value of rows and columns of the input image
    row, column = img.shape
    # set an empty matrix with the same shape of input image to store result
    result_x = np.zeros(img.shape)
    
    for r in range(row-2): # range: [0, row-3]
        for c in range(column-2): # range: [0, column-3]
            # iterate the image with the size 3*3
            result_x[r+1, c+1] = abs(np.sum(G_x * img[r:r+3, c:c+3]))
            '''Because some value is greater than 255, 
               which will lead to black after setting the type np.uint8.
               If use the value weight to arrange the pixel value, 
               the image will become darker.
               Hence, set them to 255 '''
            if result_x[r+1, c+1] >= 255:
                result_x[r+1, c+1] = 255

    return np.uint8(result_x)

def sobelXY(img, sobelx, sobely):
    # get the value of rows and columns of the input image
    row, column = img.shape
    # set an empty matrix with the same shape of input image to store result
    result_xy = np.zeros(img.shape)
    '''Because some values are greater than 255, 
       and to avoid the image too much brigher, 
       we use the value weight to rearrange the pixel value, 
       rather than set them to 255 when getting sobelx and sobely'''
    max = 0
    for r in range(row-2): # [0, row-3]
        for c in range(column-2): #[0, column-3]
            # iterate the sobelx and sobely one by one, from 1 to row-2 or column-2
            result_xy[r+1, c+1] = np.sqrt(sobelx[r+1, c+1]**2 + sobely[r+1, c+1]**2)
            if result_xy[r+1, c+1] >= max:
                max = result_xy[r+1, c+1]
    # rearrange the pixel value according to value / maxinum * 255
    # for the final value of a pixel, range: [0, 255]
    for r in range(row-2): # [0, row-3]
        for c in range(column-2): #[0, column-3]
            result_xy[r+1, c+1] = round(result_xy[r+1, c+1] / max * 255)
    return np.uint8(result_xy)

def binaryImg(threshould, sobelxy):
    # get the value of rows and columns of the input image
    row, column = img.shape
    # set an empty matrix with the same shape of input image to store result
    result_bi = np.zeros(sobelxy.shape)
    # iterate the pixel one by one
    # if the pixel value > threshould, set it to 1, otherwise, set it to 0
    for r in range(0, row):
        for c in range(0, column):
            if sobelxy[r, c] < threshould:
                result_bi[r, c] = 0
            else:
                result_bi[r, c] = 1
    return result_bi

if __name__ == '__main__': 
    path = r"D:\Finalpython\AS3\Lenna.png"
    img = cv2.imread(path, flags=0)
    # img1 = np.pad(img,1,'constant')
    # cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    thValue = 127
    # sobelx = sobelX(img)
    sobely = sobelY(img)
    # sobelxy = sobelXY(img, sobelx, sobely)

    # binary = binaryImg(thValue, sobely)


    # #openCV提供的直接使用的方法→用于检查是否正确
    # sobel_x = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
    sobel_y = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
    # sobel_x = cv2.convertScaleAbs(sobel_x)
    sobel_y = cv2.convertScaleAbs(sobel_y)
    # sobel_xy = cv2.addWeighted(sobel_x,0.5,sobel_y,0.5,0)
    # _, th = cv2.threshold(sobel_y, thValue, 255, 0)

    # cv2.imshow('Before', img) 
    # cv2.imshow('sobelx', sobelx)
    cv2.imshow('sobely', sobely)
    # cv2.imshow('sobelxy', sobelxy)
    # cv2.imshow('shouldbe_x', sobel_x)
    cv2.imshow('shouldbe_y', sobel_y)
    # cv2.imshow('shouldbe_xy', sobel_xy)

    # cv2.imshow('binary', binary)
    # cv2.imshow('sholdbe_bi', th)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
