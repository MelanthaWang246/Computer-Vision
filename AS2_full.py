import cv2
import numpy as np
from matplotlib import pyplot as plt
def calDrawHist(channel, color):  
    # hist= cv2.calcHist([array_before], [0], None, [256], [0.0,255.0])  
    # create a matrix to hold histogram value
    histogram = np.zeros(256, dtype=np.uint64)

    # add value to each column in the histogram
    row, col = channel.shape
    for r in range(row):
        for c in range(col):
            histogram[channel[r, c]] += 1

    # Draw & Show the histogram
    plt.figure(color)
    plt.plot(histogram, color=color)
    plt.show()

def Gamma(array, c, a):
    # Normalization不然<1基本全白色，>1都是噪点
    array = array / 255.0
    gamma = c * np.power(array, a)
    gamma = np.uint8(gamma * 255.0)
    return gamma
    
def equalization(img):
    # Get the histogtam value of each pixel value
    histogram = np.zeros(256, dtype=np.uint64)
    row, col = img.shape
    for r in range(row):
        for c in range(col):
            histogram[img[r, c]] += 1
    # get the cumulative array of each pixel value in the histogram
    cumulative_histogram = np.cumsum(histogram)
    # function of the equalization
    function = 255 *  cumulative_histogram / (img.size - 1)
    # process the image
    row, col = img.shape
    equal = np.zeros(img.shape, dtype='uint64') 
    for r in range(row):
        for c in range(col):
            equal[r,c] = function[img[r, c]]

    equal = equal.astype(np.uint8) #otherwise, it will show a image opening error
    return equal

if __name__ == '__main__': 
    path = r"D:\Finalpython\AS2\W02-HE-source.jpg"
    # read the image from the specific path
    img = cv2.imread(path, flags = 0)
    # Equalization
    equal = equalization(img)

    # show the histogram of the original image and image after equalization
    histImgBefore = calDrawHist(img, "black")
    histImgAfter = calDrawHist(equal, "blue")
    # show the original image and image after equalization
    cv2.imshow('Before', img) 
    cv2.imshow('After', equal)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # # read the image from the specific path
    # path = r"D:\Finalpython\AS2\W02-HE-source.jpg"
    # img = cv2.imread(path)
    # # seperate into b,g,r three channel if it is RGB image
    # b, g, r = cv2.split(img)  
  
    # histImgB = calDrawHist(b, "blue")  
    # histImgG = calDrawHist(g, "green")  
    # histImgR = calDrawHist(r, "red")  

    # # Gamma
    # gamma = Gamma(img,1,0.9)