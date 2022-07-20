import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skewnorm

def readConvert(r,g,b):
    col = len(r)
    y = np.zeros(r.shape)
    cb = np.zeros(r.shape)
    cr = np.zeros(r.shape)
    
    for o in range(col):
        y[o] = 16+0.257*r[o]+0.504*g[o]+0.098*b[o]
        cb[o] = 128-0.148*r[o]-0.291*g[o]+0.439*b[o]
        cr[o] = 128+0.439*r[o]-0.368*g[o]-0.071*b[o]
    return y, cb, cr

def predict(array, mean, cov_matrix):
    # p formula
    cov_det = np.linalg.det(cov_matrix)
    cov_inv = np.linalg.inv(cov_matrix)
    array_temp = array-mean
    coe = 1.0 / (2.0*np.pi*np.sqrt(cov_det)) * np.exp(-1.0/2)
    #size: 38804*38804
    p_before = coe * np.exp(np.dot(np.dot(array_temp.T,cov_inv),array_temp))
    
    row2, col2 = array.shape
    # 1*38804
    p_after = np.zeros(col2)
    for i in range(col2):
        p_after[i] = p_before[i,i] 
    max = 0.0
    for i in range(col2):
        if(max < p_after[i]):
            max = p_after[i]
    # [0,1]
    p_after = p_after / max

    return p_after

def reshape(p):
    image = np.zeros(p.shape)
    image = image.reshape((218,178))
    return image

if __name__ == '__main__': 
    path = r"D:\Finalpython\AS4\test.jpg"
    img = cv2.imread(path)
    b,g,r = cv2.split(img)
    b = b.flatten()
    g = g.flatten()
    r = r.flatten()
    y, cb, cr = readConvert(r,g,b)

    # 调整为二维数组, 行0为cb, 行1为cr, 方便计算
    col = len(cb)
    array = np.array([cb,cr])
    # skin
    mean1 = np.array([[109.73134227],[150.51660748]])
    cov_matrix1 = np.array([[61.21311916, -58.40563063],[-58.40563063, 80.40434063]])

    # background
    mean2 = np.array([[129.31829351],[130.35454377]])
    cov_matrix2 = np.array([[146.3295919, -214.58252164],[-214.58252164, 577.66244108]])
    
    # use two heap maps to show skin and background probabilities of each pixel respectively
    p_s = predict(array, mean1, cov_matrix1)
    pic_s = reshape(p_s)
    p_bg = predict(array, mean2, cov_matrix2)
    pic_bg = reshape(p_bg)

    plt.figure()
    im_s = plt.imshow(pic_s, cmap=plt.get_cmap('hot'), interpolation='nearest', vmin=0, vmax=1) 
    plt.colorbar(im_s, shrink=0.2)
    plt.show()

    plt.figure()
    im_bg = plt.imshow(pic_bg, cmap=plt.get_cmap('hot'), interpolation='nearest', vmin=0, vmax=1) 
    plt.colorbar(im_bg, shrink=0.2)
    plt.show()

    img_gray = pic_s * 255
    cv2.imshow('gray', img_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
