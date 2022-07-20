from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def readConvert(url):
    # read the file with specific columns
    r = np.array(pd.read_excel(url, usecols=[1]))
    g = np.array(pd.read_excel(url, usecols=[2]))
    b = np.array(pd.read_excel(url, usecols=[3]))
    # rgb = np.array([r, g, b])
    y = np.zeros(r.shape)
    cb = np.zeros(r.shape)
    cr = np.zeros(r.shape)
    # ycbcr = np.array([r, g, b])
    row, col = r.shape
    for o in range(row):
        y[o,0] = 16+0.257*r[o,0]+0.504*g[o,0]+0.098*b[o,0]
        cb[o,0] = 128-0.148*r[o,0]-0.291*g[o,0]+0.439*b[o,0]
        cr[o,0] = 128+0.439*r[o,0]-0.368*g[o,0]-0.071*b[o,0]
    return y, cb, cr

def plot(b_cb, b_cr, s_cb, s_cr):
    # 绘制散点图
    # set size
    plt.figure(figsize=(8, 8))
    # b for background, s for skin
    a1 = plt.scatter(b_cb, b_cr, marker='.', c='blue')
    a2 = plt.scatter(s_cb, s_cr , marker='+', c='red')
    # set title
    plt.title('Distribution')
    # Label the axes
    plt.xlabel('Cb value')
    plt.ylabel('Cr value')
    # set label explanation
    plt.legend([a1, a2], ['background', 'skin'])
    # Show the result
    plt.show() 

def meanCov(array, length):
    # mean np.mean
    sum1 = 0
    sum2 = 0
    for c in range(length):
        sum1 += array[0,c]
        sum2 += array[1,c]
    mean1 = sum1/length
    mean2 = sum2/length
    mean = np.array([[mean1], [mean2]])

    # covariance np.cov
    sumxx = 0
    sumyy = 0
    sumxy = 0
    for c in range(length):
        sumxx += pow((array[0,c] - mean[0,0]),2)
        sumxy += (array[0,c]-mean[0,0])*(array[1,c]-mean[1,0])
        sumyy += pow((array[1,c] - mean[1,0]),2)
    denominator = length - 1
    covxx = sumxx / denominator
    covxy = sumxy / denominator
    covyy = sumyy / denominator
    cov_matrix = np.array([[covxx,covxy],[covxy,covyy]])
    return mean, cov_matrix

def predict(array, mean, cov_matrix):
    # p formula
    cov_det = np.linalg.det(cov_matrix)
    cov_inv = np.linalg.inv(cov_matrix)
    array_temp = array-mean
    coe = 1.0 / (2.0*np.pi*np.sqrt(cov_det)) * np.exp(-1.0/2)
    p_before = coe * np.exp(np.dot(np.dot(array_temp.T,cov_inv),array_temp))

    row2, col2 = array.shape
    p_after = np.zeros((1,col2))
    # p_after will contains the real value for each pixel
    for i in range(col2):
        p_after[0,i] = p_before[i,i] 

    max = 0.0
    for i in range(col2):
        if(max < p_after[0,i]):
            max = p_after[0,i]
    # turn the probability range into [0,1]
    p_after = p_after / max

    return p_after

if __name__ == '__main__': 
    url1 = r"D:\Finalpython\AS4\backgroundRGB.xlsx"
    # url2 = r"D:\Finalpython\AS4\skinRGB.xlsx"
    # Question 1
    y_bgRGB, cb_bgRGB, cr_bgRGB = readConvert(url1)
    # y_sRGB, cb_sRGB, cr_sRGB = readConvert(url2)

    # row1, col1 = cb_sRGB.shape
    # skin = np.zeros((2, row1))
    # for c1 in range(row1):
    #     skin[0,c1] = cb_sRGB[c1,0]
    #     skin[1,c1] = cr_sRGB[c1,0]
    # mean1, cov_matrix1 = meanCov(skin, row1)
    # # print("mean", mean1)
    # # print("Covariance matrix", cov_matrix1)


    row1, col1 = cb_bgRGB.shape
    skin = np.zeros((2, row1))
    for c1 in range(row1):
        skin[0,c1] = cb_bgRGB[c1,0]
        skin[1,c1] = cr_bgRGB[c1,0]
    mean1, cov_matrix1 = meanCov(skin, row1)
    print("mean", mean1)
    print("Covariance matrix", cov_matrix1)

    # # x轴对应值
    # order = np.arange(row1)
    # # 调整为一维坐标方便绘图
    # p1 = np.zeros(order.shape)
    # for i in range(row1):
    #     p1[i] = temp1[0,i]

    # plt.figure(figsize=(6,4))
    # plt.plot(order, p1, c='red', linestyle='-')
    # plt.xlabel("skinRGB") 
    # plt.ylabel("probability")
    # plt.title("skin_probability")
    # plt.show()
