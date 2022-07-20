from re import T
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def readSplit(url):
    # read the file with specific columns
    data = pd.read_csv(url, usecols=[1,2,3,4,5], skiprows=50)
    data.columns = ['sepal length', 'sepal width','petal length', 'petal width', 'class']
    classCol = data['class'].unique()
    data1 = data[data['class'].isin([classCol[0]])]
    data2 = data[data['class'].isin([classCol[1]])]
    # get the train set and test set for each category
    train1 = np.array(data1.iloc[:40,:])
    test1 = np.array(data1.iloc[40:,:])
    train2 = np.array(data2.iloc[:40,:])
    test2 = np.array(data2.iloc[40:,:])

    row_train, col_train = train1.shape
    row_test, col_test = test1.shape

    for r1 in range(row_train):
        train1[r1,4] = 1
        train2[r1,4] = -1
    for r2 in range(row_test):
        test1[r2,4] = 1
        test2[r2,4] = -1

    train = np.append(train1, train2, axis=0)
    test = np.append(test1, test2, axis=0)
    # 将特征值和label分离成两个二维数组
    x_train, y_train = np.hsplit(train,[-1]) 
    x_test, y_test = np.hsplit(test,[-1]) 

    return  x_train, y_train, x_test, y_test
def normalize(array, column):
    row, col = array.shape
    sum = 0
    # 求mean
    for r1 in range(row):
        sum += array[r1,column]
    mean = sum / row
    # normalization: value-mean
    for r2 in range(row):
        array[r2,column] = array[r2, column] - mean
    return array

# def initial():
#     w = np.random.normal(loc=0.0, scale=1.0, size=(4,1))
#     t = np.random.normal(loc=0.0, scale=1.0, size=None)
#     return w,t
def activation(w,t,x):
    result = x.dot(w)+t
    y = 0
    if result >= 0:
        y = 1
    else:
        y = -1
    return y

def update(w,t,x_set, y_set, lr):
    y1 = 0
    row, col = x_set.shape
    for i in range(row):
        x = x_set[i,:]
        y = y_set[i,0]
        y1 = activation(w,t,x)
        error_train = 0
        if y1 != y:
            error_train += 1
            w = w + lr * (y-y1) * x.reshape((4,1))
            t = t + lr * (y-y1)
            break
    return w, t, error_train
def testAccu(w,t,x_set, y_set):
    row, col = x_set.shape
    for i in range(row):
        x = x_set[i,:]
        y = y_set[i,0]
        y1 = activation(w,t,x)
        error_test = 0
        if y1 != y:
            error_test += 1
    return error_test
if __name__ == '__main__': 
    url = r"D:\Finalpython\AS4\W06-IrisData.CSV"
    x_train, y_train, x_test, y_test = readSplit(url)
    x_train = normalize(x_train, 0)
    x_train = normalize(x_train, 1)
    x_train = normalize(x_train, 2)
    x_train = normalize(x_train, 3)
    x_test = normalize(x_test, 0)
    x_test = normalize(x_test, 1)
    x_test = normalize(x_test, 2)
    x_test = normalize(x_test, 3)

    w = np.array([[1.833912],[-0.36264566],[0.70323396],[-0.01003392]])
    t = -0.7724395926300087
    # Question 3
    # learning rate
    lr = np.array([0.001, 0.005, 0.01])
    epoch = 5

    error_train = np.zeros((3,5))
    error_test = np.zeros((3,5))
    accuracy_train = np.zeros((3,5))
    accuracy_test = np.zeros((3,5))
    row1, col1 = x_train.shape
    row2, col2 = x_test.shape

    for r in range(3):
        for e in range(epoch):
            w1, t1, error_train[r,e] = update(w,t,x_train, y_train, lr[r])
            error_test[r,e] = testAccu(w1, t1, x_test, y_test)
            accuracy_train[r,e] = 1 - error_train[r,e] / row1
            accuracy_test[r,e] = 1 - error_test[r,e] / row2
            print(w1)
    print('train set accuracy: ')
    print(accuracy_train)
    print('test set accuracy: ')
    print(accuracy_test)
    plt.plot(accuracy_train, lr, label='train')
    plt.plot(accuracy_test, lr, label='test')
    plt.xlabel("epoch")
    plt.ylabel("learning rate")
    plt.legend()
    plt.show()


    # # Question 1
    # # use the test as an example to show the success of the two functions
    # print("x_test")
    # print(x_test)
    # print("y_test")
    # print(y_test)
    # print("shape of x_test: ", x_test.shape)
    # print("shape of y_test: ", y_test.shape)
    # # show the shape of x_train and y_train
    # print("shape of x_train: ", x_train.shape)
    # print("shape of y_train: ", y_train.shape)

    # Question 2
    # w, t = initial()
    # print('w: ', w)
    # print('t: ', t)

    # sample = np.array([7, 3.2, 4.7, 1.4])
    # y = activation(w,t,sample)
    # print('y value of the sample given: ', y)