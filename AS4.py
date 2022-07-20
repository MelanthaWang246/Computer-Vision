# QUESTION 1
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def readExtract(url):
    # read the file with specific columns
    data = pd.read_csv(url, usecols=[3, 4, 5], skiprows=50)
    # reset the name of each columns for easy writing afterward
    data.columns = ['petal length', 'petal width', 'class']
    return data
def split(data):
    # 将两个不同类的分开
    # 1 for versicolor, 2 for virginica
    classCol = data['class'].unique()
    data1 = data[data['class'].isin([classCol[0]])]
    data2 = data[data['class'].isin([classCol[1]])]
    # get the train set and test set for each category
    train1 = np.array(data1.iloc[:40,:])
    test1 = data1.iloc[40:,:]
    train2 = np.array(data2.iloc[:40,:])
    test2 = data2.iloc[40:,:]
    train = np.append(train1, train2, axis=0)
    test = np.append(test1, test2, axis=0)
    return train1, test1, train2, test2, train, test 
def plot(train1, train2):
    # 绘制散点图
    # set size
    plt.figure(figsize=(8, 8))
    # 1 for versicolor, 2 for virginica
    a1 = plt.scatter(train1[:, 0], train1[:, 1], marker='.', c='blue')
    a2 = plt.scatter(train2[:, 0], train2[:, 1] , marker='+', c='black')
    # set title
    plt.title('Scatter of petal length and width')
    # Label the axes
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    # set label explanation
    plt.legend([a1, a2], ['versicolor', 'virginica'])
    # Show the result
    plt.show()
def linear(set, k, b):
    #分成标签和值两类
    from sklearn.linear_model import LogisticRegression
    x_set = set[:,:2]
    row_set, column_set = set.shape
    row_set = int(row_set/2)
    # label matrix, 1 for versicolor, 2 for virginica
    y_set = np.append(np.ones((row_set,1)), np.ones((row_set,1))*2, axis=0)

    set_length = x_set[:, :1] # x
    set_width = x_set[:, 1:]  # y_origin
    row,column = set_length.shape
    # store the prediction result
    pred = np.zeros(y_set.shape) # y_prediction
    # store the number of point when on the line
    for r in range(row):
        result = k * set_length[r,0] + b
        if set_length[r,0] >=5.0:
            if set_width[r,0] >= result:
                # above & on the line
                pred[r,0] = 2
            elif set_width[r,0] < result:
                # under the line
                pred[r,0] = 1
        else:
            if set_width[r,0] > result:
                # above the line
                pred[r,0] = 2
            elif set_width[r,0] <= result:
                # under & on the line
                pred[r,0] = 1
    # calculate the number of mistakes and correct
    correct = 0
    wrong = 0
    for r in range(row):
        if pred[r,0] == y_set[r,0]:
            correct += 1
        else:
            wrong += 1
    return x_set, y_set, wrong

def optimum(learnRate, iterations, w, b, train1, train2, x_set, y_set, train, test):
    # Plot the figure
    # sample
    plt.figure(figsize=(8, 8))
    a1 = plt.scatter(train1[:, 0], train1[:, 1], marker='.', c='blue')
    a2 = plt.scatter(train2[:, 0], train2[:, 1] , marker='+', c='black')
    color = ['b','b','b','b','g','r','c','m','y','k']
    for i in range(iterations):
        w = w - learnRate * np.sum((y_set - (b + w * x_set)) * (-x_set))
        b = b - learnRate * np.sum((y_set - (b + w * x_set)) * (-1))
        temp1, temp2, error_train = linear(train, w, b)
        temp1, temp2, error_test = linear(test, w, b)
        print(i, "time: (", k, ",", b, ")", "train error: ", error_train, "test error: ", error_test)
        # line
        x = np.linspace(2,7, 100)
        y = w*x+b
        plt.plot(x, y, c=color[i])

    # title
    plt.title('Scatter of petal length and width')
    # axes
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    # label
    plt.legend([a1, a2], ['versicolor', 'virginica'])
    # Show
    plt.show()

if __name__ == '__main__': 
    url = r"D:\Finalpython\AS4\W04-IrisData.CSV"
    data = readExtract(url)
    train1, test1, train2, test2, train, test = split(data)
    k = -1
    b = 6.6
    x_train, y_train, wrong = linear(train, k, b)
    # optimum
    w, m = 1, 7
    iterations = 10
    learnRate = 0.0001 
    optimum(learnRate, iterations, w, m, train1, train2, x_train, y_train, train, test)
   
   
    # # Question 3 test
    # # randomly choose the k, b value
    # k = -1
    # b = 6.6
    # error_train = linear(train, k, b)
    # error_test = linear(test, k, b)
    # print("training set error: ", error_train)
    # print("test set error: ", error_test)
    # # Question 2 test
    # train1, test1, train2, test2, train, test = split(data)
    # plot(train1, train2)

    # # Question 1 test
    # # to show the file content
    # print(data)
    # # to show the content shape
    # print(data.shape)

