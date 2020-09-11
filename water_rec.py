#-*- coding:utf-8 -*-

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt


# -*- coding:utf-8 -*-
def cm_plot(y, yp):
    cm = confusion_matrix(y, yp)  # 混淆矩阵
    plt.matshow(cm, cmap=plt.cm.Greens)  # 画混淆矩阵图，配色风格使用cm.Greens，更多风格请参考官网。
    plt.colorbar()  # 颜色标签

    for x in range(len(cm)):  # 数据标签
        for y in range(len(cm)):
            plt.annotate(cm[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')

    plt.ylabel('True label')  # 坐标轴标签
    plt.xlabel('Predicted label')  # 坐标轴标签
    return plt


if __name__ == "__main__" :
    path = 'data/moment.csv'
    data = pd.read_csv(path, encoding = 'gbk')
    data = data.values

    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(data[:,2:],data[:,0],test_size=0.2)
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    print(y_train)
    print(y_test)

    # 构建 SVM 模型, 参数可自行搜索
    model = SVC()
    # 放大特征
    model.fit(x_train*30, y_train)
    # 混淆矩阵
    cm_train = confusion_matrix(y_train, model.predict(x_train * 30))
    cm_test = confusion_matrix(y_test, model.predict(x_test * 30))
    print(cm_train)
    print(cm_test)
    cm_plot(y_train, model.predict(x_train * 30)).show()
    cm_plot(y_test, model.predict(x_test * 30)).show()

    train_accuracy = accuracy_score(y_train,model.predict(x_train * 30))
    test_accuracy = accuracy_score(y_test,model.predict(x_test * 30))

    print("train accuracy: %f"% train_accuracy)
    print("test accuracy: %f"% test_accuracy)

