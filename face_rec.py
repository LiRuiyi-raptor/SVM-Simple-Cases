"""
SVM 人脸识别
"""
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns

def svm():
    faces = fetch_lfw_people(min_faces_per_person=60)
    print(faces.target_names)
    print(faces.images.shape)
    ## 绘制图像
    fig, ax = plt.subplots(3,5)
    for i, axi in enumerate(ax.flat):
        axi.imshow(faces.images[i], cmap = 'bone')
        axi.set(xticks = [], yticks = [], xlabel = faces.target_names[faces.target[i]])
    plt.show()

    ## 每个图的大小是[62x47]
    ## 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(faces.data, faces.target, test_size = 0.2, random_state = 40)
    ## 使用 PCA 降维，降为 150 维
    ## whiten： 白化，就是对降维后的数据的每个特征进行标准化，让方差都为1。
    pca = PCA(n_components=150, whiten=True, random_state=42)
    ## 实例化 svm
    svc = SVC(kernel='rbf', class_weight='balanced')
    model = make_pipeline(pca, svc)
    ## GridSearchCV 确定参数
    param = {'svc__C':[1,5,10],
             'svc__gamma':[0.0001,0.0005,0.001]}
    grid = GridSearchCV(model,param_grid =param)
    grid.fit(x_train, y_train)
    print(grid.best_params_)

    model=grid.best_estimator_
    y_predict = model.predict(x_test)
    print(y_predict.shape)

    #算法分类之后的图形
    fig, ax = plt.subplots(4,6)
    for i, axi in enumerate(ax.flat):
        axi.imshow(x_test[i].reshape(62,47),cmap='bone')
        axi.set(xticks=[],yticks=[])
        axi.set_ylabel(faces.target_names[y_predict[i]].split()[-1],
                       color='black' if y_predict[i] == y_test[i] else 'red')

    fig.suptitle('Predicted Names:Incorrect Labels in Red',size=14)
    plt.show()

    print(classification_report(y_test, y_predict, target_names=faces.target_names))

    #混淆矩阵
    mat = confusion_matrix(y_test, y_predict)
    sns.heatmap(mat.T,square=True,annot=True,fmt='d',cbar=False,
                xticklabels=faces.target_names,
                yticklabels=faces.target_names)
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.show()

    return None

if __name__ == "__main__":
    svm()
