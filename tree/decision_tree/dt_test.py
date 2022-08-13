# -*- coding:utf-8 -*-


import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt


def main():
    # 加载数据集
    X, y = load_breast_cancer(return_X_y=True)
    print('shape(X): ', np.array(X).shape, 'shape(y): ', np.array(y).shape)
    # 划分训练集、测试集
    # X_train=(426, 30), X_test=(143, 30)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    print('X_train: ', np.array(X_train).shape)
    print('X_test: ', np.array(X_test).shape)
    print('y_train: {}, y_test: {}'.format(
        np.array(y_train).shape, np.array(y_test).shape))

    # 建树
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, y_train)
    # 预估每个类别概率
    print(clf.predict_proba(X_test))
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(acc)
    # 决策树可视化
    plt.figure(figsize=(15, 10))
    tree.plot_tree(clf, filled=True)
    # plt.show()
    plt.savefig('./tree_fig.png')

    # 剪枝
    # 获取alpha及每个alpha下子树叶子纯度
    path = clf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    # 可视化ccp_alphas与ccp_alphas
    fig, ax = plt.subplots()
    ax.plot(ccp_alphas, impurities, marker="o", drawstyle="steps-post")
    ax.set_xlabel('ccp_alphas')
    ax.set_ylabel('impurities')
    ax.set_title('Total Impurity vs alpha for training set')
    # plt.show()
    # plt.savefig('./ccp_alphas.png')

    # 根据alpha获取子树序列
    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        clf.fit(X_train, y_train)
        clfs.append(clf)
    # 最后一个子树只有一个叶节点，可以忽略
    clfs = clfs[:-1]
    ccp_alphas = ccp_alphas[:-1]
    # 查看每棵子树的叶节点个数和深度
    node_counts = [clf.tree_.node_count for clf in clfs]
    depths = [clf.tree_.max_depth for clf in clfs]

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
    ax[0].set_xlabel("alpha")
    ax[0].set_ylabel("number of nodes")
    ax[0].set_title("Number of nodes vs alpha")
    ax[1].plot(ccp_alphas, depths, marker="o", drawstyle="steps-post")
    ax[1].set_xlabel("alpha")
    ax[1].set_ylabel("depth of tree")
    ax[1].set_title("Depth vs alpha")
    fig.tight_layout()
    plt.savefig('./depth_node.png')

    train_scores = [clf.score(X_train, y_train) for clf in clfs]
    test_scores = [clf.score(X_test, y_test) for clf in clfs]
    for ccp_alpha, test_score in zip(ccp_alphas, test_scores):
        print(ccp_alpha, test_score)

    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs alpha for training and testing sets")
    ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
    ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
    ax.legend()
    plt.savefig('./acc.png')
    # plt.show()

    final_tree = DecisionTreeClassifier(random_state=0, ccp_alpha=0.012)
    final_tree.fit(X_train, y_train)
    res = final_tree.predict(X_test)
    print(accuracy_score(y_test, res))


if __name__ == '__main__':
    main()
