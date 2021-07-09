# -*- coding: utf-8 -*-
"""
Created on Mon Feb 05 16:56:53 2018

@author: luolei

Ref: Unpack Local Model Interpretation for GBDT
"""

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from sklearn.datasets import load_breast_cancer



##########################################################################################
#### Local Interpretation GBDT ###########################################################
##########################################################################################


def calPosAndNegSampleNumOnEachTree(decision_regression_tree, test_sample, X_train, y_train):
    '计算每个节点不同划分下的样本数和正样本分数'
    decision_path_node_series = decision_regression_tree.decision_path(test_sample)
    decision_paths_for_the_sample = list()
    s = 0
    for i in range(decision_path_node_series.shape[1]):
        if decision_path_node_series[0, i] == 1:
            s += 1
            decision_paths_for_the_sample.append(i)


    '对所有训练样本依次执行decision_path_for_the_sample中的分裂操作'
    '正负样本数'
    num_pos_and_neg_sample = dict()
    '分裂步数'
    num_pos_and_neg_sample[0] = [0, 'start', decision_regression_tree.tree_.n_node_samples[0], float(len([i for i in y_train if i ==1])) / len(y_train)]
    
    for i in range(s-1):
        if i == 0:
            X_train = X_train
            y_train = y_train
        else:
            X_train = np.array(X_1_selected).reshape(len(X_1_selected),  X_train.shape[1])
            y_train = np.array(y_1_selected).reshape(len(y_1_selected), 1)
        
        feature_num = decision_regression_tree.tree_.feature[decision_paths_for_the_sample[i]] 
        criter = decision_regression_tree.tree_.threshold[decision_paths_for_the_sample[i]] 

        # 可以更新一下。
        tmp_X_1_selected = X_train[X_train[:, feature_num] <= criter]
        tmp_y_1_selected = y_train[X_train[:, feature_num] <= criter]
        tmp_X_2_selected = X_train[X_train[:, feature_num] > criter]
        tmp_y_2_selected = y_train[X_train[:, feature_num] > criter]

        # 先判断左右，再看往哪里走。
        feature_sample = decision_regression_tree.tree_.n_node_samples[decision_paths_for_the_sample[i+1]]
        if len(tmp_X_1_selected) == int(feature_sample):
            X_1_selected = tmp_X_1_selected
            y_1_selected = tmp_y_1_selected
        else:
            X_1_selected = tmp_X_2_selected
            y_1_selected = tmp_y_2_selected

        # 纠正一些东西。
        '正样本分数'
        if len(y_1_selected) == 0:
            scores_1 = None
        else:
            scores_1 = [p for p in y_1_selected].count(1) / len(y_1_selected)
        
        num_pos_and_neg_sample[decision_paths_for_the_sample[i+1]] = [i+1, feature_num, len(X_1_selected), scores_1, scores_1 - num_pos_and_neg_sample[decision_paths_for_the_sample[i]][3]]
    return num_pos_and_neg_sample

def findAllPosAndNegNumInGBDT(GBDT, test_sample, X_train, y_train, n_estimators):
    '寻找GBDT模型所有决策树中节点分裂的样本数和正样本分数'
    pos_and_neg_num_in_all_trees = list()
    for i in range(n_estimators):
        print('this is tree %d' % i)
        gbdt_sub = GBDT[i, 0]
        pos_and_neg_num_in_all_trees.append(calPosAndNegSampleNumOnEachTree(gbdt_sub, test_sample, X_train, y_train))
    return pos_and_neg_num_in_all_trees

def main():
    cancer = load_breast_cancer()
    X_train = cancer.data
    y_train = cancer.target
    test_sample = X_train[0].reshape((1, -1))

    # GBDT参数设置
    n_estimators = 100
    max_depth = 3
    n_feature = X_train.shape[1]

    # GBDT学习
    GBDT = GradientBoostingClassifier(n_estimators = n_estimators, max_depth=max_depth)

    GBDT.fit(X_train, y_train)
    tree_weights = [0.01] * n_estimators
    ## 计算残差权重

    # 计算 Local FC
    pos_and_neg_num_in_all_trees = findAllPosAndNegNumInGBDT(GBDT, test_sample, X_train, y_train, n_estimators)
    score_per_feature_for_sample = {}
    for i in range(n_feature):
        score_per_feature_for_sample[i] = 0
    
    for num_tree in range(n_estimators):
        for key, val in pos_and_neg_num_in_all_trees[num_tree].items():
            if key != 0:
                score_per_feature_for_sample[val[1]] += tree_weights[num_tree] * val[4]


    score_per_feature_for_sample = sorted(score_per_feature_for_sample.items(), key = lambda item:item[1],  reverse = True)
    df = pd.DataFrame(score_per_feature_for_sample, columns=['feature', 'FC'])
    df.to_csv("score_history.csv", sep=',', index=False, encoding='utf_8_sig')

    # Local FC 作图
    values = list()
    keys = list()
    for i in range(len(score_per_feature_for_sample)):
        values.append(score_per_feature_for_sample[i][1])
        keys.append(score_per_feature_for_sample[i][0])

    plt.figure()
    plt.bar(range(len(list(score_per_feature_for_sample))), values, width = 0.5)
    plt.xticks(range(len(list(score_per_feature_for_sample))),
               keys,
               rotation=45)
    plt.title('Local FC')
    plt.xlabel('feature name')
    plt.ylabel('FC value')
    plt.savefig('ss.png')
    return


if __name__ == "__main__":
    main()

