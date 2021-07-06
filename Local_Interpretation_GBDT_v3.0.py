# -*- coding: utf-8 -*-
"""
Created on Mon Feb 05 16:56:53 2018

@author: luolei

Ref: Unpack Local Model Interpretation for GBDT
"""
from __future__ import division
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
import random
import seaborn as sns

from sklearn.externals import joblib
from sklearn import tree
import pydotplus
from scipy.sparse import csr_matrix
import operator
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

##########################################################################################
#### functions ###########################################################################
##########################################################################################

def fillNan(x):
    for column in x.columns:
        x[column].where(x[column].notnull(), 0.0, inplace = True)

##########################################################################################
#### data ################################################################################
##########################################################################################

'指定训练集和测试集'
X_train = pd.read_csv('X_train.csv', encoding = 'utf-8', sep = ',')
y_train = pd.read_csv('y_train.csv', encoding = 'utf-8', sep = ',')
X_test_2 = pd.read_csv('X_test_2.csv', encoding = 'utf-8', sep = ',')

'数据缺失值补全'
fillNan(X_train)
fillNan(X_test_2)

'指定特定个体'
test_sample = np.array(X_train.loc[4000]).reshape(1, 201)
X_train = np.array(X_train).reshape(len(X_train), 201)
y_train = np.array(y_train).reshape(len(y_train), 1)


#### GBDT参数设置 ————————————————————————————————————————————————————————————————————————
n_estimators = 600
max_depth = 3
num_features = X_train.shape[1]

#### GBDT学习 ————————————————————————————————————————————————————————————————————————————
GBDT = GradientBoostingClassifier(n_estimators = n_estimators, learning_rate = 0.01, min_samples_split = 250,min_samples_leaf = 60,max_depth = max_depth, subsample = 1,random_state = 10,loss = "deviance")

GBDT.fit(X_train, y_train)
feature_importances = GBDT.feature_importances_

##########################################################################################
#### Local Interpretation GBDT ###########################################################
##########################################################################################

def extractingNodesOnEachDecisionTree(dot_data):
    '从决策树结构dot_data里面提取各节点对应的特征名字, 节点编号以及节点的连接关系'
    '记录下开头为 "0 [" 形式的序号'
    nodes = dict()
    data_1 = list(dot_data.split(';'))
    connection_num = list()
    for i in range(len(data_1)):
        if ' ' in data_1[i]:
            if data_1[i][data_1[i].index(' ') + 1] == '[':
                nodes[data_1[i][1 : data_1[i].index(' ')]] = data_1[i][data_1[i].index(' ') + 9 : (data_1[i][1 : ].index('\\n') + 1)]
            elif data_1[i][data_1[i].index(' ') + 1] == '-':
                connection_num.append(i)

    connections = list()
    for i in connection_num:
        space_loc = list()
        for j in range(len(data_1[i])):
            if data_1[i][j] == ' ':
                space_loc.append(j)
        connections.append(data_1[i][1 : space_loc[2]])
    return nodes, connections

def calPosAndNegSampleNumOnEachTree(decision_regression_tree, test_sample, X_train, y_train, max_depth):
    '计算每个节点不同划分下的样本数和正样本分数'
    dot_data = tree.export_graphviz(decision_regression_tree, out_file=None)
    '提取该决策树中的节点以及节点间的连接关系'
    nodes, connections = extractingNodesOnEachDecisionTree(dot_data)
    '寻找该样本在此决策树中的决策路径, 转为字典格式'
    decision_path_node_series = decision_regression_tree.decision_path(test_sample)
    decision_paths_for_the_sample = list()
    s = 0
    for i in range(decision_path_node_series.shape[1]):
        if decision_path_node_series[0, i] == 1:
            if nodes[str(i)][0] == 'X':
                s += 1
                decision_paths_for_the_sample.append(nodes[str(i)])
                if s == max_depth - 1:
                    break
            else:
                break

    '对所有训练样本依次执行decision_path_for_the_sample中的分裂操作'
    '正负样本数'
    num_pos_and_neg_sample = dict()
    '分裂步数'
    for i in range(s):
        s_1 = 0
        s_2 = 0
        if i == 0:
            X_train = X_train
            y_train = y_train
        else:
            X_train = np.array(X_1_selected).reshape(len(X_1_selected),  X_train.shape[1])
            y_train = np.array(y_1_selected).reshape(len(y_1_selected), 1)

        '小于分裂点为负 1， 大于分裂点为正 2'
        X_1_selected = list()
        y_1_selected = list()
        X_2_selected = list()
        y_2_selected = list()
        for j in range(X_train.shape[0]):
            X = list(X_train[j])
            if eval(decision_paths_for_the_sample[i]) == True:
                s_1 += 1
                '样本指标 1'
                X_1_selected.append(X)
                '样本标签 1'
                y_1_selected.append(list(y_train[j]))
            elif eval(decision_paths_for_the_sample[i]) == False:
                s_2 += 1
                '样本指标 2'
                X_2_selected.append(X)
                '样本标签 2'
                y_2_selected.append(list(y_train[j]))

        '正样本分数'
        if len(y_1_selected) == 0:
            scores_1 = None
        else:
            scores_1 = [p[0] for p in y_1_selected].count(1) / len(y_1_selected)
        if len(y_2_selected) == 0:
            scores_2 = 0
        else:
            scores_2 = [p[0] for p in y_2_selected].count(1) / len(y_2_selected)
        num_pos_and_neg_sample[decision_paths_for_the_sample[i]] = [s_1, s_2, scores_1, scores_2]
    return num_pos_and_neg_sample

def findAllPosAndNegNumInGBDT(GBDT, test_sample, X_train, y_train, n_estimators, max_depth):
    '寻找GBDT模型所有决策树中节点分裂的样本数和正样本分数'
    pos_and_neg_num_in_all_trees = list()
    for i in range(n_estimators):
        print('this is tree %d' % i)
        gbdt_sub = GBDT[i, 0]
        pos_and_neg_num_in_all_trees.append(calPosAndNegSampleNumOnEachTree(gbdt_sub, test_sample, X_train, y_train, max_depth))
    return pos_and_neg_num_in_all_trees

def calFeatrueScoresOnEachTree(num_pos_and_neg_sample):
    '计算每棵树上的特征分数'
    feature_scores_on_each_tree = dict()
    for key in num_pos_and_neg_sample.keys():
        denom = (num_pos_and_neg_sample[key][0] + num_pos_and_neg_sample[key][1])
        if denom != 0:
            feature_scores_on_each_tree[key[0 : key.index(' ')]] = (num_pos_and_neg_sample[key][0] * num_pos_and_neg_sample[key][2] + num_pos_and_neg_sample[key][1] * num_pos_and_neg_sample[key][3]) / denom
        else:
            feature_scores_on_each_tree[key[0 : key.index(' ')]] = 0
    return feature_scores_on_each_tree

def calAllFeatureScoresInGBDT(GBDT, test_sample, X_train, y_train, n_estimators, max_depth):
    '计算GBDT中所有树上的特征分数'
    pos_and_neg_num_in_all_trees = findAllPosAndNegNumInGBDT(GBDT, test_sample, X_train, y_train, n_estimators, max_depth)
    feature_scores_on_GBDT = list()
    for feature_scores_on_each_tree in pos_and_neg_num_in_all_trees:
        feature_scores_on_GBDT.append(calFeatrueScoresOnEachTree(feature_scores_on_each_tree))
    return feature_scores_on_GBDT

def calLocalInterpretationFeatureScores(feature_scores_on_GBDT, tree_weights, n_estimators):
    '计算测试样本在GBDT上返回的局部风险点和对应分数'
    important_features = list()
    for i in range(n_estimators):
        for key in list(feature_scores_on_GBDT[i].keys()):
            important_features.append(key)
    '去重'
    important_features = list(set(important_features))
    overall_local_interpretation_feature_scores_on_GBDT = dict()
    for important_feature in important_features:
        overall_local_interpretation_feature_scores_on_GBDT[important_feature] = 0
    for i in range(len(feature_scores_on_GBDT)):
        for key in feature_scores_on_GBDT[i].keys():
            overall_local_interpretation_feature_scores_on_GBDT[key] += tree_weights[i] * feature_scores_on_GBDT[i][key] / sum(tree_weights)
    return overall_local_interpretation_feature_scores_on_GBDT

## 算例计算 ——————————————————————————————————————————————————————————————————————————————
feature_scores_on_GBDT = calAllFeatureScoresInGBDT(GBDT, test_sample, X_train, y_train, n_estimators, max_depth)
'暂时用的同一权重, 后面应该把残差作为权重'
tree_weights = np.ones([n_estimators, 1])

## 计算残差权重 ——————————————————————————————————————————————————————————————————————————
y_pred_each_tree = list()
for i in range(n_estimators):
    clf = GBDT[i, 0]
    y_pred_each_tree.append(clf.predict(X_train))

'使用每个样本在每棵树上预测值的残差平方和进行加权'
tree_weights = list()
for i in range(len(y_pred_each_tree)):
    s = 0
    for j in range(len(y_pred_each_tree[i])):
        s += pow(y_pred_each_tree[i][j], 2)
    tree_weights.append(s)

## 计算 Local FC —————————————————————————————————————————————————————————————————————————
overall_local_interpretation_feature_scores_on_GBDT = calLocalInterpretationFeatureScores(feature_scores_on_GBDT, tree_weights, n_estimators)
overall_local_interpretation_feature_scores_on_GBDT = sorted(overall_local_interpretation_feature_scores_on_GBDT.items(),
                                                             key = operator.itemgetter(1),
                                                             reverse = True)
print('对于该测试样本影响较大的特征和分数')
print(overall_local_interpretation_feature_scores_on_GBDT)

## Local FC 作图 —————————————————————————————————————————————————————————————————————————
values = list()
keys = list()
for i in range(len(overall_local_interpretation_feature_scores_on_GBDT)):
    values.append(overall_local_interpretation_feature_scores_on_GBDT[i][1])
    keys.append(overall_local_interpretation_feature_scores_on_GBDT[i][0])

plt.figure()
plt.bar(range(len(list(overall_local_interpretation_feature_scores_on_GBDT))), values, width = 0.5)
plt.xticks(range(len(list(overall_local_interpretation_feature_scores_on_GBDT))),
           keys,
           rotation = 45)
plt.title('Local FC')
plt.xlabel('feature name')
plt.ylabel('FC value')
