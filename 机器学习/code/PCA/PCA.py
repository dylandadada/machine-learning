import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class unit:
    def genData(self, mean, cov, size, alpha, beta, gamma):
        X = np.random.multivariate_normal(mean, cov, size)
        X = np.c_[X, np.ones((X.shape[0], 1))]

        alp = np.mat([[1, 0, 0, 0], 
                      [0, math.cos(alpha), math.sin(alpha), 0],
                      [0, -math.sin(alpha), math.cos(alpha), 0], 
                      [0, 0, 0, 1]])
        bet = np.mat([[math.cos(beta), 0, -math.sin(beta), 0], 
                      [0, 1, 0, 0],
                      [math.sin(beta), 0, math.cos(beta), 0], 
                      [0, 0, 0, 1]])
        gam = np.mat([[math.cos(gamma), math.sin(gamma), 0, 0],
                      [-math.sin(gamma), math.cos(gamma), 0, 0], 
                      [0, 0, 1, 0], 
                      [0, 0, 0, 1]])

        alphaX = np.dot(X, alp)
        betaX = np.dot(alphaX, bet)
        gammaX = np.dot(betaX, gam)
        data = np.mat(gammaX[:, 0:3]).T
        ori = np.mat(X[:, 0:3]).T
        return data, ori

    def pca(self, data, size, k, n):
        data = data.astype('int64')
        average = np.mean(data,axis=0)
        m, n = np.shape(data)
        data_adjust = []
        avgs = np.tile(average, (m, 1))
        data_adjust = data - avgs
        covX = np.cov(data_adjust.T)   #计算协方差矩阵
        featValue, featVec = np.linalg.eig(covX)  #求解协方差矩阵的特征值和特征向量
        index = np.argsort(-featValue) #依照featValue进行从大到小排序
        finalData = []

        #注意特征向量时列向量。而numpy的二维矩阵(数组)a[m][n]中，a[1]表示第1行值
        selectVec = np.matrix(featVec.T[index[:k]])
        finalData = data_adjust * selectVec.T 
        reconData = (finalData * selectVec) + average
        return reconData