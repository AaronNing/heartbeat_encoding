import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn import metrics

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def chooseIdcs(Y, Y_list, balancing=False, max_tol_imbalance_ratio=1):
    in_idcs = np.where([(cat in Y_list) for cat in Y])[0]
    if balancing:
        samp_idcs_list = [np.where(Y==Y_target)[0] for Y_target in np.unique(Y_list)]
        samp_size_list = [len(idcs) for idcs in samp_idcs_list]
        max_tol_size = np.floor(min(samp_size_list) * max_tol_imbalance_ratio).astype(int)
        samp_idcs_list_balanced = [np.random.choice(idcs, min(len(idcs), max_tol_size), replace=False) for idcs in samp_idcs_list]
        balanced_idcs = np.concatenate(samp_idcs_list_balanced, axis=0)
        print('%d in %d samples dicarded due to imbalance. ' % (len(in_idcs) - len(balanced_idcs), len(in_idcs)))
        out_idcs = balanced_idcs
    else:
        out_idcs = in_idcs
    return out_idcs

def chooseDataset(X, Y, Y_list, balancing=False, max_tol_imbalance_ratio=1):
    idcs = chooseIdcs(Y, Y_list, balancing, max_tol_imbalance_ratio)
    return X[idcs, :], Y[idcs]

def splitDataset(X, Y, test_ratio):
    X_train_list = []
    X_test_list = []
    Y_train_list = []
    Y_test_list = []
    for Y_class in np.unique(Y):
        X_sub, Y_sub = chooseDataset(X, Y, Y_class)
        n = X_sub.shape[0]
        n_train = np.floor(n * (1-test_ratio)).astype(int)
        assert(n_train > 0)
        idx_train = np.random.choice(np.arange(n), n_train, replace=False)
        idx_test = np.setdiff1d(np.arange(n), idx_train)
        X_train_list.append(X_sub[idx_train,:])
        X_test_list.append(X_sub[idx_test,:])
        Y_train_list.append(Y_sub[idx_train])
        Y_test_list.append(Y_sub[idx_test])
    X_train = np.concatenate(X_train_list, axis=0)
    X_test = np.concatenate(X_test_list, axis=0)
    Y_train = np.concatenate(Y_train_list, axis=0)
    Y_test = np.concatenate(Y_test_list, axis=0)
    return X_train, X_test, Y_train, Y_test

def normalize(X):
    return (X - np.mean(X)) / np.std(X)

def normalize_list(X_list):
    return (normalize(X) for X in X_list)

def doTtest(X, Y):
    assert(len(np.unique(Y))==2)
    assert(len(X.shape)==1)
    class_names = np.unique(Y)
    result = ttest_ind(X[Y==class_names[0]], X[Y==class_names[1]], equal_var=False)
    return result

def getLDAResults(X, Y, n_epochs, printing=False, plotting=False, psd_freqs=0):
    
    X = normalize(X)

    f1_score_train_list = []
    f1_score_test_list = []

    for idx_epoch in range(n_epochs):

        X_train, X_test, Y_train, Y_test = splitDataset(X, Y, test_ratio=0.2)

        lda_model = LDA()
        pcs_train = lda_model.fit_transform(X_train, Y_train)
        # score_train = lda_model.score(X_train, Y_train)
        # score_test = lda_model.score(X_test, Y_test)
        f1_score_train = metrics.f1_score(lda_model.predict(X_train), Y_train, average='macro')
        f1_score_test = metrics.f1_score(lda_model.predict(X_test), Y_test, average='macro')

        f1_score_train_list.append(f1_score_train)
        f1_score_test_list.append(f1_score_test)
    
    f1_train = (np.mean(f1_score_train_list), np.std(f1_score_train_list))
    f1_test = (np.mean(f1_score_test_list), np.std(f1_score_test_list))
    
    if printing:
        print('f1_score_train: %.3f (+- %.3f)' % f1_train)
        print('f1_score_test: %.3f (+- %.3f)' % f1_test)

    # pcs_test = lda_model.transform(X_test)
    # plt.figure(figsize=[15,5], dpi=144, facecolor=[0.9,0.9,0.9])
    # plt.subplot(1,2,1)
    # sns.kdeplot(x=pcs_train[:,0], y=pcs_train[:,1], hue=Y_train)
    # # plt.title('train (accuracy = %.3f)'%score_train)
    # plt.title('train (f1_score = %.3f)'%f1_score_train)
    # plt.subplot(1,2,2)
    # sns.scatterplot(x=pcs_test[:,0], y=pcs_test[:,1], hue=Y_test)
    # # plt.title('test (accuracy = %.3f)'%score_test)
    # plt.title('test (f1_score = %.3f)'%f1_score_test)
    # plt.show()

    if plotting:
        if len(np.unique(Y)) == 2:
            lda_model = LDA()
            pcs_all = lda_model.fit_transform(X, Y)
            f1_score_all = lda_model.score(X, Y)
            plt.figure(figsize=[8,4], dpi=144, facecolor=[0.9,0.9,0.9])
            plt.subplot(1,2,1)
            result = doTtest(pcs_all.flatten(), Y)
            sns.kdeplot(x=pcs_all.flatten(), hue=Y)
            plt.title('LDA')
            plt.text(x=plt.xlim()[0], y=plt.ylim()[1], s='p = %.3e'%result.pvalue)
            plt.subplot(1,2,2)
            sns.lineplot(x=psd_freqs, y=np.abs(lda_model.scalings_[:,0]), color='k')
            plt.grid()
            plt.title('ld1')
            plt.suptitle('all (f1_score = %.3f)'%f1_score_all)
            plt.tight_layout()
            plt.show()
        elif len(np.unique(Y)) > 2:
            lda_model = LDA()
            pcs_all= lda_model.fit_transform(X, Y)
            f1_score_all = lda_model.score(X, Y)
            plt.figure(figsize=[10,10], dpi=144, facecolor='white')
            plt.subplot(2,2,1)
            sns.kdeplot(x=pcs_all[:,0], y=pcs_all[:,1], hue=Y)
            plt.subplot(2,2,2)
            sns.scatterplot(x=pcs_all[:,0], y=pcs_all[:,1], hue=Y)
            # plt.title('train (accuracy = %.3f)'%score_train)
            plt.subplot(2,2,3)
            sns.lineplot(x=psd_freqs, y=np.abs(lda_model.scalings_[:,0]), color='k')
            plt.grid()
            plt.title('ld1')
            plt.subplot(2,2,4)
            sns.lineplot(x=psd_freqs, y=np.abs(lda_model.scalings_[:,1]), color='k')
            plt.grid()
            plt.title('ld2')
            plt.suptitle('all (f1_score = %.3f)'%f1_score_all)
            plt.tight_layout()
            plt.show()
        else:
            pass
        
    return (f1_train, f1_test)