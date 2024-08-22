from scipy import io
from tools.filter import filter
import numpy as np
import os 
import random

def readOpenBmi(root,nSub,Isfilter,freq):

    total_data=io.loadmat(root+'session1/'+'s'+str(nSub).zfill(3)+'.mat')
    train_data=total_data['x']
    train_label=total_data['y']
    train_data=np.transpose(train_data,(2,0,1))
    if Isfilter == True:
        train_data = filter(freq, train_data)
    train_data=np.expand_dims(train_data,axis=1)

    train_label=train_label[0]+1
    shuffle_num=np.random.permutation(len(train_data))
    trainData=train_data[shuffle_num,:,:,:]
    trainLabel=train_label[shuffle_num]

    test_tmp=io.loadmat(root+'session2/'+'se'+str(nSub).zfill(3)+'.mat')
    test_data=test_tmp['x']
    test_label=test_tmp['y']
    test_data=np.transpose(test_data,(2,0,1))
    if Isfilter == True:
        test_data = filter(freq, test_data)
    testData=np.expand_dims(test_data,axis=1)

    testLabel=test_label[0]+1

    #strandardize
    target_mean=np.mean(trainData)
    target_std=np.std(trainData)
    trainData=(trainData-target_mean)/target_std
    testData=(testData-target_mean)/target_std

    return trainData,trainLabel,testData,testLabel

def readBCI2A(root,nSub,Isfilter,freq):
    file = 'A0%dT.mat' % nSub
    file_test = 'A0%dE.mat' % nSub

    total_data = io.loadmat(root + file)
    train_data = total_data['data']
    train_label = total_data['label']
    train_data = np.transpose(train_data, (2, 1, 0))

    # test data
    test_tmp = io.loadmat(root + file_test)
    test_data = test_tmp['data']
    test_label = test_tmp['label']
    test_data = np.transpose(test_data, (2, 1, 0))


    # filter data
    if Isfilter == True:
        train_data = filter(freq, train_data)
    train_data = np.expand_dims(train_data, axis=1)
    train_label = np.transpose(train_label)

    allData = train_data
    allLabel = train_label[0]

    shuffle_num = np.random.permutation(len(allData))
    allData = allData[shuffle_num, :, :, :]
    allLabel = allLabel[shuffle_num]

    # filter data
    if Isfilter == True:
        test_data = filter(freq, test_data)
    test_data = np.expand_dims(test_data, axis=1)
    test_label = np.transpose(test_label)

    testData = test_data
    testLabel = test_label[0]
    # standardize
    target_mean = np.mean(allData)
    target_std = np.std(allData)
    allData = (allData - target_mean) / target_std
    testData = (testData - target_mean) / target_std

# data shape: (trial, conv channel, electrode channel, time samples)
    return allData, allLabel, testData, testLabel




def readBCI2B(root,nSub,Isfilter,freq):
    file = 'B0%dT.mat' % nSub
    file_test = 'B0%dE.mat' % nSub

    total_data = io.loadmat(root + file)
    train_data = total_data['data']
    train_label = total_data['label']
    train_data = np.transpose(train_data, (2, 1, 0))

    # test data
    test_tmp = io.loadmat(root + file_test)
    test_data = test_tmp['data']
    test_label = test_tmp['label']
    test_data = np.transpose(test_data, (2, 1, 0))


    # filter data
    if Isfilter == True:
        train_data = filter(freq, train_data)
    train_data = np.expand_dims(train_data, axis=1)
    train_label = np.transpose(train_label)

    allData = train_data
    allLabel = train_label[0]

    shuffle_num = np.random.permutation(len(allData))
    allData = allData[shuffle_num, :, :, :]
    allLabel = allLabel[shuffle_num]

    # filter data
    if Isfilter == True:
        test_data = filter(freq, test_data)
    test_data = np.expand_dims(test_data, axis=1)
    test_label = np.transpose(test_label)

    testData = test_data
    testLabel = test_label[0]
    # standardize
    target_mean = np.mean(allData)
    target_std = np.std(allData)
    allData = (allData - target_mean) / target_std
    testData = (testData - target_mean) / target_std

# data shape: (trial, conv channel, electrode channel, time samples)
    return allData, allLabel, testData, testLabel



def loadBCI2A(root,nSub):

    data_file='A0%dT' %nSub
    data_path = os.path.join(root, data_file + '_data.npy')
    label_path = os.path.join(root, data_file + '_label.npy')
    data = np.load(data_path)
    label = np.load(label_path).squeeze()

    # Shuffle
    data, label = shuffle_data(data, label)


    test_data_file='A0%dE' %nSub
    test_data_path = os.path.join(root, test_data_file + '_data.npy')
    test_label_path = os.path.join(root, test_data_file + '_label.npy')
    test_data = np.load(test_data_path)
    test_label = np.load(test_label_path).squeeze()

    target_mean = np.mean(data)
    target_std = np.std(data)
    data = (data - target_mean) / target_std
    test_data = (test_data - target_mean) / target_std

    return data, label,test_data,test_label


def loadBCI2B(root,nSub):

    data_file='B0%dT' %nSub
    data_path = os.path.join(root, data_file + '_data.npy')
    label_path = os.path.join(root, data_file + '_label.npy')
    data = np.load(data_path)
    label = np.load(label_path).squeeze()


    # Shuffle
    data, label = shuffle_data(data, label)

    test_data_file='B0%dE' %nSub
    test_data_path = os.path.join(root, test_data_file + '_data.npy')
    test_label_path = os.path.join(root, test_data_file + '_label.npy')
    test_data = np.load(test_data_path)
    test_label = np.load(test_label_path).squeeze()


    target_mean = np.mean(data)
    target_std = np.std(data)
    data = (data - target_mean) / target_std
    test_data = (test_data - target_mean) / target_std

    return data, label,test_data,test_label


def readHGD(root, nSub):

    total_data = io.loadmat(root + '%dT.mat' % nSub)
    train_data = total_data['data']
    train_label = total_data['labels']
    train_label = np.squeeze(train_label)
    shuffle_num = np.random.permutation(len(train_data))
    traindata = train_data[shuffle_num, :, :]
    train_label = train_label[shuffle_num]

    # test data
    test_temp = io.loadmat(root + '%dE.mat' % nSub)
    test_data = test_temp['data']
    test_label = test_temp['labels']
    test_label = np.squeeze(test_label)
    target_mean = np.mean(train_data)
    target_std = np.std(train_data)
    train_data = (train_data - target_mean) / target_std
    test_data = (test_data - target_mean) / target_std



    return train_data, train_label, test_data, test_label



def shuffle_data(data, label):
    index = [i for i in range(len(data))]
    random.shuffle(index)
    shuffle_data = data[index]
    shuffle_label = label[index]
    return shuffle_data, shuffle_label