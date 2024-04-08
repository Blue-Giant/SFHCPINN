import os

import numpy as np

import scipy.io as scio


# Get the data from *.off file
def getEvalData_from_txt(filename):
    f = open(filename)
    data2eval_loss = []
    data2eval_acc = []
    data2eval_class_acc = []
    while True:
        new_line = f.readline()
        if new_line == '':
            break
        print('new_line:', new_line)
        x = new_line.split(' ')
        if x[0] == 'eval' and x[1] == 'mean' and x[2] == 'loss:':
            print('x[3]:', x[3])
            data2eval_loss.append(float(x[3]))
        if x[0] == 'eval' and x[1] == 'accuracy:':
            print('x[2]:', x[2])
            data2eval_acc.append(float(x[2]))
        if x[0] == 'eval' and x[1] == 'avg' and x[2] == 'class' and x[3] == 'acc:':
            print('x[4]:', x[4])
            data2eval_class_acc.append(float(x[4]))
    return data2eval_loss, data2eval_acc, data2eval_class_acc


# Get the data from *.off file
def getTrainData_from_txt(filename):
    f = open(filename)
    data2train_loss = []
    data2train_acc = []
    data2test_loss =[]
    data2test_rel = []
    while True:
        new_line = f.readline()
        if new_line == '':
            break
        print('new_line:', new_line)
        x = new_line.split(' ')
        if x[0] == 'solution' and x[1] == 'mean':
            # print('x[2]:', x[2])
            data2train_loss.append(float(x[6]))
        elif x[0] == 'solution'and x[1] == 'relative ':
            # print('x[1]:', x[1])
            data2train_acc.append(float(x[5]))
        elif x[0] == 'mean' and x[1] == 'square ':
            data2test_loss.append(float(x[9]))
        elif x[0] == 'relative' and x[1] == 'error ':
            data2test_rel.append(float(x[8]))
    return data2train_loss, data2train_acc, data2train_acc,data2test_rel


def getData2train(BASE_DIR,fileName):
    data1, data2, data2train_acc, data2test_rel = getTrainData_from_txt(fileName)

    train_loss = np.array(data1)
    train_acc = np.array(data2)
    test_loss =np.array(data2train_acc)
    test_rel = np.array(data2test_rel)

    outFile2data = '%s/%s.mat' % (BASE_DIR, 'trainData')
    key2loss = 'train_loss'
    key2acc = 'train_acc'
    scio.savemat(outFile2data, {key2loss: train_loss, key2acc: train_acc})
    outFile2data2 = '%s/%s.mat' % (BASE_DIR, 'testData')
    key2mse = 'test_loss'
    key2rel = 'test_rel'
    scio.savemat(outFile2data2, {key2mse: test_loss, key2rel: test_rel})


def getData2eval(BASE_DIR,fileName):
    data1, data2, data3 = getEvalData_from_txt(fileName)
    eval_loss = np.array(data1)
    eval_acc = np.array(data2)
    eval_class_acc = np.array(data3)

    outFile2data = '%s/%s.mat' % (BASE_DIR, 'eval')
    key2loss = 'eval_loss'
    key2acc = 'eval_acc'
    key2class = 'eval_class'
    scio.savemat(outFile2data, {key2loss: eval_loss, key2acc: eval_acc, key2class: eval_class_acc})


if __name__ == "__main__":
    # baseline
    # BASE_DIR = '/Users/dengjiaxin/fsdownload/11749'
    # fileName = '/Users/dengjiaxin/fsdownload/11749/log2train_tanh.txt'
    #sfhcpinn
    BASE_DIR = '/Users/dengjiaxin/fsdownload/88553'
    fileName = '/Users/dengjiaxin/fsdownload/88553/log2train_sin.txt'
    # #sfhcpinn
    # BASE_DIR = '/Users/dengjiaxin/fsdownload/55494'
    # fileName = '/Users/dengjiaxin/fsdownload/55494/log2train_sin.txt'
    getData2train(BASE_DIR=BASE_DIR, fileName=fileName)