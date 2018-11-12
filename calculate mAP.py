import numpy as np
import time
import os
import pickle
from scipy.misc import imsave
# read train and test binarayCode
CURRENT_DIR = os.getcwd()

def getNowTime():
    return '['+time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))+']'


def getCode(train_codes,train_groudTruth,test_codes,test_groudTruth):

    line_number = 0
    with open(CURRENT_DIR+'/result.txt', 'r') as f:
        for line in f:
            temp = line.strip().split('\t')
            if line_number < 10000:
                test_codes.append([i if i == 1 else -1 for i in map(int, list(temp[0]))])
                list2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                list2[int(temp[1])] = 1
                test_groudTruth.append(list2) # get test ground truth(0-9)
            else:
                train_codes.append([i if i == 1 else -1 for i in map(int, list(temp[0]))]) # change to -1, 1
                list2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                list2[int(temp[1])] = 1
                train_groudTruth.append(list2) # get test ground truth(0-9)

            line_number += 1
    print(getNowTime(), 'read data finish')

def getHammingDist(code_a,code_b):
    dist = 0
    for i in range(len(code_a)):
         if code_a[i] != code_b[i]:
             dist += 1
    return dist

def read_cifar10_data():
    data_dir = CURRENT_DIR+'/data/cifar-10-batches-py/'
    train_name = 'data_batch_'
    test_name = 'test_batch'
    train_X = None
    train_Y = None
    test_X = None
    test_Y = None

    # train data
    for i in range(1, 6):
        file_path = data_dir+train_name+str(i)
        with open(file_path, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            if train_X is None:
                train_X = dict[b'data']
                train_Y = dict[b'labels']
                print(train_X)
            else:
                train_X = np.concatenate((train_X, dict[b'data']), axis=0)
                train_Y = np.concatenate((train_Y, dict[b'labels']), axis=0)
    # test_data
    file_path = data_dir + test_name
    with open(file_path, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        test_X = dict[b'data']
        test_Y = dict[b'labels']
    train_X = train_X.reshape((50000, 3, 32, 32)).transpose(0, 2, 3, 1).astype(np.float)
    # train_Y = train_Y.reshape((50000)).astype(np.float)
    test_X = test_X.reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1).astype(np.float)
    # test_Y.reshape((10000)).astype(np.float)

    train_y_vec = np.zeros((len(train_Y), 10), dtype=np.float)
    test_y_vec = np.zeros((len(test_Y), 10), dtype=np.float)
    for i, label in enumerate(train_Y):
        train_y_vec[i, int(train_Y[i])] = 1.  # y_vec[1,3] means #2 row, #4column
    for i, label in enumerate(test_Y):
        test_y_vec[i, int(test_Y[i])] = 1.  # y_vec[1,3] means #2 row, #4column

    return train_X/255., train_y_vec, test_X/255., test_y_vec

if __name__ =='__main__':
    search_picture = 1
    train_x, train_y, test_x, test_y = read_cifar10_data()
    print(getNowTime(), 'start!')
    train_codes = []
    train_groudTruth = []
    test_codes = []
    test_groudTruth = []
    getCode(train_codes, train_groudTruth, test_codes, test_groudTruth)
    train_codes = np.array(train_codes)
    train_groudTruth = np.array(train_groudTruth)
    test_codes = np.array(test_codes)
    test_groudTruth = np.array(test_groudTruth)
    numOfTest = 10000
    # generate hanmming martix, g.t. martix  10000*50000
    gt_martix = np.dot(test_groudTruth, np.transpose(train_groudTruth))
    print(gt_martix)
    print(getNowTime(), 'gt_martix finish!')
    ham_martix = np.dot(test_codes, np.transpose(train_codes)) # hanmming distance map to dot value 
    print(getNowTime(), 'ham_martix finish!')

    # sort hanmming martix,Returns the indices that would sort an array.
    sorted_ham_martix_index = np.argsort(ham_martix, axis=1)
    
    # calculate mAP


    apall = np.zeros((numOfTest, 1), np.float64)
    for i in range(numOfTest):
        x = 0.0
        p = 0
        test_oneLine = sorted_ham_martix_index[i, :]
        length = test_oneLine.shape[0]
        num_return_NN = 1000 # top 1000
        for j in range(num_return_NN):
            imgs = train_x[test_oneLine[length-j-1]]
        #     imsave("./result/{}.jpg".format(j), imgs)
        # img = test_x[0]
        # imsave("./result/0000.jpg", img)
            if gt_martix[i][test_oneLine[length-j-1]] == 1: # 浠庡皬鍒板ぇ鎺掑垪
                     x += 1
                     p += x/(j+1)
            if p == 0:
                apall[i] = 0
            else:
                apall[i] = p/x

    mAP = np.mean(apall)
    print(getNowTime(), 'mAP:', mAP)
