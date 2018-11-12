import numpy as np
import os
from tflearn import *
import pickle
IMAGE_SIZE = 32
SAVE_MODEL_PATH = './model_alexnet-83800'
Train = False
Hashbits = 48
def create_alexnet(classes, hashbits):
    network = input_data(shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3])
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation='tanh')
    BinaryHash = fully_connected(network, hashbits, activation='sigmoid')
    network = fully_connected(BinaryHash, classes, activation='softmax')
    network = regression(network, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.000001)
    return network
def HashBinaryOut(hashbits):
    network = input_data(shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3])
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation='tanh')
    BinaryHash = fully_connected(network, hashbits, activation='sigmoid')
    return BinaryHash
def toBinaryString(binary_like_values):
    numOfImage, bit_length = binary_like_values.shape
    list_string_binary = []
    for i in range(numOfImage):
        str = ''
        for j in range(bit_length):
            str += '0' if (abs(binary_like_values[i][j] -0) < abs(binary_like_values[i][j] -1)) else '1'
        list_string_binary.append(str)
    return list_string_binary

def train(network, X, Y, save_model_path):
    # Training
    model = DNN(network, checkpoint_path='model_alexnet', max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir='output')
    if os.path.isfile(save_model_path + '.index'):
        model.load(save_model_path)
        print('load model...')
    for _ in range(50):
        model.fit(X, Y, n_epoch=5, validation_set=0.1, shuffle=True, show_metric=True, batch_size=64, snapshot_step=200, snapshot_epoch=False, run_id='alexnet')
        # Save the model
        model.save(save_model_path)
        print('save model...')
def read_cifar10_data():
    data_dir = os.getcwd()+'/data/cifar-10-batches-py/'
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
def search_compute(query, databook, image, data):
    hamming_dis = np.zeros(len(databook))
    dis = np.zeros(50)
    for i in range(len(databook)):
        hamming_dis[i] = np.sum(abs(query - databook[i]))
    ham_index = hamming_dis.argsort()
    for k in range(1, 51):
        dis[k-1] = np.linalg.norm(image - data[int(ham_index[k])])
    index = dis.argsort()
    return dis, index, ham_index
if __name__ == '__main__':
    if Train:
        network = create_alexnet(10, Hashbits)
        traindata, trainlabel, testdata, testlabel = read_cifar10_data()
        train(network, traindata, trainlabel, SAVE_MODEL_PATH)
    else:
        traindata, trainlabel, testdata, testlabel = read_cifar10_data()
        net = HashBinaryOut(Hashbits)
        model = DNN(net)
        model.load(SAVE_MODEL_PATH)
        #制作数据码本
        file_res = open('result.txt', 'w')
        codebook = model.predict(traindata)
        w_res = toBinaryString(codebook)
        # print(w_res)
        for j in range(50000):
            file_res.write(w_res[j] + '\t' + str(np.argmax(trainlabel[j])) + '\n')
        file_res.close()
        # #对输入图片进行检索
        # query = testdata[0]
        # binaryfeature= model.predict(query)
        # print(binaryfeature)
        # search_compute(binaryfeature, codebook, query, traindata)
