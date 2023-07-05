from __future__ import print_function
import os
from PIL import Image

import logging
import random
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
# from visdom import Visdom
from utils import *
import torch
from vit_pytorch import ViT
import math
import torch.nn as nn
import numpy as np
from scipy.io import loadmat, savemat
from sklearn.metrics import roc_curve, auc,roc_auc_score,confusion_matrix,accuracy_score
class KZDataset():
    def __init__(self, path_0=None,path_1=None,aug_path = None, ki=0, K=5, typ='train',transform=None, rand=False):
        self.data_info_0 = self.get_img_info(path_0)
        self.data_info_1 = self.get_img_info(path_1)

        leng_0 = len(self.data_info_0)
        every_z_len_0 = leng_0 / K
        leng_1 = len(self.data_info_1)
        every_z_len_1 = leng_1 / K
        self.data_info_1_aug = self.get_img_info(aug_path)
        if typ == 'val':
            self.data_info_0 = self.data_info_0[math.ceil(every_z_len_0 * ki) : math.ceil(every_z_len_0 * (ki+1))]
            self.data_info_1 = self.data_info_1[math.ceil(every_z_len_1 * ki) : math.ceil(every_z_len_1 * (ki+1))]
            self.data_info = self.data_info_0 + self.data_info_1
        elif typ == 'train':
            self.data_info_0 = self.data_info_0[: math.ceil(every_z_len_0 * ki)] + self.data_info_0[math.ceil(every_z_len_0 * (ki+1)) :]
            self.data_info_1 = self.data_info_1[: math.ceil(every_z_len_1 * ki)] + self.data_info_1[math.ceil(every_z_len_1 * (ki+1)) :]
            aug_list = []
            for i in self.data_info_1:
                for ii in self.data_info_1_aug:
                    if i[0].split('\\')[-1][:-4]==ii[0].split('\\')[-1][:-4].split('_')[0]:
                        aug_list.append(ii[:2]+i[2:])
            self.data_info = self.data_info_0 + self.data_info_1 + aug_list
        print(len(self.data_info))
        if rand:
	        random.seed(1)
        	random.shuffle(self.data_info)
        self.transform = transform

    def __getitem__(self, index):
    	# Dataset读取图片的函数
        img_pth, label,cp = self.data_info[index]
        img = Image.open(img_pth).convert('RGB')
        cp = [int(i) for i in cp]
        cp = np.array(cp)
        if self.transform is not None:
            img = self.transform(img)
        patient = img_pth.split('\\')[-1][:-4].split('_')[0]
        return label,patient,cp

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_img_info(txt_path):
        data_info = []
        data = open(txt_path, 'r')
        data_lines = data.readlines()
        for data_line in data_lines:
            data_line = data_line.split()
            img_pth = data_line[0]
            label = int(data_line[1])
            cp = data_line[-5:]
            data_info.append((img_pth, label,cp))
        return data_info  
class logistic():
    def __init__(self,train_data,train_label,test_data,test_label,train_num,learning_rate):
        self.train_data =train_data
        self.train_label = train_label
        self.test_data = test_data
        self.test_label = test_label
        self.train_num = train_num
        self.learning_rate = learning_rate
        self.weight = np.ones(len(train_data[0])+1, dtype=np.float)

    def add_bias(self,data):
        temp = np.ones(len(data))
        new_data_transpose = np.row_stack((np.transpose(data),temp))
        new_data = np.transpose(new_data_transpose)
        return new_data

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def compute(self,data):
        #print(self.weight)
        z = np.dot(data, np.transpose(self.weight))
        # print(z)
        predict = self.sigmoid(z)
        return predict

    def error(self,predict,label):
        return np.power(predict - label, 2).sum()

    def update(self,data,diff):
        self.weight += self.learning_rate * np.dot(diff,data)/len(data)

    def train(self):
        data = self.add_bias(self.train_data)
        for i in range(self.train_num):
            predict = self.compute(data)
            #print(predict)
            error = self.error(predict,self.train_label)
            diff = self.train_label - predict
            self.update(data,diff)
            #print(error)

    def calculate_predict(self,my_data):
        data = self.add_bias(my_data)
        predict = self.compute(data)
        my_predict = np.zeros(len(predict))
        for i in range(len(predict)):
            if predict[i] > 0.5:
                my_predict[i] = 1
            else:
                my_predict[i] = 0

        return my_predict, predict

    def accuracy(self,predict):
        label = self.train_label
        num = 0
        for i in range(len(predict)):
            if predict[i] == label[i]:
                num += 1
        accuracy_num = num / len(predict)
        return accuracy_num

    def test(self):
        predict, predict_p = self.calculate_predict(self.test_data)
        label = self.test_label
        num = 0
        for i in range(len(predict)):
            if predict[i] == label[i]:
                num += 1
        accuracy_num = num / len(predict)
        aver_auc = roc_auc_score(label, predict_p)
        
        confu = confusion_matrix(label, predict,labels=list(set(label)))
        sens = confu[0][0]/(confu[0][0]+confu[0][1])
        spec = confu[1][1]/(confu[1][1]+confu[1][0])
    
        return accuracy_num, aver_auc, predict_p,sens,spec

K =5
total_auc = 0
total_spec = 0
total_sens = 0
for ki in range(K):
    transform_train = transforms.Compose([
        # transforms.Scale((512, 512)),
        # transforms.RandomCrop(256, padding=8),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset = KZDataset(path_0='/dataset/train/class_001/list_0.txt',
                                path_1='/dataset/train/class_002/list_1.txt',
                                aug_path = '/dataset/train/class_002_aug/train_list.txt',
                            ki=ki, K=K, typ='train', transform=transform_train, rand=False)
    valset = KZDataset(path_0='/dataset/train/class_001/list_0.txt',
                                path_1='/dataset/train/class_002/list_1.txt',
                                aug_path = '/dataset/train/class_002_aug/train_list.txt',
                            ki=ki, K=K, typ='val', transform=transform_train, rand=False)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False, num_workers=0)
    for (batch_idx, input) in enumerate(zip(trainloader,testloader)):
        train_num = 4000
        learning_rate = 0.1
        # logistic 分类
        train = input[0]
        test = input[1]
        cp_train,label_train,cp_test,label_test = np.array(train[2]), np.array(train[0]),np.array(test[2]),np.array(test[0])
        my_logistic = logistic(cp_train,label_train,cp_test,label_test, train_num, learning_rate)
        my_logistic.train()
        my_predict, trainpredict_p = my_logistic.calculate_predict(cp_train)
        my_accuracy = my_logistic.accuracy(my_predict)
        print("train accuracy")
        print(my_accuracy)
    test_accuracy, test_auc, testpredict_p,sens,spec = my_logistic.test()
    print(testpredict_p)
    print("test accuracy: ", test_accuracy, '  test auc: ', test_auc,'test_sens:',sens,'test_spec:',spec)
    total_auc += test_auc
    total_spec += spec
    total_sens += sens
print('cp_only_auc:',total_auc/5,'cp_only_sens:',total_sens/5,'cp_only_spec:',total_spec/5)
