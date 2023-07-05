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
import math
import torch.nn as nn

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
        img_pth, label,cp = self.data_info[index]
        img = Image.open(img_pth).convert('RGB')
        cp = [int(i) for i in cp]
        cp = np.array(cp)
        if self.transform is not None:
            img = self.transform(img)
        patient = img_pth.split('\\')[-1][:-4].split('_')[0]
        return img, label,patient,cp

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

def train(nb_epoch, batch_size, store_name, resume=False, start_epoch=0, model_path=None):
    # setup output
    exp_dir = store_name
    try:
        os.stat(exp_dir)
    except:
        os.makedirs(exp_dir)
    K =5
    for ki in range(K):
        use_cuda = torch.cuda.is_available()
        print(use_cuda)
        # viz = Visdom()
        # viz.line([[0, 0]], [[0, 0]], win='train_loss', opts=dict(title='train loss&acc.',
        #                                                          legend=['loss', 'acc.']))

        # Data
        print('==> Preparing data..')
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
        valset = KZDataset(path_0='/dataset/test/class_001/list_0.txt',
                                path_1='/dataset/test/class_002/list_1.txt',
                                aug_path = '/',
                                ki=ki, K=K, typ='val', transform=transform_train, rand=False)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
        if resume:
            net = torch.load(model_path)
        else:
            net = load_model(model_name='resnet50_pmg', pretrain=False, require_grad=True)
        net.aux_logits = False
        netp = torch.nn.DataParallel(net, device_ids=[0,2])
        # GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net.to(device)
        cudnn.benchmark = True
        CELoss = nn.CrossEntropyLoss()
        optimizer = optim.SGD([
            {'params': net.classifier_concat.parameters(), 'lr': 0.002},
            {'params': net.conv_block1.parameters(), 'lr': 0.002},
            {'params': net.classifier1.parameters(), 'lr': 0.002},
            {'params': net.conv_block2.parameters(), 'lr': 0.002},
            {'params': net.classifier2.parameters(), 'lr': 0.002},
            {'params': net.conv_block3.parameters(), 'lr': 0.002},
            {'params': net.classifier3.parameters(), 'lr': 0.002},
            {'params': net.features.parameters(), 'lr': 0.0002}
    
        ],
            momentum=0.9, weight_decay=5e-4)
    
        max_val_acc = 0
        lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002]
        for epoch in range(start_epoch, nb_epoch):
            print('\nEpoch: %d' % epoch)
            net.train()
            train_loss = 0
            train_loss1 = 0
            train_loss2 = 0
            train_loss3 = 0
            train_loss4 = 0
            correct = 0
            total = 0
            idx = 0
            for batch_idx, (inputs, targets,patient,cp) in enumerate(trainloader):
                idx = batch_idx
                if inputs.shape[0] < batch_size:
                    continue
                if use_cuda:
                    cp = cp.to(device)
                    inputs, targets = inputs.to(device), targets.to(device)
                cp = Variable(cp)
                inputs, targets = Variable(inputs), Variable(targets)
                # update learning rate
                for nlr in range(len(optimizer.param_groups)):
                    optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, nb_epoch, lr[nlr])
    
                # Step 1
                optimizer.zero_grad()
                inputs1 = jigsaw_generator(inputs, 8)
                output_1, _, _, _ = netp(inputs1,cp)
                loss1 = CELoss(output_1, targets) * 1
                loss1.backward()
                optimizer.step()
    
                # Step 2
                optimizer.zero_grad()
                inputs2 = jigsaw_generator(inputs, 4)
                _, output_2, _, _ = netp(inputs2,cp)
                loss2 = CELoss(output_2, targets) * 1
                loss2.backward()
                optimizer.step()
    
                # Step 3
                optimizer.zero_grad()
                inputs3 = jigsaw_generator(inputs, 2)
                _, _, output_3, _ = netp(inputs3,cp)
                loss3 = CELoss(output_3, targets) * 1
                loss3.backward()
                optimizer.step()
    
                # Step 4
                optimizer.zero_grad()
                _, _, _, output_concat = netp(inputs,cp)
                concat_loss = CELoss(output_concat, targets) * 2
                concat_loss.backward()
                optimizer.step()
    
                #  training log
                _, predicted = torch.max(output_concat.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()
    
                train_loss += (loss1.item() + loss2.item() + loss3.item() + concat_loss.item())
                train_loss1 += loss1.item()
                train_loss2 += loss2.item()
                train_loss3 += loss3.item()
                train_loss4 += concat_loss.item()
    
                if batch_idx % 2 == 0:
                    # viz.line([[train_loss / (batch_idx + 1), 100. * float(correct) / total]],
                    #          [[epoch*400+batch_idx, epoch*400+batch_idx]], win='train_loss', update='append')
                    print(
                        'K-fold %d,Step: %d | Loss1: %.3f | Loss2: %.5f | Loss3: %.5f | Loss_concat: %.5f | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                        ki,batch_idx, train_loss1 / (batch_idx + 1), train_loss2 / (batch_idx + 1),
                        train_loss3 / (batch_idx + 1), train_loss4 / (batch_idx + 1), train_loss / (batch_idx + 1),
                        100. * float(correct) / total, correct, total))
    
            train_acc = 100. * float(correct) / total
            train_loss = train_loss / (idx + 1)
            with open(exp_dir + '/results_train_np_%d.txt'%ki, 'a') as file:
                file.write(
                    'K-fold %d, Iteration %d | train_acc = %.5f | train_loss = %.5f | Loss1: %.3f | Loss2: %.5f | Loss3: %.5f | Loss_concat: %.5f |\n' % (
                    ki,epoch, train_acc, train_loss, train_loss1 / (idx + 1), train_loss2 / (idx + 1), train_loss3 / (idx + 1),
                    train_loss4 / (idx + 1)))

            test_acc, test_acc_com, test_loss = test(net, CELoss,valset, 10)
            net.cpu()
            torch.save(net, './' + store_name + '/model_%d.pth'%ki)
            net.to(device)
            with open(exp_dir + '/results_test_np_%d.txt'%ki, 'a') as file:
                file.write('K-fold %d,Iteration %d, test_acc = %.5f, test_acc_combined = %.5f, test_loss = %.6f\n' % (
                ki,epoch, test_acc, test_acc_com, test_loss))


train(nb_epoch=300,             # number of epoch
         batch_size=10,         # batch size
         store_name='DeepTEPP',     # folder for output
         resume=False,          # resume training from checkpoint
         start_epoch=0,         # the start epoch number when you resume the training
         model_path='')         # the saved model where you want to resume the training
