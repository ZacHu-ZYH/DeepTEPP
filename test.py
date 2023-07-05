import numpy as np
import random
import torch
import torchvision
from torch.autograd import Variable
from torchvision import transforms, models
import torch.nn.functional as F
from model import *
from Resnet import *
from PIL import Image
from sklearn.metrics import roc_curve, auc,roc_auc_score,confusion_matrix,accuracy_score
import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import operator

import re
from functools import reduce
import scipy.stats
from scipy import stats


class KZDataset():
    def __init__(self, path_0=None,path_1=None,aug_path=None,ki=0, K=5, typ='train',transform=None, rand=False,mark = 'CV'):
        self.data_info_0 = self.get_img_info(path_0)
        self.data_info_1 = self.get_img_info(path_1)
        
        leng_0 = len(self.data_info_0)
        every_z_len_0 = leng_0 / K
        leng_1 = len(self.data_info_1)
        every_z_len_1 = leng_1 / K
        self.data_info_1_aug = self.get_img_info(aug_path)
        if mark == 'CV':
            if typ == 'val':
                self.data_info_0 = self.data_info_0[math.ceil(every_z_len_0 * ki) : math.ceil(every_z_len_0 * (ki+1))]
                self.data_info_1 = self.data_info_1[math.ceil(every_z_len_1 * ki) : math.ceil(every_z_len_1 * (ki+1))]
                self.data_info = self.data_info_0 + self.data_info_1
            elif typ == 'train':
                self.data_info_0 = self.data_info_0[: math.ceil(every_z_len_0 * ki)] + self.data_info_0[math.ceil(every_z_len_0 * (ki+1)) :]
                self.data_info_1 = self.data_info_1[: math.ceil(every_z_len_1 * ki)] + self.data_info_1[math.ceil(every_z_len_0 * (ki+1)) :]
                aug_list = []
                for i in self.data_info_1:
                    for ii in self.data_info_1_aug:
                        if i[0].split('\\')[-1][:-4]==ii[0].split('\\')[-1][:-4].split('_')[0]:
                            aug_list.append(ii[:2]+i[2:])
                self.data_info = self.data_info_0 + self.data_info_1 + aug_list
        else:
            if typ == 'val':
                self.data_info_0 = sorted(self.data_info_0)
                self.data_info_1 = sorted(self.data_info_1)
                self.data_info_0 = self.data_info_0[math.ceil(every_z_len_0 * ki) : math.ceil(every_z_len_0 * (ki+1))]
                self.data_info_1 = self.data_info_1[math.ceil(every_z_len_1 * ki) : math.ceil(every_z_len_1 * (ki+1))]
                self.data_info = self.data_info_0 + self.data_info_1
            elif typ == 'train':
                self.data_info_0 = sorted(self.data_info_0)
                self.data_info_1 = sorted(self.data_info_1)
                self.data_info_0 = self.data_info_0[: math.ceil(every_z_len_0 * ki)] + self.data_info_0[math.ceil(every_z_len_0 * (ki+1)) :]
                self.data_info_1 = self.data_info_1[: math.ceil(every_z_len_1 * ki)] + self.data_info_1[math.ceil(every_z_len_1 * (ki+1)) :]
                self.data_info = self.data_info_0 + self.data_info_1
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
        return img, label,patient,cp

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_img_info(txt_path):
    	# 解析输入的txt的函数
    	# 转为二维list存储，每一维为 [ 图片路径，图片类别]
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
    
def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1

    return float(lr / 2 * cos_out)


def load_model(model_name, pretrain=True, require_grad=True):
    print('==> Building model..')
    if model_name == 'resnet50_pmg':
        net = resnet50(pretrained=pretrain)
        for param in net.parameters():
            param.requires_grad = require_grad
        net = PMG(net, 512, 2)

    return net


def model_info(model):  # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    print('\n%5s %50s %9s %12s %20s %12s %12s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        print('%5g %50s %9s %12g %20s %12.3g %12.3g' % (
            i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients\n' % (i + 1, n_p, n_g))


def jigsaw_generator(images, n):
    l = []
    for a in range(n):
        for b in range(n):
            l.append([a, b])
    block_size = 448 // n
    rounds = n ** 2
    random.shuffle(l)
    jigsaws = images.clone()
    for i in range(rounds):
        x, y = l[i]
        temp = jigsaws[..., 0:block_size, 0:block_size].clone()
        jigsaws[..., 0:block_size, 0:block_size] = jigsaws[..., x * block_size:(x + 1) * block_size,
                                                y * block_size:(y + 1) * block_size].clone()
        jigsaws[..., x * block_size:(x + 1) * block_size, y * block_size:(y + 1) * block_size] = temp

    return jigsaws

mark = 'CV'  #5-cross-validation mark = 'CV'

if mark == 'CV':
    K =5
    ff_lab = []
    ff_pre = []
    ff_pre_nor = []
    ff_name = []
    func = lambda x: [y for l in x for y in func(l)] if type(x) is list else [x]
    for ki in range(K):
        model_path='/pretrain_model/model_%d.pth'%ki
        net = torch.load(model_path)
        net.eval()
        params = list(net.parameters())
        k = 0
        for i in params:
            l = 1
            for j in i.size():
                l *= j
            k = k + l
        print("总参数数量和：" + str(k))

        use_cuda = torch.cuda.is_available()
        test_loss = 0
        correct = 0
        correct_com = 0
        total = 0
        idx = 0
        batch_size = 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net.to(device)
        transform_test = transforms.Compose([
            # transforms.Scale((512, 512)),
            # transforms.CenterCrop(448),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        testset = KZDataset(path_0='/dataset/train/class_001/list_0.txt',
                                path_1='/dataset/train/class_002/list_1.txt',
                                aug_path = '/dataset/train/class_002_aug/train_list.txt',
                                ki=ki, K=K, typ='val', transform=transform_test, rand=False,mark = mark
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
        final_lab = []
        final_pre = []
        final_pre_nor = []
        final_name = []

        for batch_idx, (inputs, targets,patient,cp) in enumerate(testloader):

            idx = batch_idx
            inputs, targets,cp = inputs.to(device), targets.to(device),cp.to(device)
            inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            output_1, output_2, output_3, output_concat= net(inputs,cp)
            # output_1, output_2, output_3, output_concat= net(inputs)
            outputs_com = output_1 + output_2 + output_3 + output_concat
            # pred = torch.softmax(outputs_com.data,dim=1)
            pred = outputs_com.data[:,1]
            
    
            # _, pred = torch.max(output_concat.data, 1)

            _, predicted_com = torch.max(outputs_com.data, 1)
    
            lab = list(targets.data.cpu().numpy())
            pre = list(pred.data.cpu().numpy())
            pre_nor = list(predicted_com.data.cpu().numpy())
            final_lab.append(lab)
            final_pre.append(pre)
            final_pre_nor.append(pre_nor)
            patient = list(patient)
            final_name.append(patient)
        ff_lab.append(final_lab)
        ff_pre.append(final_pre)
        ff_pre_nor.append(final_pre_nor)
        ff_name.append(final_name)
    ff_lab = func(ff_lab)
    ff_pre = func(ff_pre)
    ff_pre_nor = func(ff_pre_nor)
    ff_name = func(ff_name)
    final_lab =  np.array(np.array(ff_lab).flatten()).flatten()
    final_pre =  np.array(np.array(ff_pre).flatten()).flatten()
    final_pre_nor = np.array(np.array(ff_pre_nor).flatten()).flatten()
    final_name =  np.array(np.array(ff_name).flatten()).flatten()
    final_name = [str(m) for m in final_name]

    name = list(set(list(final_name)))
    final_pre_gyh = []
    final_lab_gyh = []
    final_pre_nor_gyh = []
    for n in name:
        id1 = [i for i,x in enumerate(final_name) if x==n]

        final_pre_gyh.append(sum(final_pre[id1])/len(final_pre[id1]))
        final_lab_gyh.append(round(sum(final_lab[id1])/len(final_lab[id1])))
        final_pre_nor_gyh.append(round(sum(final_pre_nor[id1])/len(final_pre_nor[id1])))         
    AUC = roc_auc_score(final_lab_gyh, final_pre_nor_gyh)
    confu = confusion_matrix(final_lab_gyh, final_pre_nor_gyh,labels=list(set(final_lab_gyh)))
    sens = confu[0][0]/(confu[0][0]+confu[0][1])
    spec = confu[1][1]/(confu[1][1]+confu[1][0])
    
    fpr,tpr,threshold = roc_curve(final_lab_gyh, final_pre_gyh,pos_label=1, drop_intermediate=False) ###计算真正率和假正率
    roc_auc = auc(fpr,tpr) ###计算auc的值
    acc = accuracy_score(final_lab_gyh, final_pre_nor_gyh)
    print('AUC:',roc_auc,'auc2:',AUC,'acc:',acc,'sens:',sens,'spec:',spec)
    font2 = {
    'size' : 16,'family': 'Arial','weight': 'bold',
    }
    
    plt.figure()
    lw = 3
    plt.figure(figsize=(7,7),dpi=800)
    plt.plot(fpr, tpr, color='red',
              lw=lw, label='AUC = %0.2f' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tick_params(labelsize=13,width=4)
    ax=plt.gca();#获得坐标轴的句柄
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(0);###设置右边坐标轴的粗细a
    ax.spines['top'].set_linewidth(0);####设置上部坐标轴的粗细
    plt.xlabel('1-Specificity',font2)
    plt.ylabel('Sensitivity',font2)
    # plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
else:
    func = lambda x: [y for l in x for y in func(l)] if type(x) is list else [x]
    model_path='/pretrain_model/model.pth'
    net = torch.load(model_path)
    net.eval()
    use_cuda = torch.cuda.is_available()
    test_loss = 0
    correct = 0
    correct_com = 0
    total = 0
    idx = 0
    batch_size = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
    net.to(device)
    transform_test = transforms.Compose([
        # transforms.Scale((512, 512)),
        # transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    testset = torchvision.datasets.ImageFolder(
        root='/independtest',
        transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    final_lab = []
    final_pre = []
    final_pre_nor = []
    final_name = []
    for batch_idx, (inputs, targets,patient,cp) in enumerate(testloader):
        idx = batch_idx
        inputs, targets,cp = inputs.to(device), targets.to(device),cp.to(device)
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        output_1, output_2, output_3, output_concat= net(inputs,cp)
        outputs_com = output_1 + output_2 + output_3 + output_concat
        pred = outputs_com.data[:,1]
        # _, pred = torch.max(output_concat.data, 1)
        _, predicted_com = torch.max(outputs_com.data, 1)
        lab = list(targets.data.cpu().numpy())
        pre = list(pred.data.cpu().numpy())
        pre_nor = list(predicted_com.data.cpu().numpy())
        final_lab.append(lab)
        final_pre.append(pre)
        final_pre_nor.append(pre_nor)
        patient = list(patient)
        final_name.append(patient)
    ff_lab = func(final_lab)
    ff_pre = func(final_pre)
    ff_pre_nor = func(final_pre_nor)
    ff_name = func(final_name)
    final_lab = np.array(ff_lab).flatten()
    final_pre = np.array(ff_pre).flatten()
    final_pre_nor = np.array(ff_pre_nor).flatten()
    final_name = np.array(ff_name).ravel()
    AUC = roc_auc_score(final_lab, final_pre_nor)
    confu = confusion_matrix(final_lab, final_pre_nor,labels=list(set(final_lab)))
    sens = confu[0][0]/(confu[0][0]+confu[0][1])
    spec = confu[1][1]/(confu[1][1]+confu[1][0])
    fpr,tpr,threshold = roc_curve(final_lab, final_pre,pos_label=1) ###计算真正率和假正率
    roc_auc2 = auc(fpr,tpr) ###计算auc的值
    acc = accuracy_score(final_lab, final_pre_nor)
    print('AUC:',AUC,'acc:',acc,'sens:',sens,'spec:',spec)
    plt.figure()
    lw = 2
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % AUC) ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
