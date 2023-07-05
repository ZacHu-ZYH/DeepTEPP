import nrrd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import scipy.ndimage
import os
import SimpleITK as sitk
def rotate_bound(image, angle):

    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image

    return cv2.warpAffine(image, M, (nW, nH))

def MV(path):
    f = open(path, 'r')
    content = f.readlines()
    import datetime
    starttime = datetime.datetime.now()

    for index in range(len(content)):
        grid_out = np.zeros((512,512))
        try:
            print(index)
            temp = content[index]
            image_path = temp.strip('\n').split(',')[0]
            pred_path = temp.strip('\n').split(',')[1]
            classes = temp.strip('\n').split(',')[2:]
            classes = [int(cla) for cla in classes]
            print(image_path, pred_path, classes)
            # 512 512 108
            img_data, nrrd_options = nrrd.read(image_path)
            pred_data, nrrd_options = nrrd.read(pred_path)
            
    #            for iii in [50,100,150,200,250,300,350]:
            angle_list = range(1,11)
            angle_list_36 = [m*36 for m in angle_list]
            for iii in angle_list_36:
                new_arr = [[]]
                new_pred = [[]]
                for dd in range(img_data.shape[1]):
                    img = img_data[:,dd,:]
                    pred = pred_data[:,dd,:]
                    img = rotate_bound(img,iii)
                    pred = rotate_bound(pred,iii)
    #                img = cv2.resize(img, (256, 256))
                    if new_lab == [[]]:
                        new_pred = np.expand_dims(pred,axis=1)
                    else:
                        new_pred = np.concatenate((new_pred, np.expand_dims(pred,axis=1)), axis=1)
                    if new_arr == [[]]:
                        new_arr = np.expand_dims(img,axis=1)
                    else:
                        new_arr = np.concatenate((new_arr, np.expand_dims(img,axis=1)), axis=1)
                cnt = 0
                img_total = [[]]
                for i in range(new_pred.shape[2]):
                    if np.max(new_pred[:, :, i])==0:
                        continue
                    else:
                        pd_data = cv2.resize(new_pred[:, :, i],(512,512))
                        img_dt = cv2.resize(new_arr[:,:,i],(512,512))
                        img_dt = (img_dt - np.min(img_dt)) / (np.max(img_dt) - np.min(img_dt)) * 255
                        pd_data = np.array(pd_data,dtype = np.uint8)

                        x, y, w, h = cv2.boundingRect(pd_data)
                        img = cv2.resize(img_dt[y-5:y + h+10, x-5:x + w+10],(64,64))
                        
                        if cnt//8==0:
                            grid_out[:64,cnt%8*64:cnt%8*64+64] = img
                        elif cnt//8==1:
                            grid_out[64:128,cnt%8*64:cnt%8*64+64] = img
                        elif cnt//8==2:
                            grid_out[128:192,cnt%8*64:cnt%8*64+64] = img
                        elif cnt//8==3:
                            grid_out[192:256,cnt%8*64:cnt%8*64+64] = img
                        elif cnt//8==4:
                            grid_out[256:320,cnt%8*64:cnt%8*64+64] = img
                        elif cnt//8==5:
                            grid_out[320:384,cnt%8*64:cnt%8*64+64] = img
                        elif cnt//8==6:
                            grid_out[384:448,cnt%8*64:cnt%8*64+64] = img
                        elif cnt//8==7:
                            grid_out[448:512,cnt%8*64:cnt%8*64+64] = img
                        if img_total ==[[]]:
                            img_total = np.expand_dims(img,axis=0)
                        else:
                            img_total = np.concatenate((img_total, np.expand_dims(img,axis=0)), axis=0)
                            
                        cnt+=1

                for d in range(cnt,64):
                    if d//8==0:
                        grid_out[:64,d%8*64:d%8*64+64] = img_total[d%cnt]
                    elif d//8==1:
                        grid_out[64:128,d%8*64:d%8*64+64] = img_total[d%cnt]
                    elif d//8==2:
                        grid_out[128:192,d%8*64:d%8*64+64] = img_total[d%cnt]
                    elif d//8==3:
                        grid_out[192:256,d%8*64:d%8*64+64] = img_total[d%cnt]
                    elif d//8==4:
                        grid_out[256:320,d%8*64:d%8*64+64] = img_total[d%cnt]
                    elif d//8==5:
                        grid_out[320:384,d%8*64:d%8*64+64] = img_total[d%cnt]
                    elif d//8==6:
                        grid_out[384:448,d%8*64:d%8*64+64] = img_total[d%cnt]
                    elif d//8==7:
                        grid_out[448:512,d%8*64:d%8*64+64] = img_total[d%cnt]
                cv2.imwrite('%s' % (os.path.split(image_path)[-1][:-5], iii), grid_out)
        except:
            pass
        endtime = datetime.datetime.now()
        print(endtime - starttime)
path = ''
MV(path_A)