from __future__ import print_function
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
torch.backends.cudnn.bencmark = True
from PIL import Image

import os,sys,random,datetime,time,math
import argparse
import numpy as np

import net_s3fd

def detect(net,img):
    BB,CC,HH,WW = img.size()
    olist = net(img)

    bboxlist = []
    for i in range(len(olist)/2): olist[i*2] = F.softmax(olist[i*2])
    for i in range(len(olist)/2):
        ocls,oreg = olist[i*2].data.float().cpu(),olist[i*2+1].data.float().cpu()
        FB,FC,FH,FW = ocls.size() # feature map size
        stride = 2**(i+2)    # 4,8,16,32,64,128
        anchor = stride*4
        for Findex in range(FH*FW):
            windex,hindex = Findex%FW,Findex//FW
            axc,ayc = stride/2+windex*stride,stride/2+hindex*stride
            score = ocls[0,1,hindex,windex]
            loc = oreg[0,:,hindex,windex].contiguous().view(1,4)
            if score<0.05: continue
            priors = torch.Tensor([[axc/1.0,ayc/1.0,stride*4/1.0,stride*4/1.0]])
            variances = [0.1,0.2]
            box = decode(loc,priors,variances)
            x1,y1,x2,y2 = box[0]*1.0
            # cv2.rectangle(imgshow,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),1)
            bboxlist.append([x1,y1,x2,y2,score])
    bboxlist = np.array(bboxlist)
    if 0==len(bboxlist): bboxlist=np.zeros((1, 5))
    return bboxlist

parser = argparse.ArgumentParser(description='PyTorch face detect')
parser.add_argument('--net','-n', default='s3fd', type=str)
parser.add_argument('--model', default='', type=str)
parser.add_argument('--path', default='CAMERA', type=str)
parser.add_argument('--onnx', default='', type=str)
parser.add_argument("--half", action='store_true',
                                 help="if set, use fp16 (on gpu)")

args = parser.parse_args()
use_cuda = torch.cuda.is_available()


net = getattr(net_s3fd,args.net)(fp16 = args.half)
if args.model!='' :net.load_state_dict(torch.load(args.model))
else: print('Please set --model parameter!')
net.cuda()
net.eval()

frame=0
# if args.path=='CAMERA': cap = cv2.VideoCapture(0)
while(True):
#     if args.path=='CAMERA': ret, img = cap.read()
#    else: img = cv2.imread(args.path)
    img = Image.open(args.path)
    imgshow = np.copy(img)

    img = img - np.array([104,117,123])
    img = img.transpose(2, 0, 1)
    img = img.reshape((1,)+img.shape)
    img = Variable(torch.from_numpy(img).float(), requires_grad=False).cuda()
    
    if args.half:
        img = img.half()
        net = net.cuda().half()
    
    if args.onnx!='':
        torch.onnx.export(net, img, args.onnx)
        exit(0)

    import cv2
    from bbox import *
    t0 = time.time()
    bboxlist = detect(net,img)
    print('Frame: %d, faces: %d time : %s'%(frame, bboxlist.size, time.time()-t0))
    ++frame 
    img = img.float()
    keep = nms(bboxlist,0.3)
    bboxlist = bboxlist[keep,:]
    for b in bboxlist:
        x1,y1,x2,y2,s = b
        if s<0.5: continue
        cv2.rectangle(imgshow,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),1)

    cv2.imshow('test',imgshow)
    
    if args.path=='CAMERA':
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    else:
        cv2.imwrite(args.path[:-4]+'_output.png',imgshow)
        if cv2.waitKey(0) or True: break
