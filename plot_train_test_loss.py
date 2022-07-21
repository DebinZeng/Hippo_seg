#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Copyright (c) 2018
@author: Debin Zeng
"""

import numpy as np
import os
import glob
import sys
import matplotlib.pyplot as plt
import math
import re
from mpl_toolkits.axes_grid1 import host_subplot
from pylab import figure, show, legend


fig=plt.figure(figsize=(30, 16))
host= fig.add_subplot(1,1,1)
plt.subplots_adjust(right=0.8) #adjust the right boundary of the plot window
#parl=host.twinx()

#set labels
host.set_xlabel("iterations")
host.set_ylabel("log loss")
#parl.set_ylabel("validation accuracy")
 

for path in glob.glob(os.path.join(os.path.abspath("./3D-DCFCN-BC-43"), "training_log*", "caffe.omnisky.omnisky.log.INFO.*")):

    print path
    fp=open(path,'r')

    train_iter=[]
    train_loss=[]
    test_iter=[]
    test_accuracy=[]
    
    for ln in fp:
        #get train_iterations and train_loss
        if '] Iteration' in ln and 'loss =' in ln:
            arr=re.findall(r'ion \b\d+\b,',ln)
            train_iter.append(int(arr[0].strip(',')[4:]))
            train_loss.append(float(ln.strip().split('=')[-1]))
            
    #      #get test_iterations
    #      if '] Iteration' in ln and 'Testing net (#0)' in ln:
    #          arr=re.findall(r'ion \b\d+\b,',ln)
    #          test_iter.append(int(arr[0].strip(',')[4:]))
    #      
    #      #get test_accuracy
    #      if '#2:' in ln and 'loss/top-5' in ln:
    #          test_accuracy.append(float(ln.strip().split('=')[-1]))
    
            
    fp.close()
    
    up1dir,basename=os.path.split(path)
    up2dir,basename=os.path.split(up1dir)
    
    #plot curves
    p1,=host.plot(train_iter,train_loss,label=basename)
    #p2,=host.plot(test_iter,test_accuracy,label="validation accuracy")
    
    #set location of legend
    #1:rightup corner 2:leftup corner 3:leftdown corner 
    #4:rightdown corner 5:rightmid
    host.legend(loc=5,fontsize='xx-large')
    
    #set label color
#    host.axis["left"].label.set_color(p1.get_color())
    #parl.axis["right"].label.set_color(p2.get_color())
    
    #set the range of x axis of host and y axis of parl
    host.set_xlim([0,30000])
    host.set_ylim([0.,0.3])
    #parl.set_ylim([0.,1.05])
    
plt.draw()
plt.show()
















