# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 09:30:34 2018

@author: 29412
"""
from glob import glob 
import os.path
import pickle
import numpy as np
import math
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import ctypes
from ctypes import *


IM_WIDTH  = 300  
IM_HEIGHT = 300  #consistent with the settings in the main program
so = ctypes.cdll.LoadLibrary
librbox = so("./librbox.so")
overlap = librbox.Overlap
OVERLAP_THRESHOLD= 0.3
overlap.argtypes = (POINTER(c_double),POINTER(c_double))
overlap.restype  =  c_double
route_test = "./data/test" 
route_result = "./data/result" 
retain_threshold = 0  #Only the detection results with the confidence over this threshold are calculated.

def cal_true_number(route = ''):
    files = glob(os.path.join(route, '*.rbox'))
    num_of_targets = 0
    for i in files:
        with open(i, 'rb') as fid:
            num_of_targets += len(fid.readlines())
    print("The number of all targets is: {}".format(num_of_targets))
    return num_of_targets
	
def cal_pr(route, route_result):
    ''' generate a file (all_figures.txt) to caculate the precision and the recall'''
    fid      = open(os.path.join(route_result, 'all_figures.txt'),'w')
    files    = glob(os.path.join(route, '*.rbox'))
    true_conf= {}
    true_idx = {}
    true_iou = {}
    idx_sort = {}
    true_num = 0
    for idx in range(len(files)):
        min_conf = 1
        i = 0
        detected_file = glob(os.path.join(route_result, os.path.basename(files[idx]) + '*.score'))
        if len(detected_file) == 0:
            print ("No score file!")
            continue
        with open(files[idx], 'r') as infile:
            print files[idx]
            if infile == None:
                continue
            for line in infile:
                true_num += 1
                rbox = []                    
                ii = 0
                for rbox_param in line.split(' '):
                    if ii == 0 or ii == 2: # center x or width
                        rbox.append(eval(rbox_param)/IM_WIDTH)
                    elif ii == 1 or ii == 3: # center y or height
                        rbox.append(eval(rbox_param)/IM_HEIGHT)
                    elif ii == 5:#angle
                        rbox.append(eval(rbox_param))
                    ii += 1
                rbox = np.array(rbox)
                rbox = rbox[np.newaxis, :]
                if i == 0:
                    gt_box = rbox
                else:
                    gt_box = np.concatenate((gt_box, rbox), axis=0)#np.empty
                i += 1
        i = 0
        conf = []
        res_box = []
        with open(detected_file[0], 'r') as infile:
            for line in infile:
                rbox = []
                ii = 0
                for rbox_param in line.split(' '):
                    if ii == 0 or ii == 2: # center x or width
                        rbox.append(eval(rbox_param)/IM_WIDTH)
                    elif ii == 1 or ii == 3: # center y or height
                        rbox.append(eval(rbox_param)/IM_HEIGHT)
                    elif ii == 5:#angle
                        rbox.append(eval(rbox_param))
                    elif ii == 6:#conf
                        conf.append(eval(rbox_param))
                    ii += 1
                rbox = np.array(rbox)
                rbox = rbox[np.newaxis, :]
                if i == 0:
                    res_box = rbox
                else:
                    res_box = np.concatenate((res_box, rbox), axis=0)#np.empty
                i += 1           
        cpriorbox = (c_double * 5)()
        cgroundtruth = (c_double * 5)()
        if len(res_box) == 0:
            continue
        overlaps = np.zeros((len(res_box),len(gt_box)))
        for i in range(len(res_box)):
            for j in range(len(gt_box)):
                for ii in range(5):
                    cpriorbox[ii] = c_double(res_box[i][ii])
                    cgroundtruth[ii] = c_double(gt_box[j][ii])
                cpriorbox[4] = cgroundtruth[4]
                overlaps[i,j] = overlap(cpriorbox, cgroundtruth)
        [res_idx, gt_idx] = np.where(overlaps > OVERLAP_THRESHOLD)
        true_idx[idx] = res_idx
        for i in range(len(conf)):
            if i in res_idx:
                fid.write(str(conf[i]))
                fid.write(' 1\n')
            else:
                fid.write(str(conf[i]))
                fid.write(' 0\n')
        scores = np.array([])
        for i in range(len(res_idx)):
            scores = np.append(scores, overlaps[res_idx[i]][gt_idx[i]])
        order = scores.copy().argsort()
        true_conf[idx] = conf
        true_iou[idx]  = scores #iou
        idx_sort[idx]  = order
        print(str(idx+1)+'/'+str(len(files)))
    fid.close()
    #return true_idx, true_conf, true_iou, idx_sort
	
def pr_curve(route=''):
    #route = '/data/t-data/data/mean30/cal'
    with open(os.path.join(route, 'all_figures.txt'), 'r') as fid:
        true_and_false = fid.readlines()
    figure_result = np.zeros((len(true_and_false), 2))
    for idx in range(len(true_and_false)):
        figure_result[idx] =  [float(true_and_false[idx].split()[i]) for i in range(2)]
    all_true_num = cal_true_number(route_test)
    pr_rec, pr_pre = Cal_index(figure_result,all_true_num, route)
    with open(os.path.basename(route)+'.txt','w') as fid:
		for i in range(len(pr_rec)):
			fid.write(str(pr_rec[i]))
			fid.write(' ')
			fid.write(str(pr_pre[i]))
			fid.write('\n')
    fug = plt.figure()
    plt.plot(pr_rec, pr_pre, '.-')
    plt.grid()
    plt.show()
    print 'Painting...'
    plt.savefig(os.path.join(route, 'pr_curve.png'))
    bep = BEP(pr_rec, pr_pre)
    aupr = AUPR(pr_rec, pr_pre)
    print 'BEP:', bep
    print 'AP:', aupr
    
def Cal_index(figure_result, all_true_num, route):
    if len(figure_result) == 0:
        pr_rec = 0
        pr_pre = 0
        return pr_rec, pr_pre
    conf = figure_result[:,0]
    istru = figure_result[:,1]
    conf_u = list(set(conf))
    conf_u = np.array(conf_u)
    conf_u = np.sort(conf_u)[::-1]
    pr_pre = np.zeros((len(conf_u)))
    pr_rec = np.zeros((len(conf_u)))
    for i in range(len(conf_u)):
        if conf_u[i] <= retain_threshold:
            break
        idx = np.where(conf >= conf_u[i])[0]
        true_pos = sum(istru[idx] == 1)
        false_pos = sum(istru[idx] == 0)
        pr_pre[i] = float(true_pos) / (true_pos+false_pos)
        pr_rec[i] = float(true_pos) / all_true_num
    for i in range(len(conf_u)-2,0-1,-1):
        if pr_pre[i] < pr_pre[i+1]:
            pr_pre[i] = pr_pre[i+1]
    with open(os.path.join(route, 'precision_recall.txt'),'w') as fid:
        print os.path.join(route, 'precision_recall.txt')
        for i in range(len(pr_rec)):
            fid.write(str(pr_rec[i]))
            fid.write(' ')
        fid.write('\n')
        for i in range(len(pr_pre)):
            fid.write(str(pr_pre[i]))
            fid.write(' ')       
    return pr_rec, pr_pre
      
def BEP(pr_rec, pr_pre):
    pr_rec = np.squeeze(pr_rec[::-1])
    pr_pre = np.squeeze(pr_pre[::-1])
    interval_rec_pre = pr_rec[0]-pr_pre[0]
    if interval_rec_pre < 0:
		print "The recall rate is too low!\n"
		return -1
    bep = 0
    for i in range(1, len(pr_rec)):
		if pr_rec[i]-pr_pre[i]< 0:
			break
		if pr_rec[i]-pr_pre[i] < interval_rec_pre:
			interval_rec_pre = pr_rec[i]-pr_pre[i]
			bep = pr_rec[i]
    return bep

def AUPR(pr_rec, pr_pre):
    Area_under_pr = 0
    rec_ascend = np.insert(pr_rec, 0, 0)
    pre_descend = np.sort(pr_pre)[::-1]
    for i in range(len(pre_descend)):
        if rec_ascend[i] <= 1.0:
            Area_under_pr += (pre_descend[i] * (rec_ascend[i+1]-rec_ascend[i]))
        else:
            break
    return Area_under_pr


if __name__ == '__main__':
    cal_pr(route_test, route_result)
    pr_curve(route_result)
