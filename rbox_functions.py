import numpy as np
import ctypes
from ctypes import *
import math

so = ctypes.cdll.LoadLibrary
librbox = so("./librbox.so")
overlap = librbox.Overlap
overlap.argtypes = (POINTER(c_double),POINTER(c_double))
overlap.restype = c_double
DecodeAndNMS = librbox.DecodeAndNMS
DecodeAndNMS.argtypes = (POINTER(c_double),POINTER(c_double),POINTER(c_int),POINTER(c_double),POINTER(c_int),c_double)
DecodeAndNMS.restype = None
NMS = librbox.NMS_ship
NMS.argtypes=(POINTER(c_double),POINTER(c_int),POINTER(c_double),POINTER(c_int),c_double)
NMS.restype=None

def PriorRBox(height, width, fheight, fwidth, stepsize, prior_angles, prior_heights, prior_widths, offset=0.5):    
    priorbox = []
    priorbox = np.array(priorbox)    
    i = 0
    priorbox = np.zeros((fheight*fwidth*len(prior_angles)*len(prior_heights), 5))
    for h in range(fheight):
        center_y = (h + offset) * stepsize / height
        for w in range(fwidth):
            center_x = (w + offset) * stepsize / width
            for pa in range(len(prior_angles)):
                angle = prior_angles[pa]
                for pw in range(len(prior_widths)):
                    prior_width = prior_widths[pw] / width
                    prior_height = prior_heights[pw] / height
                    one_box = [center_x, center_y, prior_width, prior_height, angle]
                    one_box = np.array(one_box)
                    one_box = one_box[np.newaxis, :]
                    priorbox[i] = one_box
                    i += 1
    return priorbox
    
def MatchRBox(priorbox, groundtruth, overlap_threshold, is180):
    cpriorbox = (c_double * 5)()
    cgroundtruth = (c_double * 5)()
    overlaps = np.zeros((len(priorbox),len(groundtruth)))
    pmatches = np.zeros(len(priorbox)) - 1
    for i in range(len(priorbox)):
        for j in range(len(groundtruth)):
            for ii in range(5):
                cpriorbox[ii] = c_double(priorbox[i][ii])
                cgroundtruth[ii] = c_double(groundtruth[j][ii])
            cpriorbox[4] = cgroundtruth[4]
            if is180:
                overlaps[i,j] = overlap(cpriorbox, cgroundtruth) * abs(math.cos((priorbox[i][4] - groundtruth[j][4]) / 180 * math.pi))
            else:
                overlaps[i,j] = overlap(cpriorbox, cgroundtruth) * math.cos((priorbox[i][4] - groundtruth[j][4]) / 180 * math.pi)
    overlaps_tmp = overlaps.copy()
    for i in range(len(groundtruth)):
        max_overlap = np.where(overlaps_tmp == np.max(overlaps_tmp))
        index1 = max_overlap[0][0]
        index2 = max_overlap[1][0]
        pmatches[index1] = index2
        overlaps_tmp[index1, :] = -1
        overlaps_tmp[:, index2] = -1
    for i in range(len(pmatches)):
        if pmatches[i] > -1:
            continue
        overlaps_i = overlaps[i,:]
        max_overlap = np.max(overlaps_i)
        if max_overlap < overlap_threshold:
            continue
        max_overlap = np.where(overlaps_i == max_overlap)
        pmatches[i] = max_overlap[0][0]
    indice = np.where(pmatches > -1)[0]
    encodedbox = np.zeros((len(indice), 5))
    for i in range(len(indice)):
        ii = indice[i]
        encoded = EncodeRBox(priorbox[ii], groundtruth[pmatches[ii].astype(np.int32)])
        encodedbox[i] = encoded
    ind_one_hot = pmatches > -1
    return ind_one_hot, indice, encodedbox
    
def EncodeRBox(prior, ground):
    encoded = np.zeros(5)
    xd = ground[0] - prior[0]
    yd = ground[1] - prior[1]
    anglep = prior[4] / 180. * math.pi
    encoded[0] = xd * math.cos(anglep) + yd * math.sin(anglep)
    encoded[1] = xd * math.sin(anglep) - yd * math.cos(anglep)
    encoded[0] /= prior[2]
    encoded[1] /= prior[3]
    #encoded[0] = (ground[0] - prior[0]) / prior[2]
    #encoded[1] = (ground[1] - prior[1]) / prior[3]
    encoded[2] = math.log(ground[2] / prior[2])
    encoded[3] = math.log(ground[3] / prior[3])
    encoded[4] = math.tan((ground[4] - prior[4]) / 180. * math.pi)
    return encoded
    
def Original_EncodeRBox(prior, ground):
    encoded = np.zeros(5)
    encoded[0] = (ground[0] - prior[0]) / prior[2]
    encoded[1] = (ground[1] - prior[1]) / prior[3]
    encoded[2] = math.log(ground[2] / prior[2])
    encoded[3] = math.log(ground[3] / prior[3])
    encoded[4] = math.tan((ground[4] - prior[4]) / 180. * math.pi)
    return encoded    

def DecodeNMS(loc_preds_j, prior_boxes_j, conf_preds_j, inputloc_j, index, nms_threshold, heightOut, widthOut):
    prior_var = [0.1, 0.1, 0.2, 0.2, 0.1]
    rbox = []
    score = []
    if len(loc_preds_j) > 0:
        loc_c = (c_double * len(loc_preds_j))()
        prior_c = (c_double * len(prior_boxes_j))()
        conf_c = (c_double * len(conf_preds_j))()
        indices_c = (c_int * len(index))()
        for k in range(len(index)):
            loc_c[5*k] = c_double(loc_preds_j[5*k] * prior_var[0])
            loc_c[5*k+1] = c_double(loc_preds_j[5*k+1] * prior_var[1])
            loc_c[5*k+2] = c_double(loc_preds_j[5*k+2] * prior_var[2])
            loc_c[5*k+3] = c_double(loc_preds_j[5*k+3] * prior_var[3])
            loc_c[5*k+4] = c_double(loc_preds_j[5*k+4] * prior_var[4])
            indices_c[k] = c_int(-1)
            conf_c[k] = c_double(conf_preds_j[k])
        for k in range(len(index)*5):
            prior_c[k] = c_double(prior_boxes_j[k])        
        pind = cast(indices_c, POINTER(c_int))
        pconf = cast(conf_c, POINTER(c_double))
        num_preds = c_int(len(index))
        DecodeAndNMS(loc_c, prior_c, pind, pconf, byref(num_preds), c_double(nms_threshold))
        for k in range(num_preds.value):
            index_k = indices_c[k]
            area = loc_c[5*index_k + 2] * loc_c[5*index_k + 3] * heightOut * widthOut
            if area < 100 or area > 10000:
                continue
            rbox.append(loc_c[5*index_k] * widthOut * inputloc_j[2] + inputloc_j[0])
            rbox.append(loc_c[5*index_k + 1] * heightOut * inputloc_j[2] + inputloc_j[1])
            rbox.append(loc_c[5*index_k + 2] * widthOut * inputloc_j[2])
            rbox.append(loc_c[5*index_k + 3] * heightOut * inputloc_j[2])
            rbox.append(loc_c[5*index_k + 4])
            score.append(conf_c[index_k])
        return rbox, score                

def NMSOutput(rboxlist, scorelist, nms_threshold, label, test_rbox_output_path):
    loc_c = (c_double * len(rboxlist))()
    score_c = (c_double * len(scorelist))()
    indices_c = (c_int * len(scorelist))()
    for i in range(len(rboxlist)):
        loc_c[i] = c_double(rboxlist[i])
    for i in range(len(scorelist)):
        score_c[i] = c_double(scorelist[i])
        indices_c[i] = c_int(-1)
    num_preds = c_int(len(scorelist))
    NMS(loc_c, indices_c, score_c, byref(num_preds), c_double(nms_threshold))
    with open(test_rbox_output_path, 'w') as fid:
        for i in range(num_preds.value):
            index_i = indices_c[i]
            fid.write('{} {} {} {} {} {} {}\n'.format(loc_c[5*index_i], loc_c[5*index_i+1], loc_c[5*index_i+2], loc_c[5*index_i+3], label,
                       loc_c[5*index_i+4], score_c[index_i]))
