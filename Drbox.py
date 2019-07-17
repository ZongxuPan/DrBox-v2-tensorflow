import os
import os.path
import sys
import random
import numpy as np
from glob import glob
import tensorflow as tf
from model import *
from rbox_functions import *
import scipy.misc
import pickle

TXT_DIR = './data' 
INPUT_DATA_PATH = TXT_DIR + '/train'
TEST_DATA_PATH = TXT_DIR + '/test'
PRETRAINED_NET_PATH = "./vgg16.npy"
SAVE_PATH = './result' 
TRAIN_BATCH_SIZE = 8
IM_HEIGHT = 300
IM_WIDTH = 300
IM_CDIM = 3
FEA_HEIGHT4 = 38
FEA_WIDTH4 = 38
FEA_HEIGHT3 = 75
FEA_WIDTH3 = 75
STEPSIZE4 = 8
STEPSIZE3 = 4


PRIOR_ANGLES = [0, 30, 60, 90, 120, 150]
PRIOR_HEIGHTS =[[4.0, 7.0, 10.0, 13.0],[3.0,8.0,12.0,17.0,23.0]] #[3.0,8.0,12.0,17.0,23.0] #
PRIOR_WIDTHS = [[15.0, 25.0, 35.0, 45.0],[20.0,35.0,50.0,80.0,100.0]]#[20.0,35.0,50.0,80.0,100.0]  


ITERATION_NUM = 50000 
OVERLAP_THRESHOLD = 0.5
IS180 = False
NP_RATIO = 3
LOC_WEIGHTS = [0.1, 0.1, 0.2, 0.2, 0.1]
LOAD_PREVIOUS_POS = False
WEIGHT_DECAY = 0.0005
DISPLAY_INTERVAL = 100 #100
SAVE_MODEL_INTERVAL = 2000
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # select the used GPU
TEST_BATCH_SIZE = 1
TEST_RESOLUTION_IN = 3
TEST_RESOLUTION_OUT = [3]
TEST_SCORE_THRESHOLD = 0.2
TEST_NMS_THRESHOLD = 0.1
TEST_HEIGHT_STEP = 0.85
TEST_WIDTH_STEP = 0.85
flags = tf.app.flags
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
FLAGS = flags.FLAGS

USE_THIRD_LAYER = 1
FPN_NET = 1
USE_FOCAL_LOSS = 1
focal_loss_factor = 2.5

class DrBoxNet():
    def __init__(self):                
        for stage in ['train', 'test']:
            self.get_im_list(stage)
        self.learning_rate = 0.001
        self.momentum = 0.9
        self.global_step = tf.Variable(0, trainable=False)        
        self.model_save_path = os.path.join(SAVE_PATH, 'model')
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        self.get_encoded_positive_box()
        random.shuffle(self.train_im_list)
        self.train_list_idx = 0        
        self.input_im = tf.placeholder(tf.float32, shape=[None, IM_HEIGHT, IM_WIDTH, IM_CDIM])
        #self.input_idx = tf.placeholder(tf.int32, shape=[None])
        self.prior_num = [len(PRIOR_ANGLES)*len(PRIOR_WIDTHS[0]), len(PRIOR_ANGLES)*len(PRIOR_WIDTHS[1])]
        self.total_prior_num = FEA_HEIGHT4*FEA_WIDTH4*self.prior_num[1]+FEA_HEIGHT3*FEA_WIDTH3*self.prior_num[0]*USE_THIRD_LAYER        
        self.para_num = 5
        self.cls_num = 1
        self.batch_pos_box = tf.placeholder(tf.float32, shape=[None, self.para_num])
        self.batch_pos_idx = tf.placeholder(tf.int32, shape=[None])
        self.batch_pos_ind = tf.placeholder(tf.float32, shape=[None])
        #self.batch_pos_num = tf.placeholder(tf.int32, shape=[None])
        self.batch_neg_mask = tf.placeholder(tf.float32, shape=[None])
        self.pos_label = tf.placeholder(tf.float32, shape=[None, self.cls_num + 1])
        self.neg_label = tf.placeholder(tf.float32, shape=[None, self.cls_num + 1])
        if FLAGS.train:
        		self.detector = VGG16(self.prior_num, self.para_num, self.cls_num, FPN_NET, USE_THIRD_LAYER, TRAIN_BATCH_SIZE)
        else:
        		self.detector = VGG16(self.prior_num, self.para_num, self.cls_num, FPN_NET, USE_THIRD_LAYER, TEST_BATCH_SIZE)
        self.loc, self.conf = self.detector(self.input_im)
        self.conf_softmax = tf.nn.softmax(self.conf)
        self.hard_negative_mining()
        self.compute_conf_loss()
        self.compute_loc_loss()        
        self.reg_loss = tf.add_n(self.detector.regular_loss(WEIGHT_DECAY))
        self.loss = self.loc_loss + self.conf_loss #+ self.reg_loss
        self.saver = tf.train.Saver(max_to_keep=5)
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            
    def compute_conf_loss(self):
        pos_tensor = tf.gather(self.conf, self.batch_pos_idx)
        neg_tensor = tf.gather(self.conf, self.batch_neg_idx)
        self.pos_tensor = pos_tensor
        self.neg_tensor = neg_tensor
        if USE_FOCAL_LOSS:
		        pos_prob = tf.slice(tf.nn.softmax(pos_tensor),[0,1],[-1,1])
		        neg_prob = tf.slice(tf.nn.softmax(neg_tensor),[0,0],[-1,1])
		        self.conf_pos_losses = tf.nn.softmax_cross_entropy_with_logits(logits=pos_tensor, labels=self.pos_label)
		        self.conf_neg_losses = tf.nn.softmax_cross_entropy_with_logits(logits=neg_tensor, labels=self.neg_label)
		        self.conf_pos_loss = tf.reduce_mean(tf.multiply((1-pos_prob)**focal_loss_factor, self.conf_pos_losses))
		        self.conf_neg_loss = tf.reduce_mean(tf.multiply((1-neg_prob)**focal_loss_factor, self.conf_neg_losses))
        else:
        		self.conf_pos_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pos_tensor, labels=self.pos_label))
        		self.conf_neg_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=neg_tensor, labels=self.neg_label) * self.batch_neg_mask)
        self.conf_loss = self.conf_pos_loss + self.conf_neg_loss
    
    def compute_loc_loss(self):
        loc_tensor = tf.gather(self.loc, self.batch_pos_idx)
        self.loc_tensor = loc_tensor
        loc_diff = tf.add(loc_tensor, -1*self.batch_pos_box)
        loc_diff = tf.abs(loc_diff)
        loc_l1_smooth = tf.where(tf.greater(loc_diff, 1.0), loc_diff - 0.5, tf.square(loc_diff) * 0.5)
        self.loc_loss = tf.reduce_mean(loc_l1_smooth)
    
    def hard_negative_mining(self):
        conf = self.conf_softmax
        conf = tf.transpose(conf)
        conf = tf.slice(conf, [0, 0], [1, self.total_prior_num*TRAIN_BATCH_SIZE])
        conf = tf.squeeze(conf)
        conf = -1*tf.add(conf, self.batch_pos_ind)
        for batch_idx in range(TRAIN_BATCH_SIZE):
            batch_slice = tf.slice(conf, [batch_idx*self.total_prior_num], [self.total_prior_num])
            neg_top_k = tf.nn.top_k(batch_slice, self.max_neg_num)
            neg_idx = neg_top_k.indices + batch_idx*self.total_prior_num
            neg_idx = tf.squeeze(neg_idx)
            if batch_idx == 0:
                self.batch_neg_idx = neg_idx
            else:
                self.batch_neg_idx = tf.concat([self.batch_neg_idx, neg_idx], 0)
    
    def get_im_list(self, stage):        
        if stage == 'train': 
            infile = open(os.path.join(TXT_DIR, 'train.txt'))
            self.train_im_list = []
            k = 0
            for line in infile:
                line = line.strip()
                line = str(k) + ' ' + line
                self.train_im_list.append(line)
                k += 1
                if k == 5120:
                    break                        
            infile.close()
            self.train_im_num = len(self.train_im_list)
        else:
            infile = open(os.path.join(TXT_DIR, 'test.txt'))
            self.test_im_list = []
            for line in infile:
                self.test_im_list.append(line)            
            infile.close()
            self.test_im_num = len(self.test_im_list)

    def get_encoded_positive_box(self):        
        prior_box4 = PriorRBox(IM_HEIGHT, IM_WIDTH, FEA_HEIGHT4, FEA_WIDTH4, STEPSIZE4, PRIOR_ANGLES, PRIOR_HEIGHTS[1], PRIOR_WIDTHS[1])
        prior_box3 = PriorRBox(IM_HEIGHT, IM_WIDTH, FEA_HEIGHT3, FEA_WIDTH3, STEPSIZE3, PRIOR_ANGLES, PRIOR_HEIGHTS[0], PRIOR_WIDTHS[0])
        if USE_THIRD_LAYER:            
            prior_box = np.concatenate((prior_box3, prior_box4), axis=0)
        else:
            prior_box = prior_box4            
        self.prior_box = prior_box
        self.ind_one_hot = {}
        self.positive_indice = {}
        self.encodedbox = {}
        self.pos_num = {}        
        self.max_neg_num = 0
        if not FLAGS.train:
            return
        if LOAD_PREVIOUS_POS:
            with open(os.path.join(INPUT_DATA_PATH, 'ind_one_hot.pkl'),'rb') as fid:
                self.ind_one_hot = pickle.load(fid)
            with open(os.path.join(INPUT_DATA_PATH, 'positive_indice.pkl'),'rb') as fid:
                self.positive_indice = pickle.load(fid)
            with open(os.path.join(INPUT_DATA_PATH, 'encodedbox.pkl'),'rb') as fid:
                self.encodedbox = pickle.load(fid)
        for k in range(self.train_im_num):
            if k % 100 == 0:
                print('Preprocessing {}'.format(k)) 
            im_rbox_info = self.train_im_list[k]
            im_rbox_info = im_rbox_info.split(' ')
            idx = eval(im_rbox_info[0])
            rbox_fn = im_rbox_info[2]
            rbox_path = os.path.join(INPUT_DATA_PATH, rbox_fn)
            rboxes = []
            rboxes = np.array(rboxes)
            i = 0
            with open(rbox_path, 'r') as infile:
                for line in infile:
                    rbox = []
                    ii = 0
                    for rbox_param in line.split(' '):
                        if ii == 0 or ii == 2: # center x or width
                            rbox.append(eval(rbox_param)/IM_WIDTH)
                        elif ii == 1 or ii == 3: # center y or height
                            rbox.append(eval(rbox_param)/IM_HEIGHT)
                        elif ii == 5:
                            rbox.append(eval(rbox_param))
                        ii += 1
                    rbox = np.array(rbox)
                    rbox = rbox[np.newaxis, :]
                    if i == 0:
                        gt_box = rbox
                    else:
                        gt_box = np.concatenate((gt_box, rbox), axis=0)
                    i += 1                                                        
            if not LOAD_PREVIOUS_POS:
                self.ind_one_hot[idx], self.positive_indice[idx], self.encodedbox[idx] = MatchRBox(prior_box, gt_box, OVERLAP_THRESHOLD, IS180)
                self.encodedbox[idx] /= LOC_WEIGHTS
            self.pos_num[idx] = len(self.positive_indice[idx])
            if self.max_neg_num < self.pos_num[idx]:
                self.max_neg_num = self.pos_num[idx]
        self.max_neg_num *= NP_RATIO
        if not LOAD_PREVIOUS_POS: 
            with open(os.path.join(INPUT_DATA_PATH, 'ind_one_hot.pkl'),'wb') as fid:
                pickle.dump(self.ind_one_hot, fid)
            with open(os.path.join(INPUT_DATA_PATH, 'positive_indice.pkl'),'wb') as fid:
                pickle.dump(self.positive_indice, fid)
            with open(os.path.join(INPUT_DATA_PATH, 'encodedbox.pkl'),'wb') as fid:
                pickle.dump(self.encodedbox, fid)

    def get_next_batch_list(self):    
        idx = self.train_list_idx        
        if idx + TRAIN_BATCH_SIZE > self.train_im_num:
            batch_list = np.arange(idx, self.train_im_num)
            # shuffle the data in one category
            random.shuffle(self.train_im_list)
            new_list = np.arange(0, TRAIN_BATCH_SIZE-(self.train_im_num-idx))
            batch_list = np.concatenate((batch_list, new_list))
            self.train_list_idx = TRAIN_BATCH_SIZE-(self.train_im_num-idx)
        else:
            batch_list = np.arange(idx, idx+TRAIN_BATCH_SIZE)
            self.train_list_idx = idx+TRAIN_BATCH_SIZE
        return batch_list

    def train(self):
        #train_step = tf.train.MomentumOptimizer(self.learning_rate, self.momentum).minimize(self.loss, global_step=self.global_step)
        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)
        self.sess.run(tf.global_variables_initializer())
        
        # load the model if there is one
        could_load, checkpoint_counter = self.load()
        if could_load:
            self.sess.run(self.global_step.assign(checkpoint_counter))
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load the pretrained network FINISHED")
        
        for iter_num in range(ITERATION_NUM+1):
            input_im = np.zeros((TRAIN_BATCH_SIZE, IM_HEIGHT, IM_WIDTH, IM_CDIM))
            input_im = input_im.astype('float32')
            batch_list = self.get_next_batch_list()
            batch_pos_box = []
            batch_pos_box = np.array(batch_pos_box)
            batch_pos_ind = []
            batch_pos_ind = np.array(batch_pos_ind)
            batch_pos_idx = []
            batch_pos_idx = np.array(batch_pos_idx)
            batch_pos_num = []
            batch_pos_num = np.array(batch_pos_num)
            batch_neg_mask = np.zeros(TRAIN_BATCH_SIZE*self.max_neg_num)
            k = 0            
            for batch_idx in batch_list:
                im_rbox_info = self.train_im_list[batch_idx]
                im_rbox_info = im_rbox_info.split(' ')
                real_idx = eval(im_rbox_info[0])
                #input_idx[k] = real_idx
                im = scipy.misc.imread(os.path.join(INPUT_DATA_PATH, im_rbox_info[1]))
                imm = np.zeros((IM_HEIGHT, IM_WIDTH, IM_CDIM))
                if len(im.shape) == 2:
                    for ij in range(IM_CDIM):
                        imm[:,:,ij] = im
                    im = imm
                input_im[k] = im.reshape(IM_HEIGHT, IM_WIDTH, IM_CDIM).astype('float32')
                # select all or part of regression parameters in furture (to be done)
                if k==0:
                    batch_pos_box = self.encodedbox[real_idx]
                    batch_pos_ind = self.ind_one_hot[real_idx]
                    batch_pos_idx = self.positive_indice[real_idx]
                    batch_pos_num = [self.pos_num[real_idx]]                
                else:
                    batch_pos_box = np.concatenate((batch_pos_box, self.encodedbox[real_idx]), axis=0)
                    batch_pos_ind = np.concatenate((batch_pos_ind, self.ind_one_hot[real_idx]), axis=0)
                    batch_pos_idx = np.concatenate((batch_pos_idx, self.positive_indice[real_idx]+k*self.total_prior_num), axis=0)                    
                    batch_pos_num = np.concatenate((batch_pos_num, [self.pos_num[real_idx]]), axis=0)
                batch_neg_mask[k*self.max_neg_num:k*self.max_neg_num+self.pos_num[real_idx]*NP_RATIO] = 1.0
                #self.batch_pos_num[k] = self.pos_num[real_idx]
                #self.batch_neg_num[k] = self.batch_pos_num[k] * NP_RATIO
                k += 1
            batch_pos_ind = batch_pos_ind.astype('float32')
            total_batch_pos_num = np.sum(batch_pos_num)
            #total_batch_neg_num = total_batch_pos_num * NP_RATIO
            total_batch_neg_num = TRAIN_BATCH_SIZE * self.max_neg_num
            total_batch_pos_num = total_batch_pos_num.astype('int32')
            #total_batch_neg_num = total_batch_neg_num.astype('int32')
            batch_neg_mask *= (1.0 * total_batch_neg_num / total_batch_pos_num)
            #print('total_batch_neg_num {}, total_batch_pos_num {}'.format(total_batch_neg_num, total_batch_pos_num))
            #batch_neg_mask *= 1
            pos_label = np.zeros((total_batch_pos_num, 2))
            pos_label[:,1] = 1
            neg_label = np.zeros((total_batch_neg_num, 2))
            neg_label[:,0] = 1
            
            counter = self.sess.run(self.global_step)
            if counter > 80000:
                self.learning_rate = 0.0001
            if counter > 100000:
                self.learning_rate = 0.00001
            if counter > 120000:
                self.learning_rate = 0.000001
            self.sess.run(train_step, feed_dict={self.input_im:input_im, self.batch_pos_box:batch_pos_box, self.batch_pos_ind:batch_pos_ind,
                        self.batch_pos_idx:batch_pos_idx, self.batch_neg_mask:batch_neg_mask, self.pos_label:pos_label, self.neg_label:neg_label})
            if counter % DISPLAY_INTERVAL == 0:
                loss, loc_loss, conf_loss, conf_pos_loss, conf_neg_loss, reg_loss = self.sess.run([
                            self.loss, self.loc_loss, self.conf_loss, self.conf_pos_loss, self.conf_neg_loss, self.reg_loss],
                            feed_dict={self.input_im:input_im,
                            self.batch_pos_box:batch_pos_box, self.batch_pos_ind:batch_pos_ind, self.batch_pos_idx:batch_pos_idx, self.batch_neg_mask:batch_neg_mask,
                            self.pos_label:pos_label, self.neg_label:neg_label})
                with open(SAVE_PATH + '/loss.txt', 'ab+') as files:
                		files.write(("counter:[%2d], loss:%.8f, loc_loss:%.8f, conf_loss:%.8f, conf_pos_loss:%.8f, conf_neg_loss:%.8f, reg_loss:%.8f") % (counter, loss, loc_loss,conf_loss, conf_pos_loss, conf_neg_loss, reg_loss))
                		files.write('\n')
                print("counter:[%2d], loss:%.8f, loc_loss:%.8f, conf_loss:%.8f, conf_pos_loss:%.8f, conf_neg_loss:%.8f, reg_loss:%.8f") % (counter, loss, loc_loss,
                            conf_loss, conf_pos_loss, conf_neg_loss, reg_loss)

            if counter % SAVE_MODEL_INTERVAL == 0:
                self.save(counter)

    def test(self):
        # load the trained model
        could_load, checkpoint_counter = self.load()
        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        label = 1
        for test_info in self.test_im_list:
            test_im_rbox_info = test_info.split(' ')
            test_im_path = os.path.join(TEST_DATA_PATH, test_im_rbox_info[0])
            test_rbox_gt_path = os.path.join(TEST_DATA_PATH, test_im_rbox_info[0]+'.rbox')
            test_result_path = TXT_DIR + '/' + os.path.basename(SAVE_PATH)
            if not os.path.exists(test_result_path):
            		os.makedirs(test_result_path)
            test_rbox_output_path = os.path.join(test_result_path, os.path.basename(test_rbox_gt_path) + '.score')
            test_im = scipy.misc.imread(test_im_path)
            if 'L2' in test_im_path:
                not_zero   = np.where(test_im != 0)
                is_zero    = np.where(test_im == 0)
                mean_value = np.sum(test_im[not_zero])/len(not_zero[0])
                for temp_idx in range(len(is_zero[0])):
                    test_im[is_zero[0][temp_idx], is_zero[1][temp_idx]] = mean_value   
            temp = np.zeros((test_im.shape[0], test_im.shape[1], IM_CDIM))
            for chid in range(IM_CDIM):
                temp[:,:,chid] = test_im
            test_im = temp
            [height, width, _] = test_im.shape
            print('Start detection'+test_im_path)
            count = 0
            islast = 0
            inputdata = np.zeros((TEST_BATCH_SIZE, IM_HEIGHT, IM_WIDTH, IM_CDIM))
            inputdata = inputdata.astype('float32')
            inputloc = np.zeros((TEST_BATCH_SIZE, IM_CDIM))
            rboxlist = []
            scorelist = []
            #start = time.time()
            for i in range(len(TEST_RESOLUTION_OUT)):                                
                xBegin, yBegin = 0, 0
                width_i = int(round(width * TEST_RESOLUTION_IN / TEST_RESOLUTION_OUT[i]))
                height_i = int(round(height * TEST_RESOLUTION_IN / TEST_RESOLUTION_OUT[i]))
                image_i = scipy.misc.imresize(test_im, [height_i, width_i, IM_CDIM])
                while 1:
                    if islast == 0:                        
                        width_S = IM_WIDTH * TEST_RESOLUTION_OUT[i] / TEST_RESOLUTION_IN #int(round(IM_WIDTH * TEST_RESOLUTION_OUT[i] / TEST_RESOLUTION_IN))
                        height_S = IM_HEIGHT * TEST_RESOLUTION_OUT[i] / TEST_RESOLUTION_IN #int(round(IM_HEIGHT * TEST_RESOLUTION_OUT[i] / TEST_RESOLUTION_IN))
                        xEnd = xBegin + width_S
                        yEnd = yBegin + height_S
                        xEnd = min(xEnd, width)
                        yEnd = min(yEnd, height)
                        xBeginHat = int(round(xBegin * TEST_RESOLUTION_IN / TEST_RESOLUTION_OUT[i]))
                        yBeginHat = int(round(yBegin * TEST_RESOLUTION_IN / TEST_RESOLUTION_OUT[i]))
                        xEndHat = int(round(xEnd * TEST_RESOLUTION_IN / TEST_RESOLUTION_OUT[i]))
                        yEndHat = int(round(yEnd * TEST_RESOLUTION_IN / TEST_RESOLUTION_OUT[i]))
                        subimage = np.zeros((IM_HEIGHT, IM_WIDTH, IM_CDIM))
                        subimage[0:yEndHat-yBeginHat, 0:xEndHat-xBeginHat, :] = image_i[yBeginHat:yEndHat, xBeginHat:xEndHat, :]
                        inputdata[count] = subimage.astype('float32')
                        #print xBegin,yBegin
                        inputloc[count] = [xBegin,yBegin,TEST_RESOLUTION_OUT[i]/TEST_RESOLUTION_IN]
                        count = count + 1
                    if count == TEST_BATCH_SIZE or islast == 1:
                        loc_preds, conf_preds = self.sess.run([self.loc, self.conf_softmax], feed_dict={self.input_im:inputdata})
                        for j in range(TEST_BATCH_SIZE):
                            conf_preds_j = conf_preds[j*self.total_prior_num:(j+1)*self.total_prior_num, 1]
                            loc_preds_j  = loc_preds[j*self.total_prior_num:(j+1)*self.total_prior_num, :]
                            index = np.where(conf_preds_j > TEST_SCORE_THRESHOLD)[0]
                            conf_preds_j  = conf_preds_j[index]
                            loc_preds_j   = loc_preds_j[index]
                            loc_preds_j   = loc_preds_j.reshape(loc_preds_j.shape[0]*self.para_num)
                            prior_boxes_j = self.prior_box[index].reshape(len(index) * self.para_num)
                            inputloc_j = inputloc[j]
                            if len(loc_preds_j) > 0:
                                rbox, score = DecodeNMS(loc_preds_j, prior_boxes_j, conf_preds_j, inputloc_j, index, TEST_NMS_THRESHOLD, IM_HEIGHT, IM_WIDTH)
                                rboxlist.extend(rbox)
                                scorelist.extend(score)
                        count = 0
                    if islast == 1:
                        break
                    xBegin = xBegin + int(round(TEST_WIDTH_STEP * width_S))
                    if  xEnd >= width: #xBegin
                        if yEnd >= height:
                            islast = 0
                            break
                        xBegin = 0
                        yBegin = yBegin + int(round(TEST_HEIGHT_STEP * height_S))
                        if yBegin >= height:
                            if i == len(TEST_RESOLUTION_OUT) - 1:
                                islast = 1
                            else:
                                break
            NMSOutput(rboxlist, scorelist, TEST_NMS_THRESHOLD, label, test_rbox_output_path)

    def save(self, step):
        model_name = "DrBoxNet.model"
        self.saver.save(self.sess, os.path.join(self.model_save_path, model_name), global_step=step)

    def load(self):
        import re
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(self.model_save_path)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.model_save_path, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(os.path.join(self.model_save_path, ckpt_name)))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            print(" [*] Load the pretrained network")
            self.load_prenet()
            return False, 0                                  
    
    def load_prenet(self):
        data_list = np.load(PRETRAINED_NET_PATH).item()
        data_keys = data_list.keys()
        var_list = self.detector.vars
        for var in var_list:
            for key in data_keys:
                if key in var.name:
                    if 'weights' in var.name:                        
                        self.sess.run(tf.assign(var, data_list[key][0]))
                        print("pretrained net {} weights -> scene net {}".format(key, var.name))
                        break
                    else: # for biases
                        self.sess.run(tf.assign(var, data_list[key][1]))
                        print("pretrained net {} biases  -> scene net {}".format(key, var.name))
                        break            
                
if __name__ == '__main__':
    net = DrBoxNet()
    if FLAGS.train:
        net.train()
    else:
        net.test()
                        
