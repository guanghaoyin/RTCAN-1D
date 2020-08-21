import numpy as np
import csv
import random
import os
from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image
import json
import sys

data_path = '/home/guanghaoyin/Desktop/Dataset/PMEmo/pmemo_dataset/PMEmo/'
start_time = 0  # ignore start seconds has been done
eda_fps = 50
Net_size = 50
fold_cross_num = 10
prerpocess = False

# process = True

class data_loader_1D(object):
    def __init__(self, type, cvx, fold_cross_num = 10):

        self.type = type
        self.cvx = cvx
        self.fold_cross_num = fold_cross_num
        self.fold_num = 0
        self.path = data_path
        self.start_time = start_time
        self.eda_fps = eda_fps

        self.trainId = []
        self.testID = []
        self.trainLabel = []
        self.testLabel = []

        self.label_all, self.separate_dataID = self.dataFile_separate(type)
        if prerpocess:
            self.save_img()
        self.transform = transforms.Compose([transforms.Resize(Net_size),])

    def save_img(self):
        transform = transforms.Compose([transforms.Resize(50), ])
        for i in tqdm(range(len(self.label_all)), ascii=True, desc=' data_processing'):
            label = self.label_all[i]
            subject_id = label[0]
            music_id = label[1]
            data_name_v = str(music_id) + '_' + str(subject_id) + '_V.txt'
            data_name_a = str(music_id) + '_' + str(subject_id) + '_A.txt'
            for cvx in ['origin', 'tonic', 'phasic']:
                data_path_V = os.path.join(data_path, 'Img','Valence' , cvx, data_name_v)
                data_path_v = os.path.join(data_path, 'txt','Valence' , cvx, data_name_v)
                data_path_A = os.path.join(data_path, 'Img','Arousal' , cvx, data_name_a)
                data_path_a = os.path.join(data_path, 'txt','Arousal' , cvx, data_name_a)

                data_file_V = open(data_path_V, 'r')
                data = data_file_V.read()
                data_V = np.array(json.loads(data))
                data_v = np.array(transform(Image.fromarray(data_V))).flatten()
                data_file_v = open(data_path_v, 'w')
                data_file_v.write(str(data_v.tolist()))
                data_file_v.close()
                data_file_V.close()

                data_file_A = open(data_path_A, 'r')
                data = data_file_A.read()
                data_A = np.array(json.loads(data))
                data_a = np.array(transform(Image.fromarray(data_A))).flatten()
                data_file_a = open(data_path_a, 'w')
                data_file_a.write(str(data_a.tolist()))
                data_file_a.close()
                data_file_a.close()


    def dataFile_separate(self, type):
        separate_id = []
        label = []
        label0_list = []
        label1_list = []
        label_list = []
        # read label_all.csv

        label_all = []

        label_reader = csv.reader(open(data_path + 'value_all.csv', 'r'))
        next(label_reader)
        for line in label_reader:
            if line[0] == 'subject_id':
                break
            subject_id = int(line[0])
            music_id = int(line[1])
            if type == 'Arousal':
                label = int(line[2])
            elif type == 'Valence':
                label = int(line[3])
            if label == 1:
                label1_list.append([subject_id, music_id, label])
            elif label == 0:
                label0_list.append([subject_id, music_id, label])
            label_list.append([subject_id, music_id, label])

        random.shuffle(label0_list)
        random.shuffle(label1_list)
        len_label_list0 = len(label0_list)
        len_label1_list1 = len(label1_list)

        fold0_num = int(len(label0_list) / self.fold_cross_num)
        fold1_num = int(len(label1_list) / self.fold_cross_num)
        fold_num = fold0_num + fold1_num
        for i in range(self.fold_cross_num):
            #EDA
            label_all.extend(label0_list[i * fold0_num:(i + 1) * fold0_num] + label1_list[i * fold1_num:(i + 1) * fold1_num])
            separate_id.append([i for i in range(i * fold_num, (i + 1) * fold_num)])
        return label_all, separate_id

    def fold_arrange_id(self, fold_num):
        trainLabel = []
        testLabel = []
        self.trainId = []
        self.testID = []
        for i in range(fold_cross_num):
            if i == fold_num:
                self.testID = self.separate_dataID[i]
                testLabel.extend([self.label_all[j][2] for j in self.separate_dataID[i]])
            else:
                self.trainId.extend(self.separate_dataID[i])
                trainLabel.extend([self.label_all[j][2] for j in self.separate_dataID[i]])
        self.trainLabel = trainLabel
        self.testLabel = testLabel


def get_batch_data(dataId, load_data):  # (list,class data_loader)
    batch_data = np.array([])
    start = 0
    transform = transforms.Compose([transforms.Resize(Net_size), ])
    for i in tqdm(range(len(dataId)), ascii=True, desc=' load batch data'):
        idx = dataId[i]
        subject_id = load_data.label_all[idx][0]
        music_id = load_data.label_all[idx][1]
        data_name = str(music_id) + '_' + str(subject_id) + '_' + str(load_data.type[0])

        data_all = []
        for cvx in ['origin', 'tonic', 'phasic']:
            data_path = os.path.join(load_data.path, 'txt', load_data.type, cvx, data_name + '.txt')
            data_file = open(data_path, 'r')
            data = data_file.read()
            data = np.array(json.loads(data))
            # import pylab as pl
            # t2 = pl.arange(1., len(data) + 1.)
            # pl.plot(t2, data)
            # pl.savefig('orign.png')
            # pl.show()
            # data = np.array(transform(Image.fromarray(data))).flatten()
            # t2 = pl.arange(1., len(data) + 1.)
            # pl.plot(t2, data)
            # pl.savefig('trans.png')
            # pl.show()
            data_all.append(data)

        data = np.array([[data_all[0], data_all[1], data_all[2]]])
        # data = np.array([np.concatenate((data_all[0], data_all[1], data_all[2]), axis=0)])

        if start == 0:
            start = 1
            batch_data = data.copy()
        else:
            batch_data = np.vstack([batch_data, data])
    return batch_data


def get_batch_vector(dataId, load_data):
    batch_vector = np.array([])
    start = 0
    for i in tqdm(range(len(dataId)), ascii=True, desc=' load batch vector'):
        idx = dataId[i]
        music_id = load_data.label_all[idx][1]
        vector_path = os.path.join(load_data.path, 'static_feature', str(music_id) + '.txt')
        vector_file = open(vector_path, 'r')
        vector = vector_file.read()
        vector = json.loads(vector)
        vector = np.array(vector)
        if start == 0:
            start = 1
            batch_vector = vector.copy()
        else:
            batch_vector = np.vstack([batch_vector, vector])
    return batch_vector
