import numpy as np
import csv
import random
import os
from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image
import json
import sys

data_path = '/home/guanghaoyin/Desktop/Dataset/DEAPdataset/'
start_time = 0  # ignoring start seconds has been done
eda_fps = 50
Net_size = 50
fold_cross_num = 10

class data_loader_1D(object):
    def __init__(self, type, cvx, fold_cross_num = 10):

        self.type = type
        self.cvx = cvx
        self.fold_cross_num = fold_cross_num
        self.fold_num = 0
        self.path = data_path
        self.eda_fps = eda_fps

        self.trainId = []
        self.testID = []
        self.trainLabel = []
        self.testLabel = []

        self.label_all, self.separate_dataID = self.dataFile_separate(type)

        self.transform = transforms.Compose([transforms.Resize(Net_size),])

    def dataFile_separate(self, type):
        separate_id = []
        label = []
        label0_list = []
        label1_list = []
        # read label_all.csv

        label_all = []

        label_reader = csv.reader(open(data_path + 'label_all_annotation.csv', 'r'))
        next(label_reader)
        for line in label_reader:
            subject_id = int(line[0])
            video_id = int(line[1])
            if type == 'Arousal':
                label = float(line[7])
            elif type == 'Valence':
                label = float(line[6])
            if label == 1:
                label1_list.append([subject_id, video_id, label])
            elif label == 0:
                label0_list.append([subject_id, video_id, label])

        random.shuffle(label0_list)
        random.shuffle(label1_list)
        len_label_list0 = len(label0_list)
        len_label1_list1 = len(label1_list)

        fold0_num = int(len(label0_list) / self.fold_cross_num)
        fold1_num = int(len(label1_list) / self.fold_cross_num)
        fold_num = fold0_num + fold1_num

        self.fold0_num = fold0_num
        self.fold1_num = fold1_num
        self.fold_num = fold_num

        for i in range(self.fold_cross_num):
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
        video_id = load_data.label_all[idx][1]
        data_name = str(subject_id) + '_' + str(video_id) + '.txt'
        data_all = []
        for cvx in ['tonic', 'phasic', 'origin']:
            data_path = os.path.join(load_data.path, 'img', cvx, data_name)
            data_file = open(data_path, 'r')
            data = data_file.read()
            data = np.array(json.loads(data))[384:]
            data2d = np.array(data[:7569]).reshape(87,87)
            # import pylab as pl
            # t2 = pl.arange(1., len(data) + 1.)
            # pl.plot(t2, data)
            # pl.savefig('orign.png')
            # pl.show()
            data = np.array(transform(Image.fromarray(data2d))).flatten()
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
