from model.Attention_module import NONLocal1D, CALayer1D
from model.Resnet1d import *
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, use_attention = False, use_non_local = False, input_channel = 3, num_class = 2):
        super(Model, self).__init__()
        self.input_channel = input_channel
        self.output_channel = num_class
        self.inter_channel = 64
        self.eda_vector_dimension = 896
        self.use_softmax = True
        self.use_attention = use_attention
        self.use_non_local = use_non_local
        self.use_res_sub = True if use_attention or use_non_local else False

        if self.use_res_sub:
            self.origin_block = nn.Sequential(
                nn.Conv1d(self.input_channel, self.inter_channel, kernel_size=7, stride = 2, padding = 3, bias=True),
                nn.BatchNorm1d(self.inter_channel)
            )
            self.Res_sub = ResNet1D(self.inter_channel, self.use_res_sub)

            if use_non_local:
                self.Non_local = NONLocal1D(in_feat=self.inter_channel, inter_feat=self.inter_channel // 2)
            if use_attention:
                self.Attention = CALayer1D(channel=self.inter_channel)

        else:
            self.Res_sub = ResNet1D(self.input_channel, self.use_res_sub)
        self.regression = self.Regression()

    def forward(self, eda):
        if self.use_res_sub:
            eda_out = self.origin_block(eda)
            if self.use_non_local:
                eda_out = self.Non_local(eda_out)
            if self.use_attention:
                eda_out = self.Attention(eda_out)
            eda_vector = self.Res_sub(eda_out)
        else:
            eda_vector = self.Res_sub(eda)
        eda_vector = eda_vector.view(eda_vector.size(0),-1)
        out = self.regression(eda_vector)
        return out

    def Regression(self):
        return nn.Sequential(
            nn.Linear(self.eda_vector_dimension, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.output_channel),
            nn.ReLU(inplace=True),
        )
