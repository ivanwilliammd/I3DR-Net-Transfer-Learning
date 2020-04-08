#!/usr/bin/env python
# Official implementation code for "Lung Nodule Detection and Classification from Thorax CT-Scan Using RetinaNet with Transfer Learning" and "Lung Nodule Texture Detection and Classification Using 3D CNN."
# Adapted from of [medicaldetectiontoolkit](https://github.com/pfjaeger/medicaldetectiontoolkit) and [kinetics_i3d_pytorch](https://github.com/hassony2/kinetics_i3d_pytorch)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import math
import os

import numpy as np
import torch
from torch.nn import ReplicationPad3d
import torch.nn.functional as F



def get_padding_shape(filter_shape, stride):
    def _pad_top_bottom(filter_dim, stride_val):
        pad_along = max(filter_dim - stride_val, 0)
        pad_top = pad_along // 2
        pad_bottom = pad_along - pad_top
        return pad_top, pad_bottom

    padding_shape = []
    for filter_dim, stride_val in zip(filter_shape, stride):
        pad_top, pad_bottom = _pad_top_bottom(filter_dim, stride_val)
        padding_shape.append(pad_top)
        padding_shape.append(pad_bottom)
    depth_top = padding_shape.pop(0)
    depth_bottom = padding_shape.pop(0)
    padding_shape.append(depth_top)
    padding_shape.append(depth_bottom)

    return tuple(padding_shape)


def simplify_padding(padding_shapes):
    all_same = True
    padding_init = padding_shapes[0]
    for pad in padding_shapes[1:]:
        if pad != padding_init:
            all_same = False
    return all_same, padding_init


class Unit3Dpy(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(1, 1, 1),
                 stride=(1, 1, 1),
                 activation='relu',
                 padding='SAME',
                 use_bias=False,
                 use_bn=True):
        super(Unit3Dpy, self).__init__()

        self.padding = padding
        self.activation = activation
        self.use_bn = use_bn
        if padding == 'SAME':
            padding_shape = get_padding_shape(kernel_size, stride)
            simplify_pad, pad_size = simplify_padding(padding_shape)
            self.simplify_pad = simplify_pad
        elif padding == 'VALID':
            padding_shape = 0
        else:
            raise ValueError(
                'padding should be in [VALID|SAME] but got {}'.format(padding))

        if padding == 'SAME':
            if not simplify_pad:
                self.pad = torch.nn.ConstantPad3d(padding_shape, 0)
                self.conv3d = torch.nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    bias=use_bias)
            else:
                self.conv3d = torch.nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=pad_size,
                    bias=use_bias)
        elif padding == 'VALID':
            self.conv3d = torch.nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding_shape,
                stride=stride,
                bias=use_bias)
        else:
            raise ValueError(
                'padding should be in [VALID|SAME] but got {}'.format(padding))

        if self.use_bn:
            self.batch3d = torch.nn.BatchNorm3d(out_channels)

        if activation == 'relu':
            self.activation = F.relu

    def forward(self, inp):
        if self.padding == 'SAME' and self.simplify_pad is False:
            inp = self.pad(inp)
        out = self.conv3d(inp)
        if self.use_bn:
            out = self.batch3d(out)
        if self.activation is not None:
            out = F.relu(out)
        return out


class MaxPool3dTFPadding(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding='SAME'):
        super(MaxPool3dTFPadding, self).__init__()
        if padding == 'SAME':
            padding_shape = get_padding_shape(kernel_size, stride)
            self.padding_shape = padding_shape
            self.pad = torch.nn.ConstantPad3d(padding_shape, 0)
        self.pool = torch.nn.MaxPool3d(kernel_size, stride, ceil_mode=True)

    def forward(self, inp):
        inp = self.pad(inp)
        out = self.pool(inp)
        return out


class Mixed(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Mixed, self).__init__()
        # Branch 0
        self.branch_0 = Unit3Dpy(
            in_channels, out_channels[0], kernel_size=(1, 1, 1))

        # Branch 1
        branch_1_conv1 = Unit3Dpy(
            in_channels, out_channels[1], kernel_size=(1, 1, 1))
        branch_1_conv2 = Unit3Dpy(
            out_channels[1], out_channels[2], kernel_size=(3, 3, 3))
        self.branch_1 = torch.nn.Sequential(branch_1_conv1, branch_1_conv2)

        # Branch 2
        branch_2_conv1 = Unit3Dpy(
            in_channels, out_channels[3], kernel_size=(1, 1, 1))
        branch_2_conv2 = Unit3Dpy(
            out_channels[3], out_channels[4], kernel_size=(3, 3, 3))
        self.branch_2 = torch.nn.Sequential(branch_2_conv1, branch_2_conv2)

        # Branch3
        branch_3_pool = MaxPool3dTFPadding(
            kernel_size=(3, 3, 3), stride=(1, 1, 1), padding='SAME')
        branch_3_conv2 = Unit3Dpy(
            in_channels, out_channels[5], kernel_size=(1, 1, 1))
        self.branch_3 = torch.nn.Sequential(branch_3_pool, branch_3_conv2)

    def forward(self, inp):
        out_0 = self.branch_0(inp)
        out_1 = self.branch_1(inp)
        out_2 = self.branch_2(inp)
        out_3 = self.branch_3(inp)
        out = torch.cat((out_0, out_1, out_2, out_3), 1)
        return out


class I3D(torch.nn.Module):
    def __init__(self,
                 num_classes,
                 modality='rgb',
                 dropout_prob=0,
                 name='inception'):
        super(I3D, self).__init__()

        self.name = name
        self.num_classes = num_classes
        if modality == 'rgb':
            in_channels = 3
        elif modality == 'flow':
            in_channels = 2
        else:
            raise ValueError(
                '{} not among known modalities [rgb|flow]'.format(modality))
        self.modality = modality

        conv3d_1a_7x7 = Unit3Dpy(
            out_channels=64,
            in_channels=in_channels,
            kernel_size=(7, 7, 7),
            stride=(2, 2, 2),
            padding='SAME')
        # 1st conv-pool
        self.conv3d_1a_7x7 = conv3d_1a_7x7
        self.maxPool3d_2a_3x3 = MaxPool3dTFPadding(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding='SAME')
        # conv conv
        conv3d_2b_1x1 = Unit3Dpy(
            out_channels=64,
            in_channels=64,
            kernel_size=(1, 1, 1),
            padding='SAME')
        self.conv3d_2b_1x1 = conv3d_2b_1x1
        conv3d_2c_3x3 = Unit3Dpy(
            out_channels=192,
            in_channels=64,
            kernel_size=(3, 3, 3),
            padding='SAME')
        self.conv3d_2c_3x3 = conv3d_2c_3x3
        self.maxPool3d_3a_3x3 = MaxPool3dTFPadding(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding='SAME')


        # Mixed_3b
        self.mixed_3b = Mixed(192, [64, 96, 128, 16, 32, 32])
        self.mixed_3c = Mixed(256, [128, 128, 192, 32, 96, 64])

        self.maxPool3d_4a_3x3 = MaxPool3dTFPadding(
            kernel_size=(3, 3, 3), stride=(2, 2, 2), padding='SAME')

        # Mixed 4
        self.mixed_4b = Mixed(480, [192, 96, 208, 16, 48, 64])
        self.mixed_4c = Mixed(512, [160, 112, 224, 24, 64, 64])
        self.mixed_4d = Mixed(512, [128, 128, 256, 24, 64, 64])
        self.mixed_4e = Mixed(512, [112, 144, 288, 32, 64, 64])
        self.mixed_4f = Mixed(528, [256, 160, 320, 32, 128, 128])

        self.maxPool3d_5a_2x2 = MaxPool3dTFPadding(
            kernel_size=(2, 2, 2), stride=(2, 2, 2), padding='SAME')

        # Mixed 5
        self.mixed_5b = Mixed(832, [256, 160, 320, 32, 128, 128])
        self.mixed_5c = Mixed(832, [384, 192, 384, 48, 128, 128])

        # self.avg_pool = torch.nn.AvgPool3d((2, 7, 7), (1, 1, 1))
        # self.dropout = torch.nn.Dropout(dropout_prob)
        # self.conv3d_0c_1x1 = Unit3Dpy(
        #     in_channels=1024,
        #     out_channels=self.num_classes,
        #     kernel_size=(1, 1, 1),
        #     activation=None,
        #     use_bias=True,
        #     use_bn=False)
        # self.softmax = torch.nn.Softmax(1)

        ### Return channel to original size (36)
        conv3d_return_2c = Unit3Dpy(
            in_channels=192,
            out_channels=36,
            kernel_size=(1, 1, 1),
            padding='SAME')
        self.conv3d_return_2c = conv3d_return_2c

        conv3d_return_3c = Unit3Dpy(
            in_channels=480,
            out_channels=36,
            kernel_size=(1, 1, 1),
            padding='SAME')
        self.conv3d_return_3c = conv3d_return_3c

        conv3d_return_4f = Unit3Dpy(
            in_channels=832,
            out_channels=36,
            kernel_size=(1, 1, 1),
            padding='SAME')
        self.conv3d_return_4f = conv3d_return_4f

        conv3d_return_5c = Unit3Dpy(
            in_channels=1024,
            out_channels=36,
            kernel_size=(1, 1, 1),
            padding='SAME')
        self.conv3d_return_5c = conv3d_return_5c


        # ###### pre-out ###########
        # conv3d_preout_P2 = Unit3Dpy(
        #     in_channels=36,
        #     out_channels=36,
        #     kernel_size=(3, 3, 3),
        #     padding='SAME')
        # self.conv3d_preout_P2 = conv3d_preout_P2

        # conv3d_preout_P3 = Unit3Dpy(
        #     in_channels=36,
        #     out_channels=36,
        #     kernel_size=(3, 3, 3),
        #     padding='SAME')
        # self.conv3d_preout_P3 = conv3d_preout_P3

        # conv3d_preout_P4 = Unit3Dpy(
        #     in_channels=36,
        #     out_channels=36,
        #     kernel_size=(3, 3, 3),
        #     padding='SAME')
        # self.conv3d_preout_P4 = conv3d_preout_P4

        # conv3d_preout_P5 = Unit3Dpy(
        #     in_channels=36,
        #     out_channels=36,
        #     kernel_size=(3, 3, 3),
        #     padding='SAME')
        # self.conv3d_preout_P5 = conv3d_preout_P5
        

    def forward(self, inp):
        # Preprocessing
        
        ### Permute image from b,c,y,x,z --> b,c,z,y,x
        inp_permute=inp.permute(0,1,4,2,3)
        out_1a = self.conv3d_1a_7x7(inp_permute)
        out_2a = self.maxPool3d_2a_3x3(out_1a)
        out_2b = self.conv3d_2b_1x1(out_2a)
        out_2c = self.conv3d_2c_3x3(out_2b)
        
        out_3a = self.maxPool3d_3a_3x3(out_2c)
        out_3b = self.mixed_3b(out_3a)
        out_3c = self.mixed_3c(out_3b)
        
        out_4a = self.maxPool3d_4a_3x3(out_3c)
        out_4b = self.mixed_4b(out_4a)
        out_4c = self.mixed_4c(out_4b)
        out_4d = self.mixed_4d(out_4c)
        out_4e = self.mixed_4e(out_4d)
        out_4f = self.mixed_4f(out_4e)
        
        out_5a = self.maxPool3d_5a_2x2(out_4f)
        out_5b = self.mixed_5b(out_5a)
        out_5c = self.mixed_5c(out_5b)
        
        # ############# No FPN ####################
        # P2_raw = self.conv3d_return_2c(out_2c)
        # P2 = F.interpolate(P2_raw, scale_factor=(2,1,1), mode='trilinear')

        # P3 = self.conv3d_return_3c(out_3c)
        
        # P4 = self.conv3d_return_4f(out_4f)
        
        # P5 = self.conv3d_return_5c(out_5c)
        
        # P2_total=P2
        # P3_total=P3
        # P4_total=P4
        # P5_total=P5

        # ### Reverse Permute image from b,c,z,y,x --> b,c,y,x,z 
        # P2_final = P2_total.permute(0,1,3,4,2)
        # P3_final = P3_total.permute(0,1,3,4,2)
        # P4_final = P4_total.permute(0,1,3,4,2)
        # P5_final = P5_total.permute(0,1,3,4,2)

        # out_list =[P2_final, P3_final, P4_final, P5_final]


        ############# Lossy FPN ####################
        P2_raw = self.conv3d_return_2c(out_2c)
        P2 = F.interpolate(P2_raw, scale_factor=(2,1,1), mode='trilinear')

        P3 = self.conv3d_return_3c(out_3c)
        P3_upsampled = F.interpolate(P3, scale_factor=(2,2,2), mode='trilinear')

        P4 = self.conv3d_return_4f(out_4f)
        P4_upsampled = F.interpolate(P4, scale_factor=(2,2,2), mode='trilinear')

        P5 = self.conv3d_return_5c(out_5c)
        P5_upsampled = F.interpolate(P5, scale_factor=(2,2,2), mode='trilinear')

        P2_total=P2+P3_upsampled
        P3_total=P3+P4_upsampled
        P4_total=P4+P5_upsampled
        P5_total=P5

        ### Reverse Permute image from b,c,z,y,x --> b,c,y,x,z 
        P2_final = P2_total.permute(0,1,3,4,2)
        P3_final = P3_total.permute(0,1,3,4,2)
        P4_final = P4_total.permute(0,1,3,4,2)
        P5_final = P5_total.permute(0,1,3,4,2)

        out_list =[P2_final, P3_final, P4_final, P5_final]


        ############ Dense FPN #####################
        # C2_raw = self.conv3d_return_2c(out_2c)
        # C2 = F.interpolate(C2_raw, scale_factor=(2,1,1), mode='trilinear')
        # C3 = self.conv3d_return_3c(out_3c)
        # C4 = self.conv3d_return_4f(out_4f)
        # C5 = self.conv3d_return_5c(out_5c)
        # P5_total=C5
        # P4_total=C4+F.interpolate(P5_total, scale_factor=(2,2,2), mode='trilinear')
        # P3_total=C3+F.interpolate(P4_total, scale_factor=(2,2,2), mode='trilinear')
        # P2_total=C2+F.interpolate(P3_total, scale_factor=(2,2,2), mode='trilinear')
                
        # P5_preout = self.conv3d_preout_P5(P5_total)
        # P4_preout = self.conv3d_preout_P4(P4_total)
        # P3_preout = self.conv3d_preout_P3(P3_total)
        # P2_preout = self.conv3d_preout_P2(P2_total)

        ### Reverse Permute image from b,c,z,y,x --> b,c,y,x,z 
        
        # P5_final = P5_preout.permute(0,1,3,4,2)
        # P4_final = P4_preout.permute(0,1,3,4,2)
        # P3_final = P3_preout.permute(0,1,3,4,2)
        # P2_final = P2_preout.permute(0,1,3,4,2)

        # out_list =[P2_final, P3_final, P4_final, P5_final]

        return out_list

    def load_tf_weights(self, sess):
        state_dict = {}
        if self.modality == 'rgb':
            prefix = 'RGB/inception_i3d'
        elif self.modality == 'flow':
            prefix = 'Flow/inception_i3d'
        load_conv3d(state_dict, 'conv3d_1a_7x7', sess,
                    os.path.join(prefix, 'Conv3d_1a_7x7'))
        load_conv3d(state_dict, 'conv3d_2b_1x1', sess,
                    os.path.join(prefix, 'Conv3d_2b_1x1'))
        load_conv3d(state_dict, 'conv3d_2c_3x3', sess,
                    os.path.join(prefix, 'Conv3d_2c_3x3'))

        load_mixed(state_dict, 'mixed_3b', sess,
                   os.path.join(prefix, 'Mixed_3b'))
        load_mixed(state_dict, 'mixed_3c', sess,
                   os.path.join(prefix, 'Mixed_3c'))
        load_mixed(state_dict, 'mixed_4b', sess,
                   os.path.join(prefix, 'Mixed_4b'))
        load_mixed(state_dict, 'mixed_4c', sess,
                   os.path.join(prefix, 'Mixed_4c'))
        load_mixed(state_dict, 'mixed_4d', sess,
                   os.path.join(prefix, 'Mixed_4d'))
        load_mixed(state_dict, 'mixed_4e', sess,
                   os.path.join(prefix, 'Mixed_4e'))
        # Here goest to 0.1 max error with tf
        load_mixed(state_dict, 'mixed_4f', sess,
                   os.path.join(prefix, 'Mixed_4f'))

        load_mixed(
            state_dict,
            'mixed_5b',
            sess,
            os.path.join(prefix, 'Mixed_5b'),
            fix_typo=True)
        load_mixed(state_dict, 'mixed_5c', sess,
                   os.path.join(prefix, 'Mixed_5c'))
        # load_conv3d(
        #     state_dict,
        #     'conv3d_0c_1x1',
        #     sess,
        #     os.path.join(prefix, 'Logits', 'Conv3d_0c_1x1'),
        #     bias=True,
        #     bn=False)
        self.load_state_dict(state_dict)


def get_conv_params(sess, name, bias=False):
    # Get conv weights
    conv_weights_tensor = sess.graph.get_tensor_by_name(
        os.path.join(name, 'w:0'))
    if bias:
        conv_bias_tensor = sess.graph.get_tensor_by_name(
            os.path.join(name, 'b:0'))
        conv_bias = sess.run(conv_bias_tensor)
    conv_weights = sess.run(conv_weights_tensor)
    conv_shape = conv_weights.shape

    kernel_shape = conv_shape[0:3]
    in_channels = conv_shape[3]
    out_channels = conv_shape[4]

    conv_op = sess.graph.get_operation_by_name(
        os.path.join(name, 'convolution'))
    padding_name = conv_op.get_attr('padding')
    padding = _get_padding(padding_name, kernel_shape)
    all_strides = conv_op.get_attr('strides')
    strides = all_strides[1:4]
    conv_params = [
        conv_weights, kernel_shape, in_channels, out_channels, strides, padding
    ]
    if bias:
        conv_params.append(conv_bias)
    return conv_params


def get_bn_params(sess, name):
    moving_mean_tensor = sess.graph.get_tensor_by_name(
        os.path.join(name, 'moving_mean:0'))
    moving_var_tensor = sess.graph.get_tensor_by_name(
        os.path.join(name, 'moving_variance:0'))
    beta_tensor = sess.graph.get_tensor_by_name(os.path.join(name, 'beta:0'))
    moving_mean = sess.run(moving_mean_tensor)
    moving_var = sess.run(moving_var_tensor)
    beta = sess.run(beta_tensor)
    return moving_mean, moving_var, beta


def _get_padding(padding_name, conv_shape):
    padding_name = padding_name.decode("utf-8")
    if padding_name == "VALID":
        return [0, 0]
    elif padding_name == "SAME":
        # return [math.ceil(int(conv_shape[0])/2), math.ceil(int(conv_shape[1])/2)]
        return [
            math.floor(int(conv_shape[0]) / 2),
            math.floor(int(conv_shape[1]) / 2),
            math.floor(int(conv_shape[2]) / 2)
        ]
    else:
        raise ValueError('Invalid padding name ' + padding_name)


def load_conv3d(state_dict, name_pt, sess, name_tf, bias=False, bn=True):
    # Transfer convolution params
    conv_name_tf = os.path.join(name_tf, 'conv_3d')
    conv_params = get_conv_params(sess, conv_name_tf, bias=bias)
    if bias:
        conv_weights, kernel_shape, in_channels, out_channels, strides, padding, conv_bias = conv_params
    else:
        conv_weights, kernel_shape, in_channels, out_channels, strides, padding = conv_params

    conv_weights_rs = np.transpose(
        conv_weights, (4, 3, 0, 1,
                       2))  # to pt format (out_c, in_c, depth, height, width)
    state_dict[name_pt + '.conv3d.weight'] = torch.from_numpy(conv_weights_rs)
    if bias:
        state_dict[name_pt + '.conv3d.bias'] = torch.from_numpy(conv_bias)

    # Transfer batch norm params
    if bn:
        conv_tf_name = os.path.join(name_tf, 'batch_norm')
        moving_mean, moving_var, beta = get_bn_params(sess, conv_tf_name)

        out_planes = conv_weights_rs.shape[0]
        state_dict[name_pt + '.batch3d.weight'] = torch.ones(out_planes)
        state_dict[name_pt +
                   '.batch3d.bias'] = torch.from_numpy(beta.squeeze())
        state_dict[name_pt
                   + '.batch3d.running_mean'] = torch.from_numpy(moving_mean.squeeze())
        state_dict[name_pt
                   + '.batch3d.running_var'] = torch.from_numpy(moving_var.squeeze())


def load_mixed(state_dict, name_pt, sess, name_tf, fix_typo=False):
    # Branch 0
    load_conv3d(state_dict, name_pt + '.branch_0', sess,
                os.path.join(name_tf, 'Branch_0/Conv3d_0a_1x1'))

    # Branch .1
    load_conv3d(state_dict, name_pt + '.branch_1.0', sess,
                os.path.join(name_tf, 'Branch_1/Conv3d_0a_1x1'))
    load_conv3d(state_dict, name_pt + '.branch_1.1', sess,
                os.path.join(name_tf, 'Branch_1/Conv3d_0b_3x3'))

    # Branch 2
    load_conv3d(state_dict, name_pt + '.branch_2.0', sess,
                os.path.join(name_tf, 'Branch_2/Conv3d_0a_1x1'))
    if fix_typo:
        load_conv3d(state_dict, name_pt + '.branch_2.1', sess,
                    os.path.join(name_tf, 'Branch_2/Conv3d_0a_3x3'))
    else:
        load_conv3d(state_dict, name_pt + '.branch_2.1', sess,
                    os.path.join(name_tf, 'Branch_2/Conv3d_0b_3x3'))

    # Branch 3
    load_conv3d(state_dict, name_pt + '.branch_3.1', sess,
                os.path.join(name_tf, 'Branch_3/Conv3d_0b_1x1'))



# import torch.nn as nn
# import F as F
# import torch


# class FPN(nn.Module):
#     """
#     Feature Pyramid Network from https://arxiv.org/pdf/1612.03144.pdf with options for modifications.
#     by default is constructed with Pyramid levels P2, P3, P4, P5.
#     """
#     def __init__(self, cf, conv, operate_stride1=False):
#         """
#         from configs:
#         :param input_channels: number of channel dimensions in input data.
#         :param start_filts:  number of feature_maps in first layer. rest is scaled accordingly.
#         :param out_channels: number of feature_maps for output_layers of all levels in decoder.
#         :param conv: instance of custom conv class containing the dimension info.
#         :param res_architecture: string deciding whether to use "resnet50" or "resnet101".
#         :param operate_stride1: boolean flag. enables adding of Pyramid levels P1 (output stride 2) and P0 (output stride 1).
#         :param norm: string specifying type of feature map normalization. If None, no normalization is applied.
#         :param relu: string specifying type of nonlinearity. If None, no nonlinearity is applied.
#         :param sixth_pooling: boolean flag. enables adding of Pyramid level P6.
#         """
#         super(FPN, self).__init__()

#         self.start_filts = cf.start_filts
#         start_filts = self.start_filts
#         self.n_blocks = [3, 4, {"resnet50": 6, "resnet101": 23}[cf.res_architecture], 3]
#         self.block = ResBlock
#         self.block_expansion = 4
#         self.operate_stride1 = operate_stride1
#         self.sixth_pooling = cf.sixth_pooling
#         self.dim = conv.dim

#         if operate_stride1:
#             self.C0 = nn.Sequential(conv(cf.n_channels, start_filts, ks=3, pad=1, norm=cf.norm, relu=cf.relu),
#                                     conv(start_filts, start_filts, ks=3, pad=1, norm=cf.norm, relu=cf.relu))

#             self.C1 = conv(start_filts, start_filts, ks=7, stride=(2, 2, 1) if conv.dim == 3 else 2, pad=3, norm=cf.norm, relu=cf.relu)

#         else:
#             self.C1 = conv(cf.n_channels, start_filts, ks=7, stride=(2, 2, 1) if conv.dim == 3 else 2, pad=3, norm=cf.norm, relu=cf.relu)

#         start_filts_exp = start_filts * self.block_expansion

#         C2_layers = []
#         C2_layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#                          if conv.dim == 2 else nn.MaxPool3d(kernel_size=3, stride=(2, 2, 1), padding=1))
#         C2_layers.append(self.block(start_filts, start_filts, conv=conv, stride=1, norm=cf.norm, relu=cf.relu,
#                                     downsample=(start_filts, self.block_expansion, 1)))
#         for i in range(1, self.n_blocks[0]):
#             C2_layers.append(self.block(start_filts_exp, start_filts, conv=conv, norm=cf.norm, relu=cf.relu))
#         self.C2 = nn.Sequential(*C2_layers)

#         C3_layers = []
#         C3_layers.append(self.block(start_filts_exp, start_filts * 2, conv=conv, stride=2, norm=cf.norm, relu=cf.relu,
#                                     downsample=(start_filts_exp, 2, 2)))
#         for i in range(1, self.n_blocks[1]):
#             C3_layers.append(self.block(start_filts_exp * 2, start_filts * 2, conv=conv, norm=cf.norm, relu=cf.relu))
#         self.C3 = nn.Sequential(*C3_layers)

#         C4_layers = []
#         C4_layers.append(self.block(
#             start_filts_exp * 2, start_filts * 4, conv=conv, stride=2, norm=cf.norm, relu=cf.relu, downsample=(start_filts_exp * 2, 2, 2)))
#         for i in range(1, self.n_blocks[2]):
#             C4_layers.append(self.block(start_filts_exp * 4, start_filts * 4, conv=conv, norm=cf.norm, relu=cf.relu))
#         self.C4 = nn.Sequential(*C4_layers)

#         C5_layers = []
#         C5_layers.append(self.block(
#             start_filts_exp * 4, start_filts * 8, conv=conv, stride=2, norm=cf.norm, relu=cf.relu, downsample=(start_filts_exp * 4, 2, 2)))
#         for i in range(1, self.n_blocks[3]):
#             C5_layers.append(self.block(start_filts_exp * 8, start_filts * 8, conv=conv, norm=cf.norm, relu=cf.relu))
#         self.C5 = nn.Sequential(*C5_layers)

#         if self.sixth_pooling:
#             C6_layers = []
#             C6_layers.append(self.block(
#                 start_filts_exp * 8, start_filts * 16, conv=conv, stride=2, norm=cf.norm, relu=cf.relu, downsample=(start_filts_exp * 8, 2, 2)))
#             for i in range(1, self.n_blocks[3]):
#                 C6_layers.append(self.block(start_filts_exp * 16, start_filts * 16, conv=conv, norm=cf.norm, relu=cf.relu))
#             self.C6 = nn.Sequential(*C6_layers)

#         if conv.dim == 2:
#             self.P1_upsample = Interpolate(scale_factor=2, mode='bilinear')
#             self.P2_upsample = Interpolate(scale_factor=2, mode='bilinear')
#         else:
#             self.P1_upsample = Interpolate(scale_factor=(2, 2, 1), mode='trilinear')
#             self.P2_upsample = Interpolate(scale_factor=(2, 2, 1), mode='trilinear')

#         self.out_channels = cf.end_filts
#         self.P5_conv1 = conv(start_filts*32 + cf.n_latent_dims, self.out_channels, ks=1, stride=1, relu=None) #
#         self.P4_conv1 = conv(start_filts*16, self.out_channels, ks=1, stride=1, relu=None)
#         self.P3_conv1 = conv(start_filts*8, self.out_channels, ks=1, stride=1, relu=None)
#         self.P2_conv1 = conv(start_filts*4, self.out_channels, ks=1, stride=1, relu=None)
#         self.P1_conv1 = conv(start_filts, self.out_channels, ks=1, stride=1, relu=None)

#         if operate_stride1:
#             self.P0_conv1 = conv(start_filts, self.out_channels, ks=1, stride=1, relu=None)
#             self.P0_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)

#         self.P1_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)
#         self.P2_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)
#         self.P3_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)
#         self.P4_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)
#         self.P5_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)

#         if self.sixth_pooling:
#             self.P6_conv1 = conv(start_filts * 64, self.out_channels, ks=1, stride=1, relu=None)
#             self.P6_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)


#     def forward(self, x):
#         """
#         :param x: input image of shape (b, c, y, x, (z))
#         :return: list of output feature maps per pyramid level, each with shape (b, c, y, x, (z)).
#         """
#         if self.operate_stride1:
#             c0_out = self.C0(x)
#         else:
#             c0_out = x

#         c1_out = self.C1(c0_out)
#         c2_out = self.C2(c1_out)
#         c3_out = self.C3(c2_out)
#         c4_out = self.C4(c3_out)
#         c5_out = self.C5(c4_out)
#         if self.sixth_pooling:
#             c6_out = self.C6(c5_out)
#             p6_pre_out = self.P6_conv1(c6_out)
#             p5_pre_out = self.P5_conv1(c5_out) + F.interpolate(p6_pre_out, scale_factor=2)
#         else:
#             p5_pre_out = self.P5_conv1(c5_out)

#         p4_pre_out = self.P4_conv1(c4_out) + F.interpolate(p5_pre_out, scale_factor=2)
#         p3_pre_out = self.P3_conv1(c3_out) + F.interpolate(p4_pre_out, scale_factor=2)
#         p2_pre_out = self.P2_conv1(c2_out) + F.interpolate(p3_pre_out, scale_factor=2)

#         # plot feature map shapes for debugging.
#         # for ii in [c0_out, c1_out, c2_out, c3_out, c4_out, c5_out, c6_out]:
#         #     print ("encoder shapes:", ii.shape)
#         #
#         # for ii in [p6_out, p5_out, p4_out, p3_out, p2_out, p1_out]:
#         #     print("decoder shapes:", ii.shape)

#         p2_out = self.P2_conv2(p2_pre_out)
#         p3_out = self.P3_conv2(p3_pre_out)
#         p4_out = self.P4_conv2(p4_pre_out)
#         p5_out = self.P5_conv2(p5_pre_out)
#         out_list = [p2_out, p3_out, p4_out, p5_out]

#         if self.sixth_pooling:
#             p6_out = self.P6_conv2(p6_pre_out)
#             out_list.append(p6_out)

#         if self.operate_stride1:
#             p1_pre_out = self.P1_conv1(c1_out) + self.P2_upsample(p2_pre_out)
#             p0_pre_out = self.P0_conv1(c0_out) + self.P1_upsample(p1_pre_out)
#             # p1_out = self.P1_conv2(p1_pre_out) # usually not needed.
#             p0_out = self.P0_conv2(p0_pre_out)
#             out_list = [p0_out] + out_list

#         return out_list



# class ResBlock(nn.Module):

#     def __init__(self, start_filts, planes, conv, stride=1, downsample=None, norm=None, relu='relu'):
#         super(ResBlock, self).__init__()
#         self.conv1 = conv(start_filts, planes, ks=1, stride=stride, norm=norm, relu=relu)
#         self.conv2 = conv(planes, planes, ks=3, pad=1, norm=norm, relu=relu)
#         self.conv3 = conv(planes, planes * 4, ks=1, norm=norm, relu=None)
#         self.relu = nn.ReLU(inplace=True) if relu == 'relu' else nn.LeakyReLU(inplace=True)
#         if downsample is not None:
#             self.downsample = conv(downsample[0], downsample[0] * downsample[1], ks=1, stride=downsample[2], norm=norm, relu=None)
#         else:
#             self.downsample = None
#         self.stride = stride

#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.conv2(out)
#         out = self.conv3(out)
#         if self.downsample:
#             residual = self.downsample(x)
#         out += residual
#         out = self.relu(out)
#         return out


# class Interpolate(nn.Module):
#     def __init__(self, scale_factor, mode):
#         super(Interpolate, self).__init__()
#         self.interp = nn.functional.interpolate
#         self.scale_factor = scale_factor
#         self.mode = mode

#     def forward(self, x):
#         x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
#         return x