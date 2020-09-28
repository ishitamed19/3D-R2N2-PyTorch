#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 21:04:40 2018

@author: wangchu
"""


from models.net import Net
from lib.layers import SoftmaxWithLoss3D
from lib.config import cfg

import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

##########################################################################################
#                                                                                        #
#                      Original GRUNet definition using PyTorch                          #
#                                                                                        #
##########################################################################################
class BaseGRUNet(Net):
    """
    This class is used to define some common attributes and methods that both GRUNet and 
    ResidualGRUNet have. Note that GRUNet and ResidualGRUNet have the same loss function
    and forward pass. The only difference is different encoder and decoder architecture.
    """
    def __init__(self):
        print("initializing \"BaseGRUNet\"")
        super(BaseGRUNet, self).__init__()
        """
        Set the necessary data of the network
        """
        if cfg.CONST.N_VIEWS==1:
            self.is_x_tensor4 = True #singleview
        else:
            self.is_x_tensor4 = False #multiview
        
        self.n_gru_vox = 4
        #the size of x is (num_views, batch_size, 3, img_w, img_h)
        self.input_shape = (self.batch_size, 3, self.img_w, self.img_h)
        #number of filters for each convolution layer in the encoder
        self.n_convfilter = [96, 128, 256, 256, 256, 256]
        #the dimension of the fully connected layer
        self.n_fc_filters = [1024]
        #number of filters for each 3d convolution layer in the decoder
        self.n_deconvfilter = [128, 128, 128, 64, 32, 2]
        #the size of the hidden state
        self.h_shape = (self.batch_size, self.n_deconvfilter[0], self.n_gru_vox, self.n_gru_vox, self.n_gru_vox)
        #the filter shape of the 3d convolutional gru unit
        self.conv3d_filter_shape = (self.n_deconvfilter[0], self.n_deconvfilter[0], 3, 3, 3)
        
        #set the last layer 
        self.SoftmaxWithLoss3D = SoftmaxWithLoss3D()
        
        
        #set the encoder and the decoder of the network
        self.encoder = None
        self.decoder = None
        
        
    def forward(self, x, y=None, global_step=0, test=True):
        #ensure that the network has encoder and decoder attributes
        if self.encoder is None:
            raise Exception("subclass network of BaseGRUNet must define the \"encoder\" attribute")
        if self.decoder is None:
            raise Exception("subclass network of BaseGRUNet must define the \"decoder\" attribute")

        #initialize the hidden state and update gate
        h = self.initHidden(self.h_shape)
        u = self.initHidden(self.h_shape)
        
        # since we are training for single view and batch
        x = x.view(cfg.CONST.BATCH_SIZE, 3, 127, 127)
        y = y.view(cfg.CONST.BATCH_SIZE, y.size()[1], y.size()[2], y.size()[3], y.size()[4])
        
        #a list used to store intermediate update gate activations
        u_list = []
        
        """
        x is the input and the size of x is (num_views, batch_size, channels, heights, widths).
        h and u is the hidden state and activation of last time step respectively.
        The following loop computes the forward pass of the whole network. 
        """
            
        for time in range(x.size(0)):
            gru_out, update_gate = self.encoder(x, h, u, time)

            h = gru_out

            u = update_gate
            u_list.append(u)

        out = self.decoder(h)

        """
        If test is True and y is None, then the out is the [prediction].
        If test is True and y is not None, then the out is [prediction, loss].
        If test is False and y is not None, then the out is loss.
        """
        out = self.SoftmaxWithLoss3D(out, y=y, test=test)
        if test:
            out.extend(u_list)
        return out
    
    def initHidden(self, h_shape):
        h = torch.zeros(h_shape)
        if torch.cuda.is_available():
            h = h.cuda()
        return Variable(h)
    

    
class BaseGRUNetHypernet(Net):
    """
    This class is used to define some common attributes and methods that both GRUNet and 
    ResidualGRUNet have. Note that GRUNet and ResidualGRUNet have the same loss function
    and forward pass. The only difference is different encoder and decoder architecture.
    """
    def __init__(self):
        print("initializing \"BaseGRUNetHypernet\"")
        super(BaseGRUNetHypernet, self).__init__()
        """
        Set the necessary data of the network
        """
        if cfg.CONST.N_VIEWS==1:
            self.is_x_tensor4 = True #singleview
        else:
            self.is_x_tensor4 = False #multiview
        
        self.n_gru_vox = 4
        #the size of x is (num_views, batch_size, 3, img_w, img_h)
        self.input_shape = (self.batch_size, 3, self.img_w, self.img_h)
        #number of filters for each convolution layer in the encoder
        self.n_convfilter = [96, 128, 256, 256, 256, 256]
        #the dimension of the fully connected layer
        self.n_fc_filters = [1024]
        #number of filters for each 3d convolution layer in the decoder
        self.n_deconvfilter = [128, 128, 128, 64, 32, 2]
        #the size of the hidden state
        self.h_shape = (self.batch_size, self.n_deconvfilter[0], self.n_gru_vox, self.n_gru_vox, self.n_gru_vox)
        #the filter shape of the 3d convolutional gru unit
        self.conv3d_filter_shape = (self.n_deconvfilter[0], self.n_deconvfilter[0], 3, 3, 3)
        
        #set the last layer 
        self.SoftmaxWithLoss3D = SoftmaxWithLoss3D()
        
        
        #set the encoder and the decoder of the network
        self.encoder = None
        self.decoder = None
        
        #set the hypernet and the embedding generator
        self.hypernet = HyperNet()
        
    def forward(self, x, y=None, global_step=0, test=True):
        #ensure that the network has encoder and decoder attributes
        if self.encoder is None:
            raise Exception("subclass network of BaseGRUNet must define the \"encoder\" attribute")
        if self.decoder is None:
            raise Exception("subclass network of BaseGRUNet must define the \"decoder\" attribute")

        #initialize the hidden state and update gate
        h = self.initHidden(self.h_shape)
        u = self.initHidden(self.h_shape)
        
        
        
        #a list used to store intermediate update gate activations
        u_list = []
        
        """
        x is the input and the size of x is (num_views, batch_size, channels, heights, widths).
        h and u is the hidden state and activation of last time step respectively.
        The following loop computes the forward pass of the whole network. 
        """
        if self.is_x_tensor4:
            '''x: (batch_size, channels, heights, widths) SINGLE VIEW'''
            
            
            net_loss = 0
            prob_list = []
            loss_list = []
            ulist_list = []
            
            x = x.view(cfg.CONST.BATCH_SIZE, 3, 127, 127)
            y = y.view(cfg.CONST.BATCH_SIZE, y.size()[1], y.size()[2], y.size()[3], y.size()[4])
                
            vqloss, feat_kernels_enc_conv, feat_bias_enc_conv, feat_kernels_enc_fc, feat_bias_enc_fc, feat_kernels_enc_3dgru, feat_bias_enc_3dgru, feat_kernels_dec_conv, feat_bias_dec_conv = self.hypernet(x, global_step, test)
                
            # Visualize weights on TB
#             if global_step<=20:
#                 self.tb_logger.add_histogram('resnet_embedding', self.hypernet.encodingnet(x), global_step)
#                 self.tb_logger.add_histogram('vqvae_embedding', self.hypernet.embedding.weight, global_step)
                
#                 for i, name in enumerate(self.hypernet.enc_conv_names):
#                     self.tb_logger.add_histogram(name+'.bias', feat_bias_enc_conv[i], global_step)
#                     self.tb_logger.add_histogram(name+'.weight', feat_kernels_enc_conv[i], global_step)

#                 for i, name in enumerate(self.hypernet.enc_fc_names):
#                     self.tb_logger.add_histogram(name+'.bias', feat_bias_enc_fc[i], global_step)
#                     self.tb_logger.add_histogram(name+'.weight', feat_kernels_enc_fc[i], global_step)

#                 for i, name in enumerate(self.hypernet.enc_3dgru_names):
#                     if i%2==1:
#                         self.tb_logger.add_histogram(name+'.bias', feat_bias_enc_3dgru[i//2], global_step)
#                     self.tb_logger.add_histogram(name+'.weight', feat_kernels_enc_3dgru[i], global_step)

#                 for i, name in enumerate(self.hypernet.dec_conv_names):
#                     self.tb_logger.add_histogram(name+'.bias', feat_bias_dec_conv[i], global_step)
#                     self.tb_logger.add_histogram(name+'.weight', feat_kernels_dec_conv[i], global_step)
                
        
            gru_out, update_gate = self.encoder(x, h, u, 0, feat_kernels_enc_conv, feat_bias_enc_conv, feat_kernels_enc_fc, feat_bias_enc_fc, feat_kernels_enc_3dgru, feat_bias_enc_3dgru)

            h = gru_out

            u = update_gate
            u_list.append(u)

            out = self.decoder(h, feat_kernels_dec_conv, feat_bias_dec_conv)
            """
            If test is True and y is None, then the out is the [prediction].
            If test is True and y is not None, then the out is [prediction, loss].
            If test is False and y is not None, then the out is loss.
            """
            
            out = self.SoftmaxWithLoss3D(out, y=y, test=test)
           
            if test:
                out[1] += vqloss
                out.extend(u_list)
            else:
                out += vqloss
                
        return out
    
    def initHidden(self, h_shape):
        h = torch.zeros(h_shape)
        if torch.cuda.is_available():
            h = h.cuda()
        return Variable(h)    
    
    
##########################################################################################
#                                                                                        #
#                      HYPERNET definition                                               #
#                                                                                        #
##########################################################################################        
        
        
vqvae_dict_size = cfg.CONST.vqvae_dict_size
dynamic_hypernet_probability_thresh = cfg.CONST.dynamic_hypernet_probability_thresh
hyptotal_instances = cfg.CONST.hyptotal_instances
hypernet_nonlinear = cfg.CONST.hypernet_nonlinear
B = cfg.CONST.BATCH_SIZE
use_resnet_for_hypernet = cfg.CONST.use_resnet_for_hypernet


class HyperNet(nn.Module):
    def __init__(self):
        super(HyperNet, self).__init__()

        ''' Encoder Weights/Biases Sizes '''
        self.enc_conv_wts_size = [[96, 3, 7, 7], [96, 96, 3, 3], [128, 96, 3, 3], [128, 128, 3, 3], [128, 96, 1, 1], [256, 128, 3, 3], [256, 256, 3, 3], [256, 128, 1, 1], [256, 256, 3, 3], [256, 256, 3, 3], [256, 256, 3, 3], [256, 256, 3, 3], [256, 256, 1, 1], [256, 256, 3, 3], [256, 256, 3, 3]]
        self.enc_conv_bias_size = [[96], [96], [128], [128], [128], [256], [256], [256], [256], [256], [256], [256], [256], [256], [256]]
        self.enc_conv_names = ['conv1a', 'conv1b', 'conv2a', 'conv2b', 'conv2c', 'conv3a', 'conv3b', 'conv3c', 'conv4a', 'conv4b', 'conv5a', 'conv5b', 'conv5c', 'conv6a', 'conv6b']

        self.enc_fc_wts_size = [[1024, 2304]]
        self.enc_fc_bias_size = [[1024]]
        self.enc_fc_names = ['fc7']

        # no bias in this
        self.enc_3dgru_wts_size = [[8192, 1024], [128, 128, 3, 3, 3], [8192, 1024], [128, 128, 3, 3, 3], [8192, 1024], [128, 128, 3, 3, 3]]
        self.enc_3dgru_bias_size  = [[1, 128, 1, 1, 1], [1, 128, 1, 1, 1], [1, 128, 1, 1, 1]]
        self.enc_3dgru_names = ['t_x_s_update.fc_layer', 't_x_s_update.conv3d', 't_x_s_reset.fc_layer', 't_x_s_reset.conv3d', 't_x_rs.fc_layer', 't_x_rs.conv3d']

        ''' Decoder Weights/Biases Sizes '''
        self.dec_conv_wts_size = [[128, 128, 3, 3, 3], [128, 128, 3, 3, 3], [128, 128, 3, 3, 3], [128, 128, 3, 3, 3], [64, 128, 3, 3, 3], [64, 64, 3, 3, 3], [64, 128, 1, 1, 1], [32, 64, 3, 3, 3], [32, 32, 3, 3, 3], [32, 32, 3, 3, 3], [2, 32, 3, 3, 3]]
        self.dec_conv_bias_size = [[128], [128], [128], [128], [64], [64], [64], [32], [32], [32], [2]]
        self.dec_conv_names = ['conv7a', 'conv7b', 'conv8a', 'conv8b', 'conv9a', 'conv9b', 'conv9c', 'conv10a', 'conv10b', 'conv10c', 'conv11']

        self.emb_dimension = 16

        total_instances = hyptotal_instances
        range_val = 1
        variance = ((2*range_val))/(12**0.5)
        lambda_val = 1
        # st()

        if use_resnet_for_hypernet:
            variance = 0.075
            lambda_val = 0.5
            min_embed = -0.35
            max_embed = 0.6
        else:
            variance = 0.075
            min_embed = -0.25
            max_embed = 0.25                               

        lambda_val = [lambda_val]*10 + [5,5]
        # st()
        
        self.encodingnet = ResnetEncoder()
        self.embedding = nn.Embedding(vqvae_dict_size, self.emb_dimension)
        nn.init.normal_(self.embedding.weight, mean=0.5, std=1.75)
        self.prototype_usage = torch.zeros(vqvae_dict_size).cuda()

        if hypernet_nonlinear:
            self.hidden1 = nn.Linear(16, 32)
            self.hidden2 = nn.Linear(32, 16)

        self.commitment_cost = 0.25
        bias_constant = 0.1


        # Dummy values for occ and rgbnet hypernet testing
        
        weight_variances_conv_enc = [0.0287/20, 0.0481/20, 0.044/20, 0.0416/20, 0.1336/20, 0.0340/20, 0.0294/20, 0.1020/20, 0.0294/20, 0.0294/20, 0.0294/20, 0.0294/20, 0.0883/20, 0.0294/20, 0.0294/20]
        weight_variances_fc_enc = [0.03466/20]
        weight_variances_3dgru_enc = [0.0208/20, 0.0051/20, 0.0208/20, 0.0051/20, 0.0208/20, 0.0051/20]
        
        weight_variances_conv_dec = [0.00514/20, 0.00514/20, 0.00514/20, 0.00514/20, 0.00719/20, 0.01018/20, 0.02192/20, 0.0140/20, 0.01992/20, 0.0199/20, 0.0527/20]

        
        self.kernel_conv_encoderWeights = nn.ParameterList([Parameter(torch.nn.init.normal_(torch.empty(self.emb_dimension, self.total(i)), mean=0, std=(variance*weight_variances_conv_enc[index])/((variance**2-weight_variances_conv_enc[index]**2)**0.5)),requires_grad=True) for index,i in enumerate(self.enc_conv_wts_size)])
        self.bias_conv_encoderWeights = nn.ParameterList([Parameter(torch.nn.init.constant_(torch.empty(i),val=0.1), requires_grad=True) for index,i in enumerate(self.enc_conv_bias_size)])

        self.kernel_fc_encoderWeights = nn.ParameterList([Parameter(torch.nn.init.normal_(torch.empty(self.emb_dimension, self.total(i)), mean=0, std=(variance*weight_variances_fc_enc[index])/((variance**2-weight_variances_fc_enc[index]**2)**0.5)),requires_grad=True) for index,i in enumerate(self.enc_fc_wts_size)])
        self.bias_fc_encoderWeights = nn.ParameterList([Parameter(torch.nn.init.constant_(torch.empty(i),val=0.1), requires_grad=True) for index,i in enumerate(self.enc_fc_bias_size)])

        self.kernel_3dgru_encoderWeights = nn.ParameterList([Parameter(torch.nn.init.normal_(torch.empty(self.emb_dimension, self.total(i)), mean=0, std=(variance*weight_variances_3dgru_enc[index])/((variance**2-weight_variances_3dgru_enc[index]**2)**0.5)),requires_grad=True) for index,i in enumerate(self.enc_3dgru_wts_size)])
        self.bias_3dgru_encoderWeights = nn.ParameterList([Parameter(torch.nn.init.constant_(torch.empty(i),val=0.1), requires_grad=True) for index,i in enumerate(self.enc_3dgru_bias_size)])

        self.kernel_conv_decoderWeights = nn.ParameterList([Parameter(torch.nn.init.normal_(torch.empty(self.emb_dimension, self.total(i)), mean=0, std=(variance*weight_variances_conv_dec[index])/((variance**2-weight_variances_conv_dec[index]**2)**0.5)),requires_grad=True) for index,i in enumerate(self.dec_conv_wts_size)])
        self.bias_conv_decoderWeights = nn.ParameterList([Parameter(torch.nn.init.constant_(torch.empty(i),val=0.1), requires_grad=True) for index,i in enumerate(self.dec_conv_bias_size)])

        print("initalized")

    

    def total(self,tensor_shape):
        prod = 1
        for i in tensor_shape:
            prod = prod*i
        return prod

    def update_embedding_dynamically(self):

        total = torch.sum(self.prototype_usage)
        probs = self.prototype_usage/total
        # Select 2 good embeds and take mean
        good_embeds_idxs = torch.where(probs>0.2)[0]
        if good_embeds_idxs.shape[0] < 2:
            print("Not enough highly used embedding. Skipping dynamic update.")
            return 

        random_idxs = torch.randperm(good_embeds_idxs.shape[0])[:2]
        good_embeds_idxs_random = good_embeds_idxs[random_idxs]
        good_embeds_random = self.embedding(good_embeds_idxs_random)
        good_embeds_random_detached = good_embeds_random.clone().detach()
        good_embeds_random_mean = torch.mean(good_embeds_random_detached, dim=0)

        # select 1 embed with 0 prob
        bad_embeds_idx = torch.where(probs <= 1e-7)[0]
        if bad_embeds_idx.shape[0] == 0:
            print("Not enough 0 used embedding. Skipping dynamic update.")
            return 

        random_idxs = torch.randperm(bad_embeds_idx.shape[0])[:1]
        bad_embeds_idxs_random = bad_embeds_idx[random_idxs]
        
        # replace embedding at bad_embeds_idxs_random with good_embeds_random_mean
        self.embedding.weight[bad_embeds_idxs_random] = good_embeds_random_mean

        self.prototype_usage *= 0 # clear usage

    def forward(self, rgb, global_step, test):
        # input: rgb (1, 3, 127, 127)
        loss = 0
        
        if not test:
            if cfg.CONST.dynamic_dict:
                if (global_step%1000) ==0:
                    with torch.no_grad():
                        self.update_embedding_dynamically()

        embed = self.encodingnet(rgb)
        #print(embed.mean(), embed.std())

        embed_shape = embed.shape 


        distances = (torch.sum(embed**2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.matmul(embed, self.embedding.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        for idx in encoding_indices.view(-1):
            self.prototype_usage[idx] += 1
 
        encodings = torch.zeros(encoding_indices.shape[0], vqvae_dict_size, device=embed.device) 
        encodings.scatter_(1, encoding_indices, 1) 

        '''Quantize and unflatten'''
        quantized = torch.matmul(encodings, self.embedding.weight).view(embed_shape)
        
        '''Loss'''
        e_latent_loss = F.mse_loss(quantized.detach(), embed)
        q_latent_loss = F.mse_loss(quantized, embed.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = embed + (quantized - embed).detach()
        embed = quantized


        if hypernet_nonlinear:
            embed = self.hidden1(embed)
            embed = F.leaky_relu(embed)
            embed = self.hidden2(embed)
            embed = F.leaky_relu(embed)
       
        
        
        feat_kernels_enc_conv = [(torch.matmul(embed,self.kernel_conv_encoderWeights[i])).view(self.enc_conv_wts_size[i]) for i in range(len(self.kernel_conv_encoderWeights))]
        feat_bias_enc_conv = self.bias_conv_encoderWeights #[(torch.matmul(embed,self.bias_conv_encoderWeights[i])).view([B]+self.enc_conv_bias_size[i][0]) for i in range(len(self.bias_conv_encoderWeights))]

        feat_kernels_enc_fc = [(torch.matmul(embed,self.kernel_fc_encoderWeights[i])).view(self.enc_fc_wts_size[i]) for i in range(len(self.kernel_fc_encoderWeights))]
        feat_bias_enc_fc = self.bias_fc_encoderWeights #[(torch.matmul(embed,self.bias_fc_encoderWeights[i])).view([B]+self.enc_fc_bias_size[i][0]) for i in range(len(self.bias_fc_encoderWeights))]

        feat_kernels_dec_conv = [(torch.matmul(embed,self.kernel_conv_decoderWeights[i])).view(self.dec_conv_wts_size[i]) for i in range(len(self.kernel_conv_decoderWeights))]
        feat_bias_dec_conv = self.bias_conv_decoderWeights #[(torch.matmul(embed,self.bias_conv_decoderWeights[i])).view([B]+self.dec_conv_bias_size[i][0]) for i in range(len(self.bias_conv_decoderWeights))]

        feat_kernels_enc_3dgru = [(torch.matmul(embed,self.kernel_3dgru_encoderWeights[i])).view(self.enc_3dgru_wts_size[i]) for i in range(len(self.kernel_3dgru_encoderWeights))]
        feat_bias_enc_3dgru = self.bias_3dgru_encoderWeights #[(torch.matmul(embed,self.bias_3dgru_encoderWeights[i])).view([B]+self.enc_3dgru_bias_size[i]) for i in range(len(self.bias_3dgru_encoderWeights))]
        

        return loss, feat_kernels_enc_conv, feat_bias_enc_conv, feat_kernels_enc_fc, feat_bias_enc_fc, feat_kernels_enc_3dgru, feat_bias_enc_3dgru, feat_kernels_dec_conv, feat_bias_dec_conv        
        
        
        
##########################################################################################
#                                                                                        #
#                      ENCODER definition                                                #
#                                                                                        #
##########################################################################################         

class ResnetEncoder(nn.Module):
    def __init__(self):
        super(ResnetEncoder, self).__init__()
        if cfg.CONST.use_resnet_for_hypernet:
            self.encodingnet = models.resnet18().cuda()
            self.encodingnet.fc = nn.Linear(512, 16).cuda()
        else:
            activ = nn.LeakyReLU
            self.encodingnet = nn.Sequential(
                nn.Conv2d(3, 4, 4, stride=2, padding=1),
                activ(),
                nn.Conv2d(4, 8, 4, stride=2, padding=1),
                activ(),
                nn.Conv2d(8, 16, 4, stride=2, padding=1),
                activ(),
                nn.Conv2d(16, 32, 4, stride=2, padding=1),
                activ(),
                nn.Conv2d(32, 64, 4, stride=2, padding=1),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, 16),
                activ(),
                nn.Linear(16, 16),
            ).cuda()

    def forward(self, input):
        # input -> B, 3, 128, 128  
        # label -> B
        encoding = self.encodingnet(input)
        return encoding        
        
        
        
        