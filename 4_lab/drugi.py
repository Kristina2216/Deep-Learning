import torch
import torch.nn as nn
import torch.nn.functional as F


class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, bias=True):
        super(_BNReluConv, self).__init__()
        self.add_module("batch_norm", torch.nn.BatchNorm2d(num_maps_in))
        self.add_module('relu', nn.ReLU(inplace = True))
        #padding = k // 2
        self.add_module('conv', torch.nn.Conv2d(num_maps_in, num_maps_out, kernel_size=k, bias=bias)) #, padding=padding))
        # YOUR CODE HERE

class SimpleMetricEmbedding(nn.Module):
    def __init__(self, input_channels, emb_size=32):
        super().__init__()
        self.emb_size = emb_size
        # YOUR CODE HERE
        self.mods = nn.Sequential()
        self.mods.add_module("CNReluConv_1", _BNReluConv(1, emb_size, 3))
        self.mods.add_module("MaxPool_1", torch.nn.MaxPool2d(kernel_size=3, stride=2))
        self.mods.add_module("CNReluConv_2", _BNReluConv(emb_size, emb_size, 3))
        self.mods.add_module("MaxPool_2", torch.nn.MaxPool2d(kernel_size=3, stride=2))
        self.mods.add_module("CNReluConv_3", _BNReluConv(emb_size, emb_size, 3))
        self.mods.add_module("AveragePool_3", torch.nn.AvgPool2d(kernel_size=2))

    def get_features(self, img):
        # Returns tensor with dimensions BATCH_SIZE, EMB_SIZE
        # YOUR CODE HERE
        x = self.mods.forward(img)
        print(torch.flatten(x, start_dim = 1).size())
        return torch.flatten(x, start_dim = 1)

    def loss(self, anchor, positive, negative):
        a_x = self.get_features(anchor)
        p_x = self.get_features(positive)
        n_x = self.get_features(negative)
        # YOUR CODE HERE
        triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
        return triplet_loss(anchor, positive, negative)