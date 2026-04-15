import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict


class MNISTClassifier(nn.Module):
    def __init__(self, n_hidden_layers: int, nclasses: int = 10, imsize: int = 28, bn: bool = True):
        """
        Constructor of class MNISTClassifier

        :param n_hidden_layers: number of neurons in the hidden layer
        :param nclasses: number of class of images in MNIST dataset
        :param imsize: size ( = height = width) of an image in MNIST dataset
        :param bn: to use BatchNorm layer or not
        """
        super(MNISTClassifier, self).__init__()
        # TODO: define a Multi-Layer Perceptron (MLP) with 1 input layer, 1 hidden layer, 1 output layer
        # NOTE: the number of neurons in the hidden layer is defined by `n_hidden_layers`

        # NOTE: the resulted MLP is assigned to attribute self.net
        self.net = nn.Sequential(
            nn.Linear(imsize*imsize, n_hidden_layers),
            nn.Linear(n_hidden_layers, n_hidden_layers),
            nn.Linear(n_hidden_layers, nclasses)
        )
        '''
        Hint: 
        Linear layer (i.e. fully connected layer): https://pytorch.org/docs/1.7.0/generated/torch.nn.Linear.html?highlight=linear#torch.nn.Linear
        BatchNorm1d: https://pytorch.org/docs/1.7.0/generated/torch.nn.BatchNorm1d.html?highlight=batchnorm1d#torch.nn.BatchNorm1d
        ReLU: https://pytorch.org/docs/1.7.0/generated/torch.nn.ReLU.html?highlight=relu#torch.nn.ReLU
        Sequential: https://pytorch.org/docs/1.7.0/generated/torch.nn.Sequential.html?highlight=sequential#torch.nn.Sequential
        '''

        self.loss_fn = torch.nn.functional.cross_entropy

    def forward(self, data_dict: Dict):
        """
        This method define the forward pass of the model

        :param data_dict: {
            'image': torch.Tensor, shape (N, 784), a batch of FLATTENED images,
            'label': torch.Tensor, shape (N), a batch of labels
        }
        :return: {
            'loss': float
            'cls': predicted class for each image in the batch (only in EVAL mode)
            'prob': probability of predicted class for each image in the batch (only in EVAL mode)
        }
        """
        logits = self.net(data_dict['image'])  # (N, 10)

        ret_dict = dict()
        if self.training:
            loss = self.loss_fn(logits, data_dict['label'])
            ret_dict = {'loss': loss}
        else:
            cls, prob = self.decode_prediction(logits)
            ret_dict = {'cls': cls, 'prob': prob, 'logits': logits}
        return ret_dict

    @staticmethod
    def decode_prediction(logits: torch.Tensor):
        """
        TODO: Decode logits into predicted class and associated probability

        :param logits: (N, 10) - predicted logits for each image in the batch
        :return:
            * cls: (N) - predicted class
            * prob: (N) - probability of predicted class
        """
        #TODO
        cls = []
        prob = []

        for n in logits: # each n will be an array with 10 values of probabilities
            cls.append(np.argmax(n)) # predicted - highest value between the 10 values
            prob.append(max(n))

        return cls, prob
