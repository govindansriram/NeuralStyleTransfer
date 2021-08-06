import torch
import torch.nn.functional as F
from torch import nn, hub
from torchvision import models
import copy

hub.set_dir("/Users/sriramgovindan/Documents/App/projects/bloum/TorchModels")


def normalize_image(picture, device):
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    cnn_normalization_mean = cnn_normalization_mean.view(-1, 1, 1)
    cnn_normalization_std = cnn_normalization_std.view(-1, 1, 1)

    return (picture - cnn_normalization_mean) / cnn_normalization_std


class ModifiedVGG(nn.Module):
    def __init__(self, device):
        super(ModifiedVGG, self).__init__()
        self.__model = models.vgg19(pretrained=True).features
        self.__model.eval()
        self.__device = device

    def forward(self,
                image_tensor,
                conv_layer_list):

        image_tensor = normalize_image(image_tensor, self.__device)
        # conv_int = 0
        activation_list = []

        # for layer in self.__model:
        #     if isinstance(layer, nn.Conv2d):
        #         conv_int += 1
        #         image_tensor = layer(image_tensor)
        #         if conv_int in conv_layer_list:
        #             activation_list.append(image_tensor)
        #     elif isinstance(layer, nn.ReLU):
        #         image_tensor = F.relu(image_tensor, inplace=False)
        #         # image_tensor = self.__relu(image_tensor)
        #     else:
        #         image_tensor = layer(image_tensor)

        for idx, layer in enumerate(self.__model):
            if idx in conv_layer_list:
                image_tensor = layer(image_tensor)
                activation_list.append(image_tensor)
            elif isinstance(layer, nn.ReLU):
                image_tensor = F.relu(image_tensor, inplace=False)
                # image_tensor = self.__relu(image_tensor)
            else:
                image_tensor = layer(image_tensor)

        return activation_list
