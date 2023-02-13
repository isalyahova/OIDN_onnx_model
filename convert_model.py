import numpy as np
import torch
import torch.nn as nn
from tza import Reader
from model import UNet, concat


def convert_model(model, input_path_tza, output_path_onnx, input_names):
    weights = Reader(input_path_tza)
    layers_name = list(weights._table.keys())
    for i, layer in enumerate(layers_name):
        name, parameters = layer.split('.')
        layer_weights = nn.Parameter(torch.from_numpy(np.copy(weights[layer][0]))).data
        {
            'weight': model._modules[name].weight,
            'bias': model._modules[name].bias
        }[parameters].data = layer_weights

    dummy_input = torch.randn((1, 3, 224, 224))
    dynamic_axis = {axis: {0: 'batch_size', 1: 'channel', 2: 'height', 3: 'width'}
                    for axis in input_names + ['output']}
    torch.onnx.export(model,
                      args=(dummy_input, dummy_input),
                      f=output_path_onnx,
                      verbose=False,
                      export_params=True,
                      input_names=input_names,
                      output_names=['output'],
                      dynamic_axes=dynamic_axis)


def preprocess_rt_hdr_alb(inputs):
    input_color, input_albedo, input_normal = inputs
    input_color = torch.clamp(input_color, min=0)
    input_albedo = torch.clamp(input_albedo, min=0, max=1)
    return concat(input_color, input_albedo)


def get_model(num_input_channels):
    return UNet(num_input_channels)


def get_model_rt_hdr_alb():
    return get_model(6)
