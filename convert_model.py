import numpy as np
import torch
import torch.nn as nn
from tza import Reader
from model import UNet, concat
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', '-f', type=str, nargs='*', required=True,
                        choices=['hdr', 'ldr', 'albedo', 'alb', 'normal', 'nrm'],
                        help='Set of input features')
    parser.add_argument('--input_names', type=str, nargs='*', required=True,
                        help='Set of input features')
    parser.add_argument('--input_path_tza', type=Path, required=True,
                        help='Path to tza model')
    parser.add_argument('--output_path_onnx', type=Path, required=True,
                        help='Path to save converting model')
    cfg = parser.parse_args()
    return cfg


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
                      args=[dummy_input, dummy_input, dummy_input],
                      f=output_path_onnx,
                      verbose=False,
                      export_params=True,
                      input_names=input_names,
                      output_names=['output'],
                      dynamic_axes=dynamic_axis)


def preprocess_rt_hdr(inputs):
    input_color = inputs[0]
    input_color = torch.clamp(input_color, min=0)
    return input_color


def preprocess_rt_ldr(inputs):
    input_color = inputs[0]
    input_color = torch.clamp(input_color, min=0, max=1)
    return input_color


def preprocess_rt_hdr_alb(inputs):
    input_color, input_albedo = inputs[0], inputs[1]
    input_color = torch.clamp(input_color, min=0)
    input_albedo = torch.clamp(input_albedo, min=0, max=1)
    return concat(input_color, input_albedo)


def preprocess_rt_ldr_alb(inputs):
    input_color, input_albedo = inputs[0], inputs[1]
    input_color = torch.clamp(input_color, min=0, max=1)
    input_albedo = torch.clamp(input_albedo, min=0, max=1)
    return concat(input_color, input_albedo)


def preprocess_rt_hdr_alb_nrm(inputs):
    input_color, input_albedo, input_normal = inputs[0], inputs[1], inputs[2]
    input_color = torch.clamp(input_color, min=0)
    input_albedo = torch.clamp(input_albedo, min=0, max=1)
    input_normal = torch.clamp(input_normal, min=-1, max=1) * 0.5 + 0.5
    return concat(concat(input_color, input_albedo), input_normal)


def preprocess_rt_ldr_alb_nrm(inputs):
    input_color, input_albedo, input_normal = inputs[0], inputs[1], inputs[2]
    input_color = torch.clamp(input_color, min=0, max=1)
    input_albedo = torch.clamp(input_albedo, min=0, max=1)
    input_normal = torch.clamp(input_normal, min=-1, max=1) * 0.5 + 0.5
    return concat(concat(input_color, input_albedo), input_normal)


def get_model(num_input_channels, preprocess_func):
    return UNet(num_input_channels, 3, preprocess_func)


def get_model_rt_hdr():
    return get_model(3, preprocess_rt_hdr)


def get_model_rt_ldr():
    return get_model(3, preprocess_rt_ldr)


def get_model_rt_hdr_alb():
    return get_model(6, preprocess_rt_hdr_alb)


def get_model_rt_ldr_alb():
    return get_model(6, preprocess_rt_ldr_alb)


def get_model_rt_hdr_alb_nrm():
    return get_model(9, preprocess_rt_hdr_alb_nrm)


def get_model_rt_ldr_alb_nrm():
    return get_model(9, preprocess_rt_ldr_alb_nrm)


def main():
    cfg = parse_args()
    features, input_path_tza, output_path_onnx, input_names = cfg.features, cfg.input_path_tza, \
                                                              cfg.output_path_onnx, cfg.input_names
    if features == ['hdr']:
        model = get_model_rt_hdr()
    elif features == ['ldr']:
        model = get_model_rt_ldr()
    elif features == ['hdr', 'alb']:
        model = get_model_rt_hdr_alb()
    elif features == ['ldr', 'alb']:
        model = get_model_rt_ldr_alb()
    elif features == ['hdr', 'alb', 'nrm']:
        model = get_model_rt_hdr_alb_nrm()
    elif features == ['ldr', 'alb', 'nrm']:
        model = get_model_rt_ldr_alb_nrm()
    convert_model(model,
                  input_path_tza=input_path_tza,
                  output_path_onnx=output_path_onnx,
                  input_names=input_names)


if __name__ == '__main__':
    main()
