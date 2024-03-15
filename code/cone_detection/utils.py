import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

def weight_histograms_conv2d(writer, step, weights, layer_number):
  weights_shape = weights.shape
  num_kernels = weights_shape[0]
  for k in range(num_kernels):
    flattened_weights = weights[k].flatten()
    tag = f"Weights/layer_{layer_number}/kernel_{k}"
    writer.add_histogram(tag, flattened_weights, global_step=step, bins='tensorflow')


def weight_histograms_linear(writer, step, weights, layer_number):
  flattened_weights = weights.flatten()
  tag = f"Weights/layer_{layer_number}"
  writer.add_histogram(tag, flattened_weights, global_step=step, bins='tensorflow')


def log_weight_histograms(writer, step, model):
  # Iterate over all model layers
  for layer_number in range(len(model.layers)):
    # Get layer
    layer = model.layers[layer_number]
    # Compute weight histograms for appropriate layer
    if isinstance(layer, nn.Conv2d):
      weights = layer.weight
      weight_histograms_conv2d(writer, step, weights, layer_number)
    elif isinstance(layer, nn.Linear):
      weights = layer.weight
      weight_histograms_linear(writer, step, weights, layer_number)

def log_gradient_histograms(writer, step, model):
    # Iterate over all model layers
    for layer_number in range(len(model.layers)):
      # Get layer
      layer = model.layers[layer_number]

      for tag, value in layer.named_parameters():
          if value.grad is not None:
              log_tag = f"Grads/layer_{layer_number}/{tag}"
              writer.add_histogram(log_tag, value.grad.cpu(), global_step=step)

