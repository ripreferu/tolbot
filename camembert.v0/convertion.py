# import numpy as np
# import os
# import time
import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
import onnx
# from onnx_tf.backend import prepare
# import tensorflow as tf
model_pytorch = torch.load('/home/pierre/Documents/Bachelor_Arbeit/code/git_test/camembert.v0/model.pt', map_location= torch.device('cpu'))
dummy_input ="Le camembert est <mask> :)"
model_pytorch.eval()
model_pytorch.fill_mask(dummy_input, topk=5)
#torch.onnx.export(model_pytorch, dummy_input, '/home/pierre/Documents/Bachelor_Arbeit/code/git_test/model_test.onnx')
