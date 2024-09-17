
"""
    Requires CompressAI environment
"""


import math
import io
import os
import torch
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import time
from calflops import calculate_flops
import onnx
import onnxruntime

from PIL import Image


#importing the model
from compressai.zoo import bmshj2018_factorized_relu
from compressai.zoo import models

#checking the device available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device="cpu"
print(device)



#exporting the PyTorch model to the ONNX version
net = models["bmshj2018-factorized"](quality=4, metric="ms-ssim", pretrained=True)
# net = cheng2020_anchor(quality=5, pretrained=True).to(device)

# Some dummy input
x=torch.randn(1,3,1232,1640,requires_grad=True) #x = torch.randn(1, 1, 720, 1280, requires_grad=True) Cannot work by simply changing the channels as the model is trained for the color images

#export_path=r"C:\Swapnil\Narrowband_DRONE\Image_compression_code\Method_7_compress_ai\compressai\examples\bmshj2018_factorized_models"
export_path=r"C:\Swapnil\Narrowband_DRONE\Image_compression_code\Final_DL_compession_September_24\onnx_scripts\onnx_export\onnx_models"
export_path=os.path.join(export_path,"bmshj_halfUHD_ssim_4.onnx")
# Export the model
torch.onnx.export(net,                       # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  export_path,              # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=14,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input': {0 : 'batch_size'},    # variable length axes
                                'output': {0 : 'batch_size'}}
                 )
print("Model is exported")

onnx_model = onnx.load(export_path)
onnx_model_graph = onnx_model.graph
onnx_session = onnxruntime.InferenceSession(onnx_model.SerializeToString())
# onnx_session = onnxruntime.InferenceSession("cheng2020.onnx")

input_shape = (1, 3, 1232, 1640)
x = torch.randn(input_shape).numpy()
print("size of x:",x.size)

input_names = ["input"]
output_names = ["output"]

onnx_output = onnx_session.run(output_names, {input_names[0]: x})[0]



