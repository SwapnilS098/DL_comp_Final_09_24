import math
import io
import os
import torch
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import time

from PIL import Image
import onnx
from compressai.zoo import models
import onnxruntime
import cv2



def model_handling():
    
    net = models["bmshj2018-factorized"](quality=4, metric="ms-ssim", pretrained=True)
    # net = cheng2020_anchor(quality=5, pretrained=True).to(device)

    # Some dummy input
    x=torch.randn(1,3,720,1280,requires_grad=True) #x = torch.randn(1, 1, 720, 1280, requires_grad=True) Cannot work by simply changing the channels as the model is trained for the color images

    export_path=r"C:\Swapnil\Narrowband_DRONE\Image_compression_code\Method_7_compress_ai\compressai\examples\bmshj2018_factorized_models"
    export_path=os.path.join(export_path,"bmshj_model_hd_ssim.onnx")
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

    input_shape = (1, 3, 720, 1280)
    x = torch.randn(input_shape).numpy()
    print("size of x:",x.size)

    input_names = ["input"]
    output_names = ["output"]

    onnx_output = onnx_session.run(output_names, {input_names[0]: x})[0]

def preprocess_image(image_path):
    """
    Preprocess the input image to the format expected by the model.
    """
    # Load the image using PIL
    image = Image.open(image_path).convert("RGB")

    image=image.resize((1280,720))
    
    # Convert to numpy array and normalize the image (0 to 1)
    image = np.array(image) / 255.0
    
    # Convert to CHW format (Channels x Height x Width)
    image = np.transpose(image, (2, 0, 1))
    
    # Add a batch dimension (1, Channels, Height, Width)
    image = np.expand_dims(image, axis=0).astype(np.float32)
    
    return image

def save_compressed_image(output, save_path):
    """
    Save the compressed output image from the model inference.
    """
    # Post-process the output (if needed)
    output = output.squeeze(0)  # Remove batch dimension
    
    # Convert to HWC format (Height x Width x Channels)
    output = np.transpose(output, (1, 2, 0))
    
    # Clip values to valid range (0, 1) and convert to 8-bit (0-255)
    output = np.clip(output, 0, 1) * 255.0
    output = output.astype(np.uint8)
    
    # Convert to image format and save
    output_image = Image.fromarray(output)
    output_image.save(save_path)

def run_inference(image_path, model_path, save_path):
    """
    Run inference on an input image using the exported ONNX model.
    """
    # Load the image and preprocess it
    input_image = preprocess_image(image_path)
    
    # Load the ONNX model

    session_options = onnxruntime.SessionOptions()
    session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    #session_options.log_severity_level = onnxruntime.logging.LoggingLevel.WARNING

    #Check if CUDA is available
    providers = [('CUDAExecutionProvider',{"use_tf32":1})] if 'CUDAExecutionProvider' in onnxruntime.get_available_providers() else ['CPUExecutionProvider']
    print("Using:",providers)
    onnx_session = onnxruntime.InferenceSession(model_path,sess_options=session_options, providers=providers)
    #onnx_session = onnxruntime.InferenceSession(model_path)

    #logging.info(f"Using execution provider(s) :{onnx_session.get_providers()}")
    
    # Run inference
    input_names = ["input"]
    output_names = ["output"]
    start=time.time()
    onnx_output = onnx_session.run(output_names, {input_names[0]: input_image})[0]
    end=time.time()

    print("Time in seconds for the inference is:",round(end-start,2),"seconds")

    
    # Save the output image
    save_compressed_image(onnx_output, save_path)
    print(f"Compressed image saved at: {save_path}")





def main():

    image_path=r"C:\Swapnil\Narrowband_DRONE\Image_compression_code\FLOPS\image.png"
    model_path=r"C:\Swapnil\Narrowband_DRONE\Image_compression_code\Method_7_compress_ai\compressai\examples\bmshj2018_factorized_models\bmshj_model_hd_ssim.onnx"
    save_path=r"C:\Swapnil\Narrowband_DRONE\Image_compression_code\FLOPS\image_py_onnx_export_HD_ssim.png"

    run_inference(image_path,model_path,save_path)
    


if __name__=="__main__":
    main()


