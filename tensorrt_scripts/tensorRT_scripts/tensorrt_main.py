


"""
Run this script only.

No special environment is needed for running the thiscript
Installation of the tensorRT 10.3 wheels (tensorrt, tensorrt_lean,tensorrt_dispatch from the directory downloaded from internet)

This script has hardcoded paths and runs the inference on the small airpeak FPV dataset of PNG images of size 50

This script will
    -import other script for building the engine if not present on disc or load the engine from the disc
    -run the inference from the engine file
    -preprocess the image and postprocess and save the image to the disc
    
"""


import tensorrt as trt
print(trt.__version__)
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
#import torch
from PIL import Image
import time
import cv2
import os



from build_engine import Build_engine
print("Build_engine class is imported")


from trt_inference import TensorRTInference
print("TensorRTInference class is imported")





def trt_main(onnx_model_name,onnx_path_base,engine_path_base,input_shape,output_shape,dataset_path,export_data_path,engine_name):

    #final onnx model path
    onnx_path=os.path.join(onnx_path_base,onnx_model_name+".onnx")

    #final engine model path
    if engine_name=="same":
        engine_path=os.path.join(engine_path_base,onnx_model_name+".engine")
    else:
        engine_name=onnx_model_name+"_"+engine_name
        engine_path=os.path.join(engine_path_base,engine_name+".engine")
    print("engine path is:",engine_path)

    #loading or building the engine
    start=time.time()
    Engine=Build_engine(onnx_path,engine_path,input_shape)

    #check if the engine exists on the path
    if os.path.exists(engine_path):
        engine=Engine.load_engine()
        print("Engine is loaded from the disc")
    else:
        print("Engine not found at the path, Building the engine")
        start=time.time()
        engine=Engine.build_engine()
        end=time.time()
        print("engine is built, Time:",round(end-start,2),"seconds")
        Engine.save_engine(engine)
        print("Engine is exported to the disc")
    print("====================Engine done=============================")
    end=time.time()
    print("Engine time:",round(end-start,2),"seconds")

    #Now running the inference on the whole dataset
  
    #instantiating the TensorRTInference class
    trt_inference=TensorRTInference(engine_path,dataset_path,export_data_path,input_shape,output_shape,gray) #output_path is the image output path

    trt_inference.inference_over_dataset()
    


if __name__=="__main__":

    onnx_model_name="bmshj_halfUHD_mse_4" #write the name without the extension
    engine_name="same"   #if some optimization parameter is added then can write here
                         #it will append this name to the onnx_model name while exporting
    input_shape=[3,1232,1640]
    output_shape=[3,1232,1648]  #take care of the 8 pixels added by the model in the output shape
    gray=False
    
    onnx_path_base=r"C:\Swapnil\Narrowband_DRONE\Image_compression_code\Final_DL_compession_September_24\onnx_scripts\onnx_export\onnx_models"
    engine_path_base=r"C:\Swapnil\Narrowband_DRONE\Image_compression_code\Final_DL_compession_September_24\tensorrt_scripts\engine_models"
    dataset_path=r"C:\Swapnil\Narrowband_DRONE\Image_compression_code\Final_DL_compession_September_24\Dataset_50"
    export_data_path=r"C:\Swapnil\Narrowband_DRONE\Image_compression_code\Final_DL_compession_September_24\tensorrt_scripts\trt_output_data"


    trt_main(onnx_model_name,
             onnx_path_base,
             engine_path_base,
             input_shape,output_shape,
             dataset_path,
             export_data_path,
             engine_name)

