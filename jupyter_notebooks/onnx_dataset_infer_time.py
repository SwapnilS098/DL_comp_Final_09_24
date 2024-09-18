

#Need to run the script in the compressAI env
#also maybe works in the normal system

import os 
import io
import math
import torch 
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import time
from PIL import Image
import onnxruntime
import onnx

from onnxruntime_tools import optimizer
#from onnxruntime_tools.optimizer import get_fused_onnx_model
from onnxruntime_tools.transformers.onnx_model_bert import BertOnnxModel



def calculate_conv_flops(node, input_shape, output_shape):
    """Calculate FLOPs for a convolution layer"""
    flops_per_instance = 2 * np.prod(output_shape[2:]) * input_shape[1] * np.prod(node.attribute[0].ints)
    total_flops = flops_per_instance * output_shape[1]
    return total_flops

def calculate_matmul_flops(input_shape, output_shape):
    """Calculate FLOPs for a matrix multiplication"""
    return 2 * np.prod(input_shape) * output_shape[-1]

def calculate_flops_onnx(model_path):
    # Load the ONNX model
    model = onnx.load(model_path)

    # Initialize total FLOPs counter
    total_flops = 0

    # Iterate through nodes in the ONNX graph
    for node in model.graph.node:
        if node.op_type == 'Conv':
            # Get input and output shape for the convolution node
            input_shape = [dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim]
            output_shape = [dim.dim_value for dim in model.graph.output[0].type.tensor_type.shape.dim]
            # Calculate FLOPs for the convolution
            flops = calculate_conv_flops(node, input_shape, output_shape)
            total_flops += flops

        elif node.op_type == 'MatMul':
            # Get input and output shape for the matrix multiplication node
            input_shape = [dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim]
            output_shape = [dim.dim_value for dim in model.graph.output[0].type.tensor_type.shape.dim]
            # Calculate FLOPs for matrix multiplication
            flops = calculate_matmul_flops(input_shape, output_shape)
            total_flops += flops

    print(f"Total FLOPs: {total_flops / 1e9} GFLOPs")
    

def preprocess_gray(image_path):
    image=Image.open(image_path).convert("L")
    #resize the image
    image=image.resize((1640,1232))
    #convert to numpy and normalize
    image=np.array(image)/255.0
    
    #create an empty numpy array
    img_blank=np.zeros((3,1232,1640))

    #assign the gray image to the blank array
    img_blank[0]=image
    
    
    img_final=np.expand_dims(img_blank,axis=0).astype(np.float32)

   
    return img_final

def inference_setup(model_path):
    #load the ONNX model
    session_options=onnxruntime.SessionOptions()
    session_options.graph_optimization_level=onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

    #Check if CUDA is available
    providers = [('CUDAExecutionProvider',{"use_tf32":0})] if 'CUDAExecutionProvider' in onnxruntime.get_available_providers() else ['CPUExecutionProvider']
    print("Using:",providers)
    onnx_session = onnxruntime.InferenceSession(model_path,sess_options=session_options, providers=providers)

    input_names=["input"]
    output_names=["output"]

    return onnx_session,input_names,output_names

def run_infer(onnx_session,input_image,output_names,input_names):
    onnx_output = onnx_session.run(output_names, {input_names[0]: input_image})[0]
    return onnx_output


def save_compressed_image(output, save_path):
    """
    Save the compressed output image from the model inference.
    """
    # Post-process the output (if needed)
    output = output.squeeze(0)  # Remove batch dimension
    output=output[0]
    print("dimension",output.shape)
    
    # Convert to HWC format (Height x Width x Channels)
    #output = np.transpose(output, (1, 2, 0))
    
    # Clip values to valid range (0, 1) and convert to 8-bit (0-255)
    output = np.clip(output, 0, 1) * 255.0
    output = output.astype(np.uint8)
    
    # Convert to image format and save
    output_image = Image.fromarray(output)
    output_image=output_image.convert("L")
    output_image.save(save_path,quality=80) 


def export_to_buffer(output):

    #post process the output
    output=output.squeeze(0)

    #convert to the HWC format
    output=np.transpose(output,(1,2,0))

    #clip values to the valid range(0,1)
    output=np.clip(output,0,1)*255.0
    output=output.astype(np.uint8)

    #convert to PIL image and write the data to the in_memory buffer
    output_image=Image.fromarray(output)
    buffer_stream=io.BytesIO()
    output_image.save(buffer_stream,format="WEBP")

    return buffer_stream.getvalue()



def analyze_time(preprocess,infer,buff):
    preprocess=np.array(preprocess)
    infer=np.array(infer)
    buff=np.array(buff)
    print("Average time for infer per image:",np.mean(infer))
    print("Average time for preprocess per image:",np.mean(preprocess))
    print("Average time for export to buffer per image:",np.mean(buff))
    print("average FPS achievable is:",round(1/np.mean(infer),4),"At 1232x1640")



def image_handling(dataset_path):
    lst=os.listdir(dataset_path)
    images=[]
    for image in lst:
        if image.lower().endswith("jpg") or image.lower().endswith("jpeg") or image.lower().endswith("png") or image.lower().endswith("webp"):
            images.append(image)
    print("dataset has:",len(images),"images")

    return images

def export_image_from_buffer(buffer, save_path, quality=40):
      """
      Exports an image from a buffer to disk with a specified quality.

      Args:
        buffer: A bytes object containing the image data.
        filename: The filename for the exported image.
        quality: The desired image quality (0-100).
      """

      # Create an Image object from the buffer
      image = Image.frombytes("RGB", (1640,1232), buffer)

      # Save the image to disk with the specified quality
      image.save(save_path, format="JPEG", quality=quality)
      



def main(model_path,dataset_path):

    #get the images dataset
    images=image_handling(dataset_path)

    #Doing for the Gray only

    #setting up for the inference
    onnx_session,input_names,output_names=inference_setup(model_path)

    #calculate the FLOPS
    calculate_flops_onnx(model_path)


    preprocess_time=[]
    infer_time=[]
    buff_export_time=[]
    save_path=r"C:\Swapnil\Narrowband_DRONE\Image_compression_code\FLOPS\export_from_buffer.jpg"
    
    for image in images:
        image_path=os.path.join(dataset_path,image)
        #preprocess the image as grayscale
        start=time.time()
        input_image=preprocess_gray(image_path)
        #print("input image type:",type(input_image),"input_image:",input_image.shape)
        #print("image is like:",input_image[0])
        end=time.time()
        preprocess_time.append(round(end-start,2))
        print("preprocess time:",round(end-start,2))
        
        #run the inference
        start=time.time()
        onnx_output=run_infer(onnx_session,input_image,output_names,input_names)
        end=time.time()
        infer_time.append(round(end-start,2))
        print("infer time:",round(end-start,2))
        
        #export to the buffer
        start=time.time()
        #save_compressed_image(onnx_output, save_path)
        save_compressed_image(onnx_output, save_path)
        #buffer=export_to_buffer(onnx_output)
        end=time.time()
        buff_export_time.append(round(end-start,2))
        print("Done for image:",image)
        print("export time:",round(end-start,2))
        

    analyze_time(preprocess_time,infer_time,buff_export_time)
    

if __name__=="__main__":
    image_path=r"C:\Swapnil\Narrowband_DRONE\Image_compression_code\FLOPS\image.png"
    model_path=r"C:\Swapnil\Narrowband_DRONE\Image_compression_code\FLOPS\bmshj2018_factorized_models\bmshj_halfUHD_mse_8.onnx"
    save_path=r"C:\Swapnil\Narrowband_DRONE\Image_compression_code\FLOPS\export_from_buffer.jpg"
    dataset_path=r"C:\Swapnil\Narrowband_DRONE\atsugi_dataset_small_50\uncomp_images"
    export_path=r"C:\Swapnil\Narrowband_DRONE\Image_compression_code\FLOPS\ONNX_pytorch_dataset\onnx_compressed_images"

    main(model_path,dataset_path)
    

