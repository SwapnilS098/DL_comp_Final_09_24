
#----------------------------------------------------#
import math
import os
import io
import torch
from torchstat import stat
from torchvision import transforms
import numpy as np
from torch import profiler

from PIL import Image

import matplotlib.pyplot as plt
from pytorch_msssim import ms_ssim
from compressai.zoo import bmshj2018_factorized
from ipywidgets import interact, widgets
#----------------------------------------------------#
device="cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
#----------------------------------------------------#

global model

def import_model(quality):
    
    #loading the pretrained model
    net=bmshj2018_factorized(quality=quality,pretrained=True).eval().to(device)
    print(f"parameters: {sum(p.numel() for p in net.parameters())}")

    global model
    model=net


def run_model(img):
    """
    function to reconstruct the image
    input: image and the quality of the model which is integer
    output: output tensor of the model and reconstructed image
    
    """
    global model
    net=model

    #running the model
    with profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,\
                                      torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
        with torch.no_grad():
            out_net=net.forward(img)
    print(prof.key_averages())
    with open('key_averages.txt','w') as f:
        print(prof.key_averages(),file=f)
    print(os.getcwd())
    print("exported")
    prof.export_chrome_trace("trace.json")
    out_net['x_hat'].clamp(0,1)
    #print(out_net.keys())
    print("------------------iamge type:",type(img))
    
    input_size=list(tuple(img.shape))
    input_size=input_size[1:]
    print("----shape----",input_size)
    net=net.to('cpu')
    stat(net,input_size=input_size)
    net=net.to('cuda')
    recon_image=transforms.ToPILImage()(out_net['x_hat'].squeeze().cpu())

    return out_net,recon_image

def visualize_results(img, recon_img):

    """
    function to visualize the original and the reconstructed image
    side by side
    Input: image in the PIL format and reconstructed image in the PIL transformed format
    Output: Nothing
    """
   
    fix, axes = plt.subplots(1, 3, figsize=(16, 12))
    for ax in axes:
        ax.axis('off')
        
    axes[0].imshow(img)
    axes[0].title.set_text('Original')

    axes[1].imshow(recon_img)
    axes[1].title.set_text('Reconstructed')

    #axes[2].imshow(diff, cmap='viridis')
    #axes[2].title.set_text('Difference')

    plt.show()


                    

def load_image(path):
    """
    function to load the image and return the image after the
    transformations
    input: path of the image on the disc
    output: image in PIL format and image in the PyTorch format
    """
    img=Image.open(path).convert('RGB')
    x=transforms.ToTensor()(img).unsqueeze(0).to(device)

    #show the image
    #plt.figure(figsize=(12,9))
    #plt.axis('off')
    #plt.imshow(img)
    #plt.show()
    return x,img


def export_image_webp(recon_img,filename,quality=1):
    """
    function to export the PIL imported image
    into the WebP format
    input: reconstructed image in PIL format filename and quality whose default parameter is 1
    output: nothing
    """
    filename=filename[:-4]
    filename+=".webp"
    #print(filename)
    #recon_img.save(filename,"WEBP",quality=quality)
    try:
        recon_img.save(filename,"WEBP",quality=quality)
    except:
        print("The recon image has some problem , maybe not of the PIL type")
        print("Export failed")


def export_image_jpeg(recon_img,filename,quality=1):
    """
    function to export the PIL imported image
    into the WebP format
    input: reconstructed image in PIL format filename and quality whose default parameter is 1
    output: nothing
    """
    filename=filename[:-4]
    filename+=".jpg"
    #print(filename)
    #recon_img.save(filename,"WEBP",quality=quality)
    try:
        recon_img.save(filename,"JPEG",quality=50)
    except:
        print("The recon image has some problem , maybe not of the PIL type")
        print("Export failed")


    

def compress_dataset(path,quality=1):
    """
    function to compress the dataset and export in the webp format
    
    """
    os.chdir(path)
    print("We are in:",path)
    print()
    files=os.listdir()
    #expecting only .jpg or .png files
    images=[]
    for file in files:
        if ".jpg" in file or ".JPG" in file or ".png" in file or ".PNG" in file:
            images.append(file)
    if len(images)!=0:
        print("Number of images:",len(images),"image type:",images[0][-4:])
    else:
        print("Empty directory or no images of the desired format")

    #make a new directory
    output_path=path+"\\compressed"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        print("Directory with the name already exists")
    
    #import the model
    import_model(quality)
    #run through all the images list
    for image in images:

        #load the image
        img_path=path+"\\"+image
        
        print(img_path)
        x,img=load_image(img_path)

        #run the model
        out_net,recon_image=run_model(x)

        #export to the disk
        export_path=output_path+"\\"+image[:-4]+"_c"+image[-4:]
        #export_path="C:\\Swapnil\\Narrowband_DRONE"+"\\"+image[:-4]+"_c"+image[-4:]
        print(export_path)
        export_image_jpeg(recon_image,export_path,quality=50)

    #visualize result at the end of single image
    #visualize_results(img, recon_image)

        

        
        
    
    

    


    


if __name__=="__main__":
    path=r"C:\Swapnil\Narrowband_DRONE\small_data_10"
    compress_dataset(path,quality=1)
