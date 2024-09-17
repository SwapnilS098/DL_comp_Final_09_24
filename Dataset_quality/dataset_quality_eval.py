#importing the modules for the SIFT features part
import cv2
import os
import numpy as np
import time
from numpy import polyfit
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from PIL import Image
import math
import torch
from torchvision import transforms
from torchvision.utils import save_image

from pytorch_msssim import ms_ssim #metric for the image
#another image reconstruction quality metric
from skimage.metrics import structural_similarity as ssim 
#PSNR metric for image reconstruction
from skimage.metrics import peak_signal_noise_ratio as psnr
import pandas as pd

import lpips #already installed with pip install lpips 
#used for the lpips metric
loss_fn_alex=lpips.LPIPS(net="alex")
loss_fn_vgg=lpips.LPIPS(net="vgg") 



def compression_ratio(img_1_path,img_2_path):
    """
        Assuming the first image is uncompressed version
    """
    print(img_1_path)
    try:
        size_1=os.path.getsize(img_1_path) #getting the size of the image on the disc
    except:
        print("img_1 path not valid")

    try:
        size_2=os.path.getsize(img_2_path) #getting the size of the image on the disc
    except:
        print("img_2 path not valid")

    ratio=size_1/size_2
    #print("Compression ratio is:",ratio)
    return ratio


def calculate_ssim(img_org,img_recon):
    """
    img_org and img_recon are the path of the 
    original image and the reconstructed image
    """
##    if crop_image(img_org,img_recon) is None:
##        img_recon=np.asarray(Image.open(img_recon))
##        img_org=np.asarray(Image.open(img_org))
##    else:
##        img_recon=np.asarray(crop_image(img_org,img_recon))
##        img_org=np.asarray(Image.open(img_org))

    img_recon=np.asarray(Image.open(img_recon).convert("L"))
    img_org=np.asarray(Image.open(img_org))
        
    ssim_value=ssim(img_org,img_recon,win_size=3) #for 1 channel
    #print("SSIM:",ssim_value)
    return ssim_value


def calculate_lpips(img_org,img_recon):
    """
    img_org and img_recon are the path of the 
    original image and the reconstructed image
    """
##    if crop_image(img_org,img_recon) is None:
##        img_recon=np.asarray(Image.open(img_recon))
##        img_org=np.asarray(Image.open(img_org))
##    else:
##        img_recon=np.asarray(crop_image(img_org,img_recon))
##        img_org=np.asarray(Image.open(img_org))

    

    img_recon=np.asarray(Image.open(img_recon).convert("L"))
    img_org=np.asarray(Image.open(img_org))

    print("img_org:",img_org.shape)
    print("img_recon:",img_recon.shape)
    
    transform=transforms.ToTensor() #convert the image to Tensor 
    #print(img_org.shape,img_recon.shape)
    img_org=transform(img_org)
    img_recon=transform(img_recon)
    
    lpips=loss_fn_alex.forward(img_org,img_recon)
    lpips=lpips.detach().numpy() #to cpu then 
    #print("LPIPS:",lpips) 
    return lpips[0][0][0][0]


def calculate_psnr(img_org,img_recon):

    """
    img_org and img_recon are the path of the 
    original image and the reconstructed image
    """
    
##    if crop_image(img_org,img_recon) is None:
##        img_recon=np.asarray(Image.open(img_recon))
##        img_org=np.asarray(Image.open(img_org))
##    else:
##        img_recon=np.asarray(crop_image(img_org,img_recon))
##        img_org=np.asarray(Image.open(img_org))

    img_recon=np.asarray(Image.open(img_recon).convert("L"))
    img_org=np.asarray(Image.open(img_org))

    psnr_value=psnr(img_org,img_recon)
    #print("PSNR:",psnr_value)
    return psnr_value


#Utility function for writing the dictionary to the file

def write_dct_file(dct,filename):
    with open(filename,'w') as f:
        for metric, value in dct.items():
            f.write(f"{metric}:{value}\n")

#SIFT FEATURES

def sift(image):
    '''
    Function can draw keypoints on the image too, 
    but calculation is commented.
    '''

    if image.endswith(".avif"):
        image=Image.open(image).convert("L")
        image=np.array(image)
    else:       
        image=cv2.imread(image,cv2.IMREAD_GRAYSCALE)
    
    sift=cv2.SIFT.create() #initiate SIFT detector

    #Find keypoints on the image
    keypoints,descriptors=sift.detectAndCompute(image,None)

    #Draw keypoints on the image
    #image_with_sift=cv2.drawKeyPoints(image.copy(),keypoints,None,(255,0,0),4)

    return keypoints#,image_with_sift

def file_handling(path):
    """
    path to the images dataset
    """
    #org_dir=os.getcwd()

    #changing to the dataset dir
    try:
        os.chdir(path)
    except:
        print("Path not valid")
        return

    files=os.listdir(path) #files in the directory
    images=[] #getting only the jpg and png images
    for file in files:
        if file.lower().endswith(".jpeg") or file.lower().endswith(".jpg") \
        or file.lower().endswith(".png") or file.lower().endswith(".webp") \
        or file.lower().endswith(".avif"):
            images.append(file)

    print("Images in dataset:",len(images))
    if len(images)!=0:
        print("First filename:",images[0])
    else:
        print("No images in the path or different format")
    
    #go to the original directory back
    #os.chdir(org_dir)
    
    return images, path

def crop_image(img_org_path,img_recon_path):
    """
        top left X and Y
        bottom right X and Y is the crop box
    """

    img_recon=Image.open(img_recon_path)
    img_org=Image.open(img_org_path)

    output_width, output_height =img_recon.size
    original_width,original_height=img_org.size

    #checking the dimensions of the images
    if original_width==output_width and \
    original_height==output_height:
        return None
    else:
        crop_box = (0, 0, output_width, original_height)
        cropped_image = img_recon.crop(crop_box)
        #print(cropped_image.size)
        return cropped_image


#calculate the metrics for the batch
def batch_metrics(org_path,recon_path):
    """
    Input: Path of the original dataset
    Output: Dataframe of reconstruction quality
            metrics and compression ratio for each
            image in dataset
    """

    print("Runnning Batch metrics")
    
    images,path=file_handling(org_path)
    images_r,path_r=file_handling(recon_path)
        

    #initialize the metrics
    ssim=[]
    psnr=[]
    lpips=[]
    cr=[]
    org_size=[]
    recon_size=[]
    sift_features_o=[]
    sift_features_r=[]
    sift_features_per_red=[]

    for img_o,img_r in zip(images,images_r):

        print("Doing for:",img_o)
        

        img_o=os.path.join(path,img_o)
        img_r=os.path.join(path_r,img_r)

        ratio=compression_ratio(img_o,img_r)
        cr.append(round(ratio,2))

        lpips_=calculate_lpips(img_o,img_r)
        lpips.append(lpips_)
    
    
        ssim_=calculate_ssim(img_o,img_r)
        ssim.append(ssim_)

        
        psnr_=calculate_psnr(img_o,img_r)
        psnr.append(psnr_)

        #sizes are in KB
        org_size.append(os.path.getsize(os.path.join(path,img_o))/1000)
        recon_size.append(os.path.getsize(os.path.join(path_r,img_r))/1000)

        features_o=len(sift(img_o)) #returns the list , we need count of those features
        sift_features_o.append(features_o)

        features_r=len(sift(img_r)) #returns the list , we need count of those features
        sift_features_r.append(features_r)

        sift_features_per_red_=(features_o-features_r)/(features_o)*100
        sift_features_per_red.append(sift_features_per_red_)
        
        

        
        

    metrics={"SSIM":ssim,
             "PSNR":psnr,
             "LPIPS":lpips,
             "CR":cr,
            "ORG_SIZE":org_size,
            "RECON_SIZE":recon_size,
            "ORG_SIFT":sift_features_o,
            "RECON_SIFT":sift_features_r,
            "SIFT_PER_RED":sift_features_per_red}

    #make a dataframe from the metrics for exporting
    df=pd.DataFrame(metrics)
    return df



def average_batch_metrics(df,filename):
    """
    Gives the average of each of the metric 
    for the whole batch
    """
    cols=df.columns
    avg_metric={}

    #number of images
    avg_metric["Dataset_size"]=len(df[cols[0]])
    for metric in cols:
        metric_array=np.array(df[metric]) #convert the each column of df to np array
        avg_metric[metric]=round(np.average(metric_array),2)
    
    write_dct_file(avg_metric,filename)
    print("\n Exported the text form of the summary\n")
    print("average_metrics:",avg_metric)
    return avg_metric



def make_gray_dataset(org_path):
    """
    Since the original path has the RGB images and the
    reconstructed is in the gray form
    To allow proper comparison of the data
    We generate a corresponding gray dataset for the original images dataset
    """
    images, path=file_handling(org_path)

    gray_export_path=os.path.join(org_path,"org_gray_dataset")
    print("gray export path",gray_export_path)

    if not os.path.exists(gray_export_path):
        os.makedirs(gray_export_path)

    for image in images:
        img_path=os.path.join(path,image)
        img=Image.open(img_path).convert("L")
        final_path=os.path.join(gray_export_path,image)
        img.save(final_path,quality=95)
        print("done for image:",image)
    return gray_export_path


def resize_dataset(gray_export_path,resize_shape):
    """
    Since our application of the image compression
    we are resizing the image. Hence this function will
    resize the exported gray dataset"""


    images, path=file_handling(gray_export_path)

    resize_export_path=os.path.join(org_path,"org_gray_dataset")
    print("resize export path",resize_export_path)

    if not os.path.exists(resize_export_path):
        os.makedirs(resize_export_path)

    for image in images:
        img_path=os.path.join(path,image)
        img=Image.open(img_path).resize(resize_shape)
        final_path=os.path.join(resize_export_path,image)
        img.save(final_path,quality=95)
        print("done for image:",image)
    return resize_export_path

    

if __name__=="__main__":
    #org_path=r"C:\Swapnil\Narrowband_DRONE\Image_compression_code\Final_DL_compession_September_24\Dataset_50"
    org_path=r"C:\Swapnil\Narrowband_DRONE\Image_compression_code\Final_DL_compession_September_24\Dataset_50\org_gray_dataset"
    recon_path=r"C:\Swapnil\Narrowband_DRONE\Image_compression_code\Final_DL_compession_September_24\onnx_scripts\onnx_infer\compressed_dataset"
    #org_path=r"C:\Swapnil\Narrowband_DRONE\Image_compression_code\Final_DL_compession_September_24\Dataset_50\org_gray_dataset"
    
    recon_gray=False #this is a flag if the reconstructed path has the grayscale images and the org has RGB

    #grayscale conversion and resizing the original images dataset
    if recon_gray==True:
        print("Creating the corresponding gray dataset")
        recon_path=make_gray_dataset(org_path)

        resize_shape=(1648,1232)
        resize_export_path=resize_dataset(recon_path,resize_shape)
        org_path=resize_export_path

    

    print("Now org_path:",org_path)
    print("Now recon_path:",recon_path)
    df=batch_metrics(org_path,recon_path)
    avg_metric=average_batch_metrics(df,"ONNX_summary.txt")
