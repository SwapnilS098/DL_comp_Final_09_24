"""
    Script to get the quality parameters of the
    trt output and the org image
    """

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import os
import numpy as np
from PIL import Image

path_org=r"C:\Swapnil\Narrowband_DRONE\Image_compression_code\Final_DL_compession_September_24\tensorrt_scripts\trt_output_data\trt_result_metric\000000_left_org.png"
path_recon=r"C:\Swapnil\Narrowband_DRONE\Image_compression_code\Final_DL_compession_September_24\tensorrt_scripts\trt_output_data\trt_result_metric\000000_left_trt.jpg"

img_org=Image.open(path_org)
img_recon=Image.open(path_recon)

#change the org_image resolution
img_org=img_org.resize((1648,1232))


img_org=np.array(img_org)
img_recon=np.array(img_recon)

print("org-shape:",img_org.shape)
print("recon-shape:",img_recon.shape)



compression_ratio=os.path.getsize(path_org)/os.path.getsize(path_recon)
print("compression ratio:",compression_ratio)


ssim_=ssim(img_org,img_recon,win_size=3)
print("SSIM:",ssim_)



psnr_=psnr(img_org,img_recon)
print("SSIM:",psnr_)
