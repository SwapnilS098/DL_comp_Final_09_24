
"""
    -Evaluation of the reconstructed/compressed image quality
    -Image quality parameters such as SSIM, PSNR, LPIPS are evaluated
    -compression ratio is calculated

    Input:
        -Reconstructed images dataset
        -Original images dataset

    Output:
        -Text file containing the individual values
        -Dataset average values
"""

import os
import torch
from PIL import Image

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from torchvision import transforms
import lpips
loss_fn_alex=lpips.LPIPS(net="alex")
loss_fn_vgg=lpips.LPIPS(net="vgg")

class Dataset_eval:

    def __init__(self,org_dataset_path,recon_dataset_path):
        self.org_dataset=org_dataset_path
        self.recon_dataset=recon_dataset_path

    def images_handling(self,path):
        """
        path to the images dataset
        """

        files=os.listdir(path) #files in the directory
        images=[] #getting only the jpg and png images
        for file in files:
            if file.lower().endswith(".jpeg") or file.lower().endswith(".jpg") \
            or file.lower().endswith(".png") or file.lower().endswith(".webp") \
            or file.lower().endswith(".avif"):
                images.append(file)
        
        print("Images in dataset:",len(images))
        if len(images)!=0:
            img_format=images[0].split('.')[1]
            print("File_type:",img_format)
        else:
            print("No images in the path or different format")
        
        return images, path, img_format

    def convert_gray_3_channel(self,rgb_image):
        """
        Expects rgb_image as pillow object
        Returns the numpy array of 1 channel gray data and other 2 as 0
        """
        #convert the rgb to gray
        img=rgb_image.convert("L")
        #convert to numpy array
        img_array=np.array(img)
        #get the shape
        shape=img_array.shape
        #new blank 3 channel array
        new_array=np.zeros((3,shape[0],shape[1]))
        #assign the gray part to first channel
        new_array[0]=img_array
        return new_array

    def get_image_details(self,image):
        """
        input the image as pillow image
        """
        shape=list(image.shape) #check the shape
        res=shape[1:] #resolution
        channel=shape[0] #channel
        return shape,res,channel
        

    def check_and_match_dataset(self,gray_3_channel):
        """
        #Assumption: All the images in the dataset will have the same characteristics
        
        -checks the images in the org_dataset shape(color gray), resolution, type (PNG etc)
        -checks the images in the recon_dataset
        -call the resize function, color_conversion option to modify the org image according to
        recon image

        gray_3_channel: is boolean if True convert_gray_3_channel will be called
        
        """

        #first for the original images dataset
        images_org,org_path,img_format_org=self.file_handling(self.org_dataset)
        
        image_org=Image.open(os.path.join(org_path,images_org[0])) #using Pillow
        shape_org,res_org,channel_org=self.get_image_details(image_org)
    

        #now for the recon images dataset
        images_recon,recon_path,img_format_recon=self.file_handling(self.recon_dataset)

        image_recon=Image.open(os.path.join(recon_path,images_org[0])) #using Pillow
        shape_recon,res_recon,channel_recon=self.get_image_details(image_recon)
        

        
        #Now checking if the channels match?
        #check once and do for the whole dataset
        convert_gray_all=False
        
        if channel_org==channel_recon and gray_3_channel==False:
            #means both are gray or both are rgb and no need for the change
            pass
        elif channel_org==channel_recon and channel_org==3 and gray_3_channel==True:
            #both are rgb and need to change the rgb to 3 channel gray version 
            #then convert the org image to 3channel gray version
            convert_gray_all=True

        #check the resolution
            change_res_all=False
            if res_recon==res_org:
                pass
            else:
                change_res_all=True

        return change_res_all,convert_gray_all,
        
    def match_dataset(self,images_org,images_recon,):
        """
        pass
        """

        for image_org,image_recon in zip(images_org,image_recon):

            #open the image and read as PIL version
            img_org=Image.fromarray(self.convert_gray_3_channel(img_org)) #convert the numpy array to pillow image
            
        
            

    def compression_ratio(self,img_1_path,img_2_path):
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

    def calculate_ssim(self,img_org,img_recon):

        #converting the image to the numpy array
        img_recon=np.asarray(img_recon)
        img_org=np.asarray(img_org)
        
        ssim_value=ssim(img_org,img_recon)
        return ssim_value

    def calculate_lpips(self,img_org,img_recon):
        """
        img_org and img_recon are the path of the 
        original image and the reconstructed image
        """

        #converting the image to the numpy array
        img_recon=np.asarray(img_recon)
        img_org=np.asarray(img_org)

        
        transform=transforms.ToTensor() #convert the image to Tensor 
        #print(img_org.shape,img_recon.shape)

        #converting to the tensors
        img_org=transform(img_org)
        img_recon=transform(img_recon)

        #calculate the loss value
        lpips=loss_fn_alex.forward(img_org,img_recon)
        lpips=lpips.detach().numpy() #to cpu then 
        #print("LPIPS:",lpips) 
        return lpips[0][0][0][0]

    def calculate_psnr(self,img_org,img_recon):

        """
        img_org and img_recon are the path of the 
        original image and the reconstructed image
        """

        #converting the image to the numpy array
        img_recon=np.asarray(img_recon)
        img_org=np.asarray(img_org)

        psnr_value=psnr(img_org,img_recon)
        #print("PSNR:",psnr_value)
        return psnr_value

        

if __name__=="__main__":
    org_dataset_path="pass"
    recon_dataset_path="pass"
    eval_=Dataset_eval(org_dataset_path,recon_dataset_path)
