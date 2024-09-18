
"""
Program to import the files and compress them using different deep learning
image compression models.

For bpp: using average bits per pixel using
 avg bpp=size on disc /(resolution)

"""

import os
from PIL import Image
import compressor 


class compress_data:

    def __init__(self,path):
        """
        reuires the path where the images or the image is present
        """
        self.path=path

    def file_details(self):

        try:
            os.chdir(path)
            files=os.listdir()
            #print("Files:",files)
        except:
            print("Problem in path --exiting")
            exit()
        return files

    def get_images(self,files):

        #needs to update the file types list
        img_types=[".png",".jpg","jpeg",".pgm"]
        images=[]
        for file in files:
            for name in img_types:
                if name in file:
                    images.append(file)
        print("Number of images in the directory are:",len(images),"First file:",images[0],\
              "last file is:",images[-1])
        return images

    def get_image_details(self,img):

        """
        takes the image file path and returns the
        dictionary of the image details
        """
        img_path=self.path+"\\"+img
        
        details={}
        
        try:
            image=Image.open(img_path)
            details["Name"]=img

            width,height=image.size
            details["width"]=width
            details["height"]=height
            

            color_schema=image.mode
            details["mode"]=color_schema #RGB, L: grascale,P: indexed pallete , 1: binary, I integer image, F: float image

            details["size_disc"]=os.path.getsize(img_path) #in bytes

            details["avg_bpp"]=(details["size_disc"]/(width*height))*8 #in bits per pixel

            return details
        
        except OSError as err:
            print("Error opening the file")
            return None

def compress_single(path):
    compressor.compress_main(path)

        
    

if __name__=="__main__":
    #path=os.getcwd()
    path=r"C:\Swapnil\Narrowband_DRONE\Datasets\3_fpv_direct_20m_1k_png"
    #data=compress_data(path)
    #files=data.file_details()
    #images=data.get_images(files)
    #details=data.get_image_details(images[0])
    #print(details)
    output_path=r"C:\Swapnil\Narrowband_DRONE\Datasets\3_fpv_direct_20m_1k_png\compress"
    compress_single(path)
    data=compress_data(output_path)
    
    
    files=data.file_details()
    print(files)
    images=data.get_images(files)
    details=data.get_image_details(images[0])
    print(details)
