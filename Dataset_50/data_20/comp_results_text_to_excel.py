

"""
    Simple script to read the text files from different directories and then
    make excel file from it
    """

import os
import numpy as  np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_directories(path):
    files=os.listdir(path)

    directories=[]
    for file in files:
        if '.' in file: #files with the extensions are ignored
            pass
        else:
            directories.append(file)
    print("directories are:",directories)
    return directories


def image_handling(path):
    """
    path to the directory containing the images;
    """
    files=os.listdir(path)
    images=[]
    
    for file in files:
        if file.lower().endswith('.png') or \
           file.lower().endswith('.jpg') or \
           file.lower().endswith('.jpeg') or \
           file.lower().endswith('.webp') or \
           file.lower().endswith('.avif'):
            images.append(file)
    #get the size of the images
    sizes=[]
    for image in images:
        image_path=os.path.join(path,image)
        size=os.path.getsize(image_path)
        sizes.append(size)
    if len(sizes)!=0:
        sizes=np.array(sizes)
        mean_size=round(sizes.mean()/1000,2)
        #print("Average file size is:",mean_size,"KB")
        return images,mean_size
    else:
        print("There are no images in the directory",path)
        return '',0
        
        
        

def main(path):

    directories=get_directories(path)
    #got all the directories
    #open the directory

    df_main=pd.DataFrame() #main dataframe to contain other dataframes

    #get the file size of the original dataset
    images,mean_size=image_handling(path)
    org_dataset_mean_size=mean_size
    
    for dir_ in directories:
        dir_path=os.path.join(path,dir_)

        images,mean_size=image_handling(dir_path)
        #get the images in the directory
        if images=='' and mean_size==0:
            continue
        
        #calculation of the compression ratio
        compression_ratio=round(org_dataset_mean_size/mean_size,2)
        
        text=get_text_files(dir_path) #name of the text file only\
        if text=='':
            continue
        #print(text)
        #get the list of the items
        text_path=os.path.join(dir_path,text)
        data=read_text(text_path)
        
        #modify the name of the model to more detail one
        data[0]=list(data[0])
        data[0][1]=text.split('.')[0]
        #add more fields to data
        data,avg_total_comp_time=add_more_fields_to_data(data)

        #add the compression ratio to the data
        data.append(["Compression Ratio",compression_ratio])
        data.append(["Avg_comp_img_size_KB",mean_size])

        #Calculate the FPS based on the Data limit
        #Max 5000000 i.e (5 Megabits)
        bandwidth_required=(1/avg_total_comp_time)*(mean_size)*(8)*(1000)
        bandwidth_required=bandwidth_required/1000000
        data.append(["Bandwidth_required_Mbps",bandwidth_required])
        
        #print(data)
        final_data=process_data(data)
        #print(final_data) # for verification
        #make the dataframe
        df=pd.DataFrame(final_data,index=[0])
        #print(df)
        df_main=pd.concat([df_main,df])
        

    single_param=quality_single_parameter(df_main)
    df_main["quality_single_param"]=single_param
    #exporting the dataframe to the excel file
    excel_path=os.path.join(path,"results.xlsx")

    #plotting the data
    plot_model_data(df_main)
    plot_3d(df_main)
    #df_main.to_excel(excel_path,index=True)
    print("Results exported to the disc in excel file")
    df_main.info()
    return df_main

def plot_model_data(df_main):

    model_name=df_main["Model Name"]
    bandwidth_required=df_main["Bandwidth_required_Mbps"]
    quality_param=df_main["quality_single_param"]

    #create a scatter plot
    plt.scatter(bandwidth_required,quality_param)
    plt.xlabel("bandwidth required in Mbps")
    plt.ylabel("quality single param")
    plt.title("DL model performance")

    
    
    #adding the data labels
    for i,row in df_main.iterrows():
        if row["Bandwidth_required_Mbps"] <=5:
            plt.text(row["Bandwidth_required_Mbps"],row["quality_single_param"],row["Model Name"],\
                     fontsize=12, horizontalalignment='center', verticalalignment='bottom')
    plt.show()


def calculate_best_model(bandwidth,quality,fps,models):

    """
    best model has the least bandwidth, highest quality and highest fps"""
    b_b=100
    q_b=0
    f_b=0
    model_b=''
    
    for model,b,q,f in zip(models,bandwidth,quality,fps):
        if b<b_b:
            b_b=b
        if q>q_b:
            q_b=q
        if f>f_b:
            f_b=f
    for model,b,q,f in zip(models,bandwidth,quality,fps):
        if b_b==b and q_b==q and f_b==f:
            model_b=model
    return b_b,q_b,f_b,model

def plot_3d(df_main):
    model_name=df_main["Model Name"]
    bandwidth_required=df_main["Bandwidth_required_Mbps"]
    quality_param=df_main["quality_single_param"]
    fps=df_main["FPS_time_based"]

    #find out the best model
    b_b,q_b,f_b,model=calculate_best_model(bandwidth_required,quality_param,fps,model_name)

    x=bandwidth_required
    y=quality_param
    z=fps

    # Create the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot with labels
    
    colors = ['red' if val > 5 else 'blue' for val in x]  # Use list comprehension for color assignment
    
    
    scatter = ax.scatter(x, y, z,c=colors, label='Data Points')

    #add the best solution
    print(b_b,q_b,f_b)
    count=0
    for model,b,q,f in zip(model_name,bandwidth_required,quality_param,fps):
        #ax.text(b_b, q_b, f_b, str(model), fontsize=12, color='black')
        if count==0:
            ax.text(b, q, f, str(model), fontsize=8, horizontalalignment='left',color='green')
            count=1
        elif count==1:
            ax.text(b, q, f, str(model), fontsize=8, horizontalalignment='right',color='black')
            count=0

    # Add labels and title
    ax.set_xlabel('Bandwidth required in Mbps')
    ax.set_ylabel('Quality Parameter (Higher better)')
    ax.set_zlabel('FPS delivered')
    plt.title('DL Model performance')

    # Add legend
    plt.legend()

# Show the plot
    plt.show()
    
    

def normalize_numpy_array(array):
    min_=np.min(array)
    max_=np.max(array)
    normalized=(array-min_)/(max_-min_)
    return normalized

def quality_single_parameter(df_main):
    """
    calculate a single parameter for
    combining the psnr, ssim, FPS
    """
    ssim=np.array(df_main["Avg. SSIM"])
    
    
    psnr=np.array(df_main["Avg. PSNR"])
    psnr_n=normalize_numpy_array(psnr)
    
    
    fps=np.array(df_main["FPS_time_based"])
    fps_n=normalize_numpy_array(fps)
    
    
    comp_ratio=np.array(df_main["Compression Ratio"])
    comp_n=normalize_numpy_array(comp_ratio)
    

    #arrays=[ssim,psnr_n,fps_n,comp_n]
    
    weights=[0.25,0.25,0.25,0.25]
    w_avg_vector=[]
    #giving equal weights to all
    #weighted_avg = np.sum([array*weight for array,weight in zip(arrays,weights)])#\
    for i in range(len(ssim)):
        w_avg=ssim[i]*weights[0]+psnr_n[i]*weights[1]+fps_n[i]*weights[2]+comp_n[i]*weights[3]
        w_avg_vector.append(round(w_avg,3))
        

    return w_avg_vector
    

def process_data(data):
    """
    processing  the data to make the dictionary
    """
    #print("data is:",data)
    final_data={}
    for item in data:
        #print(item,item[0],item[1])
        try:
            final_data[item[0]]=round(float(item[1]),2)
        except:
            final_data[item[0]]=item[1]
    #print(final_data)
    return final_data

def add_more_fields_to_data(data):
    comp_time=float(data[1][1])
    pre_process_time=float(data[4][1])
    avg_total_comp_time=comp_time+pre_process_time
    data.append(['avg_total_comp_time',avg_total_comp_time])
    FPS_time=round(1/avg_total_comp_time,2)
    data.append(['FPS_time_based',FPS_time])
    return data,avg_total_comp_time

        #Now read the text file and make the numpy arr
def get_text_files(path):
    lst=os.listdir(path)
    text=[]
    for file in lst:
        if file.lower().endswith('.txt'):
            text.append(file)

    if len(text)==0:
        return ''
    else:
        return text[0]

def read_text(file_path):
    """get the file path read the data and make the list from it"""
    data=[]
    with open(file_path,"r") as f:
        for line in f:
            key,value=line.strip().split(":")
            data.append((key,value))
    return data

if __name__=="__main__":
    path=r"C:\Swapnil\Narrowband_DRONE\Image_compression_code\Final_DL_compession_September_24\Dataset_50\data_20"
    df=main(path)
    
