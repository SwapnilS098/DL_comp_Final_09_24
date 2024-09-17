

#Need to run the script in the compressAI env

"""
    CompressAI environment is required to run
    Dataset of S1 FPV 50 images is used for running the inference
    Average inference time per image is calculated from it
"""



    

if __name__=="__main__":
    image_path=r"C:\Swapnil\Narrowband_DRONE\Image_compression_code\FLOPS\image.png"
    model_path=r"C:\Swapnil\Narrowband_DRONE\Image_compression_code\Final_DL_compession_September_24\onnx_scripts\onnx_export\onnx_models\bmshj_halfUHD_mse_8.onnx"
    save_path=r"C:\Swapnil\Narrowband_DRONE\Image_compression_code\Final_DL_compession_September_24\onnx_scripts\onnx_infer\compressed_dataset"
    dataset_path=r"C:\Swapnil\Narrowband_DRONE\Image_compression_code\Final_DL_compession_September_24\Dataset_50"
    export_path=r"C:\Swapnil\Narrowband_DRONE\Image_compression_code\Final_DL_compession_September_24\onnx_scripts\onnx_infer\compressed_dataset"


    

