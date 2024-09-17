
Here the 

tensorrt_engine_infer.py is the complete file 
-load the onnx model
-generate the engine file or load from the disc
-runs the inference on the single image
-paths are hardcoded in the file and need to be changed in the script


trt_main.py is the main file which executes the same build and inference on the
dataset.

Run trt_main.py only and not the build_engine.py or trt_inference.py 