**UDACITY PROJECT WORKSPACE RUN COMMANDS**

usage: main.py [-h] -m MODEL -i INPUT [-l CPU_EXTENSION] [-d DEVICE] [-pt PROB_THRESHOLD] [-pc PERF_COUNTS] [-mc MAX_PERSON_COUNT]

**Intel Pretrained Model**

python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m models/tensorflow/IR/ssd_mobilenet_v2_coco_2018_03_29/FP32/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm



**Caffe Converted Model (see write-up.md for details)**

python main.py -i resources/people-detection.mp4 -m models/caffe/IR/MobileNetSSD_deploy.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm



**Tensorflow Converted Model**

python main.py -i resources/people-detection.mp4 -m models/tensorflow/IR/ssd_mobilenet_v2_coco_2018_03_29/{FP32,FP16}/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

