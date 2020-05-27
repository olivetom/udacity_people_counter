source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5


# REGION___________________________ PRETRAINED INTEL MODEL
# run pretrained intel model FP32. Person count should be 6. Returned 6.
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m models/pretrained/intel/person-detection-retail-0013/FP32/person-detection-retail-0013.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU --max_person_count 2 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

# run pretrained intel model FP16. Person count should be 6. Returned 6.
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m models/pretrained/intel/person-detection-retail-0013/FP16/person-detection-retail-0013.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU --max_person_count 2 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

# run pretrained intel model INT8. Person count should be 6. Returned 6.
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m models/pretrained/intel/person-detection-retail-0013/INT8/person-detection-retail-0013.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU --max_person_count 2 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

# run pretrained intel model FP32 on different video. Person count should be 7. Returned 7
python main.py -i resources/people-detection.mp4 -m /home/workspace/models/pretrained/intel/person-detection-retail-0013/FP32/person-detection-retail-0013.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU --max_person_count 2 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

# run pretrained intel model FP16 on different video. Person count should be 7. Returned 7
python main.py -i resources/people-detection.mp4 -m /home/workspace/models/pretrained/intel/person-detection-retail-0013/FP16/person-detection-retail-0013.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU --max_person_count 2 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

# run pretrained intel model INT8 on different video. Person count should be 7. Returned 7
python main.py -i resources/people-detection.mp4 -m /home/workspace/models/pretrained/intel/person-detection-retail-0013/INT8/person-detection-retail-0013.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU --max_person_count 2 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

#________________________________




# REGION___________________________ CONVERTED TENSORFLOW MODEL
# run converted/optimized tensorflow model. Person count should be 6. But returned 14.
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m models/tensorflow/IR/ssd_mobilenet_v2_coco_2018_03_29/FP32/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU --max_person_count 2  | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

# run converted/optimized tensorflow model. Person count should be 6. But returned 13
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m models/tensorflow/IR/ssd_mobilenet_v2_coco_2018_03_29/FP16/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU --max_person_count 2  | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm



# run converted/optimized tensorflow model FP32 on different video. Person count should be 7. But returned 11.
python main.py -i resources/people-detection.mp4 -m models/tensorflow/IR/ssd_mobilenet_v2_coco_2018_03_29/FP32/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU --max_person_count 2  | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

# run converted/optimized tensorflow model FP16 on different video. Person count should be 7. But returned 11.
python main.py -i resources/people-detection.mp4 -m models/tensorflow/IR/ssd_mobilenet_v2_coco_2018_03_29/FP16/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU --max_person_count 2  | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
# ________________________________



#REGION___________________________ DEBUG
# run pretrained intel model FP32 without FFMPEG for debug purpose
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m /home/workspace/models/pretrained/intel/person-detection-retail-0013/FP32/person-detection-retail-0013.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU
