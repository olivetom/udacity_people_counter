# MODEL #1 (model optimizer succeded, but inference engine throws runtime error)

Although this model is converted successfully using the following command:

/opt/intel/openvino/deployment_tools/model_optimizer/mo_caffe.py --input_model MobileNetSSD_deploy.caffemodel --input_shape=[1,3,300,300] --input=data --mean_values=data[104.0,117.0,123.0] --output=detection_out --input_model=MobileNetSSD_deploy.caffemodel --input_proto=MobileNetSSD_deploy.prototxt

Source: https://github.com/chuanqi305/MobileNet-SSD

ERROR FOUND:

(venv) root@922858f438fc:/home/workspace# python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m models/caffe/MobileNetSSD_deploy.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

Traceback (most recent call last):

 File "main.py", line 297, in <module>

  main()

 File "main.py", line 291, in main

  infer_on_stream(args, client)

 File "main.py", line 169, in infer_on_stream

  infer_network.load_model(args.model, args.device, args.cpu_extension)

 File "/home/workspace/inference.py", line 71, in load_model

  self.exec_network = self.plugin.load_network(self.network, device)

 File "ie_api.pyx", line 85, in openvino.inference_engine.ie_api.IECore.load_network

 File "ie_api.pyx", line 92, in openvino.inference_engine.ie_api.IECore.load_network

RuntimeError: Weights/biases are empty for layer: conv0/bn used in MKLDNN node: conv0/bn

Use ReadWeights and SetWeights methods of InferenceEngine::CNNNetReader to load them from .bin part of the IR

Output file is empty, nothing was encoded (check -ss / -t / -frames parameters if used)

# MODEL #2 (model optimizer succeded)

http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz

wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz

/opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model=ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb --output=detection_classes,detection_scores,detection_boxes,num_detections --tensorflow_object_detection_api_pipeline_config=ssd_mobilenet_v2_coco_2018_03_29/pipeline.config —tensorflow_use_custom_operations_config=/opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --input=image_tensor --input_shape=[1,300,300,3] --reverse_input_channels --data_type FP16

# MODEL #3 (model optimizer succeded)

 Use of the converted model in the python app requires special handling of input tensors of the network model:

Information of input image size, name: `image_info`, shape: [1x3], format: [BxC],
   where:

    - B - batch size
    - C - vector of 3 values in format [H,W,S], where H is an image height, W is an image width, S is an image scale factor (usually 1) 


http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz

/opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model=faster_rcnn_resnet101_coco_2018_01_28/frozen_inference_graph.pb --output=detection_classes,detection_scores,detection_boxes,num_detections --tensorflow_object_detection_api_pipeline_config=faster_rcnn_resnet101_coco_2018_01_28/pipeline.config —tensorflow_use_custom_operations_config=/opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json --input=image_tensor --input_shape=[1,300,300,3] --reverse_input_channels --data_type FP16

