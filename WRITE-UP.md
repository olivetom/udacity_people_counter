## PROJECT SPECIFICATION

### Deploy a People Counter App at the Edge

Overview of my work.

This project was very helpful to understand deep learning object detection using Intel OpenVINO Toolkit. I found it extremely helpful for accelerating the project development time.

However, I should say that a marketable people counter app should deal with model accuracy by augmenting the plain detection with object tracking to avoid false negatives that increments the people count incorrectly. 

Another option to avoid counting persons repeatedly for low confidence detectors or detectors where no instance detection is available is to use 

​	a)  **skimage.metrics** the structural_similarity frame comparator in order to discard counting people from very similar frames.

​	b) **scipy.spatial.distance.cdist** to compare prediction boxes between N consecutive frames to discard repeated box counts in predictions.

Although I tried to install scipy and skimage it in the workspace I didn't succeded.

Regarding model optimizer, I found the process of transforming models that were not pretrained by Intel quiet hard. I think that the app developer should have detailed knowledge of the model in order to convert it properly and successfully. Examples: 

​	a) faster_rcnn_resnet101_coco_2018_01_28 although converted successfully with model optimizer, it requires special case when handling input shape in the people counter python app. See MODELS.md for further details.

​	b) MobileNetSSD_deploy.caffemodel. although it was converted successfully with model optimizer when loading the model with inference engine throws run-time error “Weights/biases are empty for layer: conv0/bn used in MKLDNN node: conv0/bn”

One more nice to have is a unit test suite to support the coding activity.

Despite all previous drawbacks, overall experience was very engaging and satisfactory.

Thank you very much.

Mauricio

**WRITE-UP**

1. Explain the process behind converting any custom layers. Explain the potential reasons for handling custom layers in a trained model.

If a model have a layer/operation/operator not already implemented/supported in the Intel OpenVINO toolkit, then the developer has to provide implementation of the corresponding operator in order for the model optimizer to reference them as a layer that must be handled by a CPU and not by the FPGA, GPU, etc. In this way, Intel reassures that all models can be executed with Intel Accelerated Hardware with fallback to CPU if they have custom layers.



2. Run the pre-trained model without the use of the OpenVINO™ Toolkit. Compare the performance of the model with and without the use of the toolkit (size, speed, CPU overhead). What about differences in network needs and costs of using cloud services as opposed to at the edge? 

When running the model without using the OpenVINO Toolkit, the size, speed and CPU overhead are worst because the OpenVINO toolkit includes the Model Optimizer that lowers size and processing overhead and increases inference speed of the transformed models.



3. Explain potential use cases of a people counter app, such as in retail applications. This is more than just listing the use cases - explain how they apply to the app, and how they might be useful.

People counter applications are diverse and useful. It can be deployed in any computer vision edge device that requires counting of people, audience counting, entrance/exit of people (although it should be extended with object tracking).



4. Discuss lighting, model accuracy, and camera focal length/image size, and the effects these may have on an end user requirement.

Scene lighting, model accuracy, and camera focal length/image size may substantially affect the end user requirements in various ways.

Lightning and focal length/image variations may lower the model’s accuracy and may require that model should be retrained using data augmentation techniques to include images with lightning variations/blur/rotation/flip/etc in the dataset.



5. Comparing Model Performance

Having that /opt/intel/openvino/deployment_tools/open_model_zoo/tools/accuracy_checker only works for model comparison using a pre-cured dataset which I don’t have for this application, then my method to compare models before and after conversion to Intermediate Representations were performing inferencing by hand using inference time in my computer’s CPU before conversion and using Udacity CPU after conversion. (Note I don’t have any kind of acceleration hardware such as GPU, FPGA, etc)

Results I found:

The difference between *model accuracy* pre- and post-conversion was almost null using FP32, 10% lower using FP16 and 0.5% lower using INT8. This implies, that edge applications for deep neural networks are possible. 



For *size comparison*, I used the size on disk of the model pre- and post-conversion and found that it was much smaller after model optimizer. For example, ssd_mobilenet_v2_coco_2018_03_29 Tensorflow format size was 202Mb. After conversion it was 33Mb in FP16 and 65Mb in FP32.



The *inference time* decreased between those of the model pre- and post-conversion. Increase in the throughput was about 2.0 and 3.7 times faster when using FP16/INT8 respectively,  instead of FP32.



Finally, I should say that significant model size and inference time reduction is accomplished by using Intel OpenVINO precision decreasing with a small penalty in accuracy. This is useful for deploying applications at the edge where much lower storage and computing power resources are available. Although, could be some circumstances where model complexity (parameters, gflops, custom operations) can adversely affect the throughput and must be carefully taken into consideration.

