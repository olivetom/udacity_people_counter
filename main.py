"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2
import numpy as np

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60





def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    parser.add_argument("-pc", "--perf_counts", type=str, default=False,
                        help="Print performance counters. {yes/no} Default:no")
    parser.add_argument("-mc", "--max_person_count", type=int, default=None,
                        help="Alert when max person count surpassed")
    
    return parser







def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client







def frame_and_count(frame, results, width, height, prob_threshold):
#  The net outputs blob with shape: (1, 1, N, 7), where N is the number of detected bounding boxes. 
#   For each detection, the description has the format: [image_id, label, conf, x_min, y_min, x_max, y_max]
#         image_id - ID of the image in the batch
#         label - predicted class ID
#         conf - confidence for the predicted class
#         (x_min, y_min) - coordinates of the top left bounding box corner
#         (x_max, y_max) - coordinates of the bottom right bounding box corner.

    # Check model output blob is supported
    output_dims = results.shape
    if len(output_dims) != 4:
        log.error("Incorrect output dimensions for SSD like model")
        sys.exit(1)
        
    max_proposal_count, object_size = output_dims[2], output_dims[3]

    if object_size != 7:
        log.error("Output item should have 7 as a last dimension")
        sys.exit(1)
        
        
    # Extract Information from Model Output

    current_count = 0 
    
    for index, detection in enumerate(results[0][0]):
            
        confidence = detection[2]
        
        if confidence  > prob_threshold:
            # scale up coordinates
            box = detection[3:7] * np.array([width, height, width, height])
            
            (x_min, y_min, x_max, y_max) = box.astype("int")
            
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            current_count = current_count + 1
    
    
    return frame, current_count
        

    
    
def log_performance_counts(perf_count):
   
    log.info("{:<70} {:<15} {:<15} {:<15} {:<10}".format('name', 'layer_type',
                                                      'exec_type', 'status',
                                                      'real_time, us'))
    for layer, stats in perf_count.items():
        log.info("{:<70} {:<15} {:<15} {:<15} {:<10}".format(layer,
                                                          stats['layer_type'],
                                                          stats['exec_type'],
                                                          stats['status'],
                                                          stats['real_time']))


    
    
def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    
    
    
    # Initialise people counters
    person_last_count = 0
    person_total_count = 0
    person_start_time = 0
    
    # only 1 async inference request at a time
    current_request_id = 0

    
    client.publish("person", json.dumps({"total": person_total_count}))
    client.publish("person", json.dumps({"count": 0}))

    
    # Handle Different Input Streams
    single_image_mode = False

    if args.input == 'CAM':
        input_stream = 0
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp'):
        single_image_mode = True
        input_stream = args.input
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"

    
    # Initialise the Network class
    infer_network = Network()
    
    
#     if args.cpu_extension and "CPU" in args.device:
#         infer_network.add_extension(args.cpu_extension, "CPU")
#         log.info("CPU extension loaded: {}".format(args.cpu_extension))
    
    
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.device, args.cpu_extension)
    
    # get width and height of image the model process
    net_input_shape = infer_network.get_input_shape()

    ### TODO: Handle the input stream ###
    cap = cv2.VideoCapture(args.input)
    
    cap.open(args.input)

    source_width = int(cap.get(3))
    source_height = int(cap.get(4))

    #out = cv2.VideoWriter('out.mp4', 0x00000021, 30, (source_width, source_height))
    
    ### TODO: Loop until stream is over ###
    while cap.isOpened():

        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(44)
        
        
        ### TODO: Pre-process the image as needed ###
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        # HWC => CHW
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape) #(n,c,h,w)

        ### TODO: Start asynchronous inference for specified request ###
        inference_start = time.time()
        infer_network.exec_net(p_frame, current_request_id)

        ### TODO: Wait for the result ###
        if infer_network.wait(current_request_id) == 0:
            inference_stop = time.time() - inference_start

            ### TODO: Get the results of the inference request ###
            results = infer_network.get_output(current_request_id)
            
            
            if args.perf_counts:
                perf_count = infer_network.performance_counter(current_request_id)
                log.basicConfig(stream=sys.stdout, level=log.DEBUG)
                log_performance_counts(perf_count)

            
            
            ### TODO: Extract any desired stats from the results ###
            out_frame, person_count = frame_and_count(p_frame, 
                                                       results, 
                                                      net_input_shape[3],
                                                      net_input_shape[2],
                                                       args.prob_threshold)

            # to avoid counting persons repeatedly for low confidence detectors or 
            # detectors where no instance detection is available
            # here we could use from skimage.metrics import structural_similarity as ssim
            # in order to discard counting from very similar frames.
            
            # out_frame shape (1,c,h,w)

            inference_time_txt = "Inference time {:.3f}ms".format(inference_stop * 1000)
            cv2.putText(out_frame, inference_time_txt, (15, 15),
                        cv2.FONT_HERSHEY_PLAIN, 0.5, (200, 10, 10), 1)
            
            out_frame = out_frame[0] 
            out_frame = out_frame.transpose((1,2,0))
            out_frame = cv2.resize(out_frame, (source_width, source_height))

            ### TODO: Calculate and send relevant information on ###
            ### Perform analysis on the output to determine the number of people in ###
            ### frame, time spent in frame, and the total number of people counted ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            # Publish messages to the MQTT server
            
            if person_count > person_last_count:
                person_start_time = time.time()
                person_total_count = person_total_count + person_count - person_last_count
                client.publish("person", json.dumps({"total": person_total_count}))
            
            elif person_count < person_last_count:
                person_duration = int(time.time() - person_start_time)
                client.publish("person/duration",json.dumps({"duration": person_duration}))
            
            client.publish("person", json.dumps({"count": person_count}))
            person_last_count = person_count
            
            if args.max_person_count and person_count > args.max_person_count:
                txt = "Max person count alert!"
                (txt_width, txt_height) = cv2.getTextSize(txt, cv2.FONT_HERSHEY_PLAIN, 0.5, thickness=1)[0]
                txt_offset_x = 10
                txt_offset_y = out_frame.shape[0] - 10
                box_coords = ((txt_offset_x, txt_offset_y + 2), (txt_offset_x + txt_width, xt_offset_y - text_height - 2))
                cv2.rectangle(out_frame, box_coords[0], box_coords[1], (0, 0, 0), cv2.FILLED)
                
                cv2.putText(out_frame, txt2, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 0, 255), 1)
            
            
        ### TODO: Write an output image if `single_image_mode` ###
            if single_image_mode:
                cv2.imwrite('out_image.jpg', out_frame)
                
        ### TODO: Send the frame to the FFMPEG server ###
            sys.stdout.buffer.write(out_frame)
            sys.stdout.flush()
       
            if key_pressed == 27:
                break
            
    # Release the out writer, capture, and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


    
    
    
    
def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)
    # disconnet MQTT client
    client.disconnect()
    

if __name__ == '__main__':
    main()


