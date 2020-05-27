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


# Bound box
box_bgr_color = (0, 255, 0)
box_thickness = 1
box_linetype = cv2.LINE_AA

# Text
font = cv2.FONT_HERSHEY_PLAIN
font_position_xy_info = (5, 15)
font_position_xy_warning = (5, 35)
font_scale = 1
font_bgr_color_info = (255, 255, 255)
font_bgr_color_warn = (0, 0, 255)
font_thickness = 1
font_background_color = (0, 0, 0)



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
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.6,
                        help="Probability threshold for detections filtering"
                        "(0.6 by default)")
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
    return



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


    # draw boxes
    for detection in results[0][0]:
        confidence = detection[2]
        if confidence > prob_threshold:
            # scale up coordinates
            box = detection[3:7] * np.array([width, height, width, height])
            (x_min, y_min, x_max, y_max) = box.astype("int")
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_bgr_color, box_thickness, box_linetype)
            cv2.putText(frame, "Person detected", font_position_xy_warning, font,
                        font_scale, font_bgr_color_warn, font_thickness, box_linetype)

    # Extract Information from Model Output
    confidence_array = results[0,0,:,2]

#     log.debug("#boxes {}, #lowBox {}, #discardBox {}".format(
#         np.count_nonzero(confidence_array > prob_threshold),
#         np.count_nonzero((confidence_array > 0.1) & (confidence_array <= prob_threshold)),
#         np.count_nonzero(confidence_array <= 0.1)))


    current_count = np.count_nonzero(confidence_array > prob_threshold)

    return frame, int(current_count)



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

    # Handle Different Input Streams
    single_image_mode = False

    if args.input == 'CAM':
        input_stream = 0
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp'):
        single_image_mode = True
        input_stream = args.input
    elif args.input.endswith('.mp4') or args.input.endswith('.avi'):
        input_stream = args.input
    else:
        assert os.path.isfile(args.input), "Specified input file doesn't exist"
        log.info("File type not supported:", args.input)
        sys.exit(1)

    # Initialise the Network class
    infer_network = Network()


    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold


    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.device, args.cpu_extension, current_request_id)


    # get width and height of image the model process
    net_input_shape = infer_network.get_input_shape()


    ### TODO: Handle the input stream ###
    cap = cv2.VideoCapture(input_stream)


    cap.open(input_stream)


    source_width = int(cap.get(3))
    source_height = int(cap.get(4))

    #out = cv2.VideoWriter('out.mp4', 0x00000021, 30, (source_width, source_height))

    # frame sliding window based count
    frame_window = 8
    windowed_person_count = np.zeros(frame_window)
    windowed_zero_count = np.zeros(frame_window)
    frame_number = -1

    ### TODO: Loop until stream is over ###
    while cap.isOpened():

        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(44)

        frame_number += 1

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

            ### TODO: Get the results of the inference request ###
            results = infer_network.get_output(current_request_id)


        inference_stop = time.time() - inference_start

        if args.perf_counts:
            perf_count = infer_network.performance_counter(current_request_id)
            log_performance_counts(perf_count)

        ### TODO: Extract any desired stats from the results ###
        frame, person_count = frame_and_count(frame,
                                                results,
                                                source_width,
                                                source_height,
                                                args.prob_threshold)
        windowed_person_count[frame_number] = person_count
        person_count = int(np.amax(windowed_person_count))

        if (frame_number == frame_window - 1):
            frame_number = -1


        # IMPORTANT NOTICE:
        # another way to avoid counting persons repeatedly for low confidence detectors or
        # detectors where no instance detection is available
        # here we could use from skimage.metrics import structural_similarity as ssim
        # in order to discard counting from very similar frames.

        # out_frame shape (1,c,h,w)

        #_________________ TODO: Calculate and send relevant information on ###
        ### Perform analysis on the output to determine the number of people in ###
        ### frame, time spent in frame, and the total number of people counted ###
        ### current_count, total_count and duration to the MQTT server ###
        ### Topic "person": keys of "count" and "total" ###
        ### Topic "person/duration": key of "duration" ###
        # Publish messages to the MQTT server

        if person_count > person_last_count:
            person_start_time = time.time()
            person_total_count = person_total_count + person_count - person_last_count
            client.publish("person", json.dumps({"count": person_count}))
            client.publish("person", json.dumps({"total": person_total_count}))
            log.debug("person total count: {}".format(person_total_count))


        elif person_count < person_last_count:
                person_duration = int(time.time() - person_start_time)
                client.publish("person/duration",json.dumps({"duration": person_duration}))
                client.publish("person", json.dumps({"count": person_count}))

        person_last_count = person_count

        if args.max_person_count and person_count > args.max_person_count:
            client.publish("person", json.dumps({"max_count": person_count}))

            txt = "Max person count alert!"
            (txt_width, txt_height) = cv2.getTextSize(txt, font, font_scale, thickness=font_thickness)[0]
            txt_offset_x = 100
            txt_offset_y = frame.shape[0] - 10
            box = ((txt_offset_x, txt_offset_y + 2), (txt_offset_x + txt_width, txt_offset_y - txt_height - 2))
            cv2.rectangle(frame, box[0], box[1], font_background_color, box_linetype)
            cv2.putText(frame, txt, (txt_offset_x, txt_offset_y), font, font_scale, font_bgr_color_warn, font_thickness)


        #__________________ OUTPUT FRAME REGION___________________________
        inference_time_txt = "Inference time {:.3f}ms".format(inference_stop * 1000)
        cv2.putText(frame, inference_time_txt, font_position_xy_info, font, font_scale, font_bgr_color_info, font_thickness, box_linetype)

        ### TODO: Write an output image if `single_image_mode` ###
        if single_image_mode:
            cv2.imwrite('out_image.jpg', out_frame)

        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)
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
    log.basicConfig(filename='./people_counter.log', level=log.DEBUG)

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
