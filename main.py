# import cv2
# from ultralytics import YOLO
#
# # Load YOLOv8 model (pre-trained on COCO dataset)
# model = YOLO('yolov8l.pt')
#
# # Animal class indices in COCO dataset (from bird to giraffe)
# animal_classes = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]  # Bird, Cat, Dog, Horse, Sheep, Cow, Elephant, Bear, Zebra, Giraffe
#
# # Define COCO class names directly
# classnames = [
#     'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
#     'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
#     'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
#     'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
#     'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
#     'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
#     'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
#     'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
#     'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
#     'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
# ]
#
# # RTSP stream URL
# rtsp_url = 'rtsp://admin:admin@789@192.168.1.199:554/unicast/c1/s0/live'  # Replace with your RTSP URL
#
# # Open the RTSP stream
# cap = cv2.VideoCapture(rtsp_url)
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # Run YOLOv8 detection
#     results = model(frame)
#
#     # Loop through the detections
#     for info in results:
#         parameters = info.boxes
#         for box in parameters:
#             x1, y1, x2, y2 = box.xyxy[0]
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#             confidence = box.conf[0]
#             class_detect = box.cls[0]
#             class_detect = int(class_detect)
#             conf = confidence * 100
#
#             # Check if the detected class is an animal in the specified range
#             if class_detect in animal_classes:
#                 class_name = classnames[class_detect]
#                 # Draw bounding box and label for the detected animal
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 label = f'{class_name} {conf:.2f}%'
#                 cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#
#     # Show the frame with detections
#     cv2.imshow('Animal Detection', frame)
#
#     # Exit if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Release resources
# cap.release()
# cv2.destroyAllWindows()

#
# import cv2
# import threading
# import time
# import json
# import os
# from ultralytics import YOLO
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import paho.mqtt.client as mqtt
#
# # Flask app setup
# app = Flask(__name__)
# CORS(app)
#
# # MQTT configuration
# broker = "192.168.1.120"  # Replace with your MQTT broker address
# port = 1883  # MQTT port
# topic = "pet/detection"
# mqtt_client = mqtt.Client(client_id="AnimalDetection")
#
# mqtt_client.connect(broker, port, keepalive=300)
# mqtt_client.loop_start()
#
# # YOLO model (pre-trained on COCO dataset)
# model = YOLO('yolov8n.pt')
#
# # Animal class indices in COCO dataset
# animal_classes = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]  # Bird, Cat, Dog, etc.
#
# # COCO class names
# classnames = [
#     'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
#     'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
#     'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
#     'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
#     'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
#     'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
#     'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
#     'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
#     'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
#     'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
# ]
#
# # Global dictionary to keep track of threads and stop events
# tasks_threads = {}
#
# # Ensure directories exist
# image_dir = "images"
# video_dir = "videos"
# os.makedirs(image_dir, exist_ok=True)
# os.makedirs(video_dir, exist_ok=True)
#
#
# def capture_image(frame):
#     timestamp = time.strftime("%Y%m%d_%H%M%S")
#     image_filename = os.path.join(image_dir, f"Animal_{timestamp}.jpg")
#     cv2.imwrite(image_filename, frame)
#     absolute_image_path = os.path.abspath(image_filename)
#     # print(f"Captured image: {absolute_image_path}")
#     return absolute_image_path
#
#
# def capture_video(rtsp_url):
#     timestamp = time.strftime("%Y%m%d_%H%M%S")
#     video_filename = os.path.join(video_dir, f"Animal_{timestamp}.mp4")
#
#     # Use the MP4V codec for MP4 format
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#
#     cap_video = cv2.VideoCapture(rtsp_url)
#     width = int(cap_video.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
#     # Create a VideoWriter object with MP4 format
#     out = cv2.VideoWriter(video_filename, fourcc, 20.0, (width, height))
#
#     start_time = time.time()
#     while int(time.time() - start_time) < 5:  # Capture for 5 seconds
#         ret, frame = cap_video.read()
#         if not ret:
#             break
#         out.write(frame)
#
#     cap_video.release()
#     out.release()
#     absolute_video_path = os.path.abspath(video_filename)
#     # print(f"Captured video: {absolute_video_path}")
#     return absolute_video_path
#
# # Function to run animal detection
# def detect_animal(rtsp_url, camera_id, site_id, display_width, display_height, type, stop_event):
#     cap = cv2.VideoCapture(rtsp_url)
#     last_detection_time = 0
#     while not stop_event.is_set():
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         frame = cv2.resize(frame, (display_width, display_height))
#
#         # Run YOLO detection
#         results = model(frame)
#
#         for info in results:
#             parameters = info.boxes
#             for box in parameters:
#                 x1, y1, x2, y2 = box.xyxy[0]
#                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#                 confidence = box.conf[0]
#                 class_detect = int(box.cls[0])
#
#                 current_time = time.time()
#                 if class_detect in animal_classes and (current_time - last_detection_time > 10):
#                     class_name = classnames[class_detect]
#                     conf = confidence * 100
#
#                     # Draw bounding box and label
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     label = f'{class_name} {conf:.2f}%'
#                     cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#
#                     frame_copy = frame.copy()
#                     image_filename = capture_image(frame_copy)
#                     video_filename = capture_video(rtsp_url)
#
#                     # Publish MQTT message
#                     message = {
#                         "cameraId": camera_id,
#                         "class": class_name,
#                         "siteId": site_id,
#                         "type": type,
#                         "image": image_filename,
#                         "video": video_filename
#                     }
#                     mqtt_client.publish(topic, json.dumps(message))
#                     print(f"Published message: {json.dumps(message)}")
#                     last_detection_time = current_time
#
#         # Display the frame
#         cv2.imshow(f'Animal Detection - Camera {camera_id}', frame)
#
#         # Break loop on 'q' key press
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyWindow(f'Animal Detection - Camera {camera_id}')
#
#
# # Function to start detection in a separate thread
# def start_detection(task):
#     camera_id = task["cameraId"]
#     site_id = task["siteId"]
#     display_width = task["display_width"]
#     display_height = task["display_height"]
#     type = task["type"]
#     stop_event = threading.Event()
#     tasks_threads[camera_id] = stop_event
#     rtsp_url = task["rtsp_link"]
#     thread = threading.Thread(target=detect_animal, args=(rtsp_url, camera_id, site_id, display_width, display_height, type, stop_event))
#     thread.start()
#
#
# # API to start detection
# @app.route('/start', methods=['POST'])
# def start_detection_endpoint():
#     tasks = request.json
#     for task in tasks:
#         camera_id = task["cameraId"]
#         if camera_id not in tasks_threads:
#             start_detection(task)
#     return jsonify({"status": "Animal detection tasks started"}), 200
#
#
# # API to stop detection
# @app.route('/stop', methods=['POST'])
# def stop_detection():
#     camera_ids = request.json.get('camera_ids', [])
#     if not isinstance(camera_ids, list):
#         return jsonify({"error": "cameraIds should be an array"}), 400
#
#     stopped_tasks = []
#     not_found_tasks = []
#
#     for camera_id in camera_ids:
#         if camera_id in tasks_threads:
#             print(f"Stopping detection for camera {camera_id}")  # Debugging print
#             tasks_threads[camera_id].set()  # Signal to stop the detection thread
#             del tasks_threads[camera_id]  # Remove from the active tasks
#             stopped_tasks.append(camera_id)
#             try:
#                 cv2.destroyWindow(f"Animal Detection - Camera {camera_id}")  # Close the window if it exists
#             except cv2.error as e:
#                 print(f"Error closing window for camera {camera_id}: {e}")  # Handle case where window doesn't exist
#         else:
#             not_found_tasks.append(camera_id)
#
#     success = len(stopped_tasks) > 0
#     response = {
#         "success": success,
#         "stopped": stopped_tasks,
#         "not_found": not_found_tasks
#     }
#
#     print(f"Stop API Response: {response}")  # Debugging print
#     return jsonify(response), 200
#
#
# # Run the Flask app
# if __name__ == '__main__':
#     from waitress import serve
#     serve(app, host='192.168.1.248', port=5000)


import cv2
import multiprocessing
import time
import json
import os
from ultralytics import YOLO
from flask import Flask, request, jsonify
from flask_cors import CORS
import paho.mqtt.client as mqtt

# Flask app setup
app = Flask(__name__)
CORS(app)

# MQTT configuration
broker = "192.168.1.120"  # Replace with your MQTT broker address
port = 1883  # MQTT port
topic = "pet/detection"
mqtt_client = mqtt.Client(client_id="AnimalDetection")

mqtt_client.connect(broker, port, keepalive=300)
mqtt_client.loop_start()

# YOLO model (pre-trained on COCO dataset)
model = YOLO('yolov8n.pt')

# Animal class indices in COCO dataset
animal_classes = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]  # Bird, Cat, Dog, etc.

# COCO class names
classnames = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Global dictionary to keep track of processes
tasks_processes = {}
process_lock = multiprocessing.Lock()

# Ensure directories exist
image_dir = "images"
video_dir = "videos"
os.makedirs(image_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)


def capture_image(frame):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    image_filename = os.path.join(image_dir, f"Animal_{timestamp}.jpg")
    cv2.imwrite(image_filename, frame)
    absolute_image_path = os.path.abspath(image_filename)
    return absolute_image_path


def capture_video(rtsp_url):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    video_filename = os.path.join(video_dir, f"Animal_{timestamp}.mp4")

    # Use the MP4V codec for MP4 format
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    cap_video = cv2.VideoCapture(rtsp_url)
    width = int(cap_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a VideoWriter object with MP4 format
    out = cv2.VideoWriter(video_filename, fourcc, 20.0, (width, height))

    start_time = time.time()
    while int(time.time() - start_time) < 5:  # Capture for 5 seconds
        ret, frame = cap_video.read()
        if not ret:
            break
        out.write(frame)

    cap_video.release()
    out.release()
    absolute_video_path = os.path.abspath(video_filename)
    return absolute_video_path


def detect_animal(rtsp_url, camera_id, site_id, display_width, display_height, type, stop_event):
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(f"Error: Unable to open video stream for camera {camera_id}")
        return

    window_name = f'Animal Detection - Camera {camera_id}'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Create a window that can be resized

    last_detection_time = 0

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (display_width, display_height))

        # Run YOLO detection
        results = model(frame)

        for info in results:
            parameters = info.boxes
            for box in parameters:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                confidence = box.conf[0]
                class_detect = int(box.cls[0])

                current_time = time.time()
                if class_detect in animal_classes and (current_time - last_detection_time > 10):
                    class_name = classnames[class_detect]
                    conf = confidence * 100

                    # Draw bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f'{class_name} {conf:.2f}%'
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    frame_copy = frame.copy()
                    image_filename = capture_image(frame_copy)
                    video_filename = capture_video(rtsp_url)

                    # Publish MQTT message
                    message = {
                        "cameraId": camera_id,
                        "class": class_name,
                        "siteId": site_id,
                        "type": type,
                        "image": image_filename,
                        "video": video_filename
                    }
                    mqtt_client.publish(topic, json.dumps(message))
                    print(f"Published message: {json.dumps(message)}")
                    last_detection_time = current_time

        # Display the frame
        cv2.imshow(window_name, frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow(window_name)


def start_detection(task):
    camera_id = task["cameraId"]
    site_id = task["siteId"]
    display_width = task["display_width"]
    display_height = task["display_height"]
    type = task["type"]
    rtsp_url = task["rtsp_link"]

    stop_event = multiprocessing.Event()
    process = multiprocessing.Process(target=detect_animal, args=(
    rtsp_url, camera_id, site_id, display_width, display_height, type, stop_event))

    with process_lock:
        # Stop any existing task for the same camera ID
        if camera_id in tasks_processes:
            print(f"Stopping existing detection for camera {camera_id}")
            tasks_processes[camera_id]['stop_event'].set()  # Signal the existing process to stop
            tasks_processes[camera_id]['process'].join()  # Wait for the existing process to finish
            del tasks_processes[camera_id]  # Remove from the active tasks

        tasks_processes[camera_id] = {'process': process, 'stop_event': stop_event}
        process.start()


@app.route('/start', methods=['POST'])
def start_detection_endpoint():
    tasks = request.json
    for task in tasks:
        start_detection(task)
    return jsonify({"message": 'Animal detection tasks started'}), 200


@app.route('/stop', methods=['POST'])
def stop_detection():
    camera_ids = request.json.get('camera_ids', [])
    if not isinstance(camera_ids, list):
        return jsonify({"error": "camera_ids should be an array"}), 400

    stopped_tasks = []
    not_found_tasks = []

    with process_lock:
        for camera_id in camera_ids:
            if camera_id in tasks_processes:
                print(f"Stopping detection for camera {camera_id}")  # Debugging print
                tasks_processes[camera_id]['stop_event'].set()  # Signal to stop the detection process
                tasks_processes[camera_id]['process'].join()  # Wait for the process to finish
                del tasks_processes[camera_id]  # Remove from the active tasks
                stopped_tasks.append(camera_id)
            else:
                not_found_tasks.append(camera_id)

    success = len(stopped_tasks) > 0
    response = {
        "success": success,
        "stopped": stopped_tasks,
        "not_found": not_found_tasks
    }

    print(f"Stop API Response: {response}")  # Debugging print
    return jsonify(response), 200


if __name__ == '__main__':
    from waitress import serve

    serve(app, host='0.0.0.0', port=5000)

