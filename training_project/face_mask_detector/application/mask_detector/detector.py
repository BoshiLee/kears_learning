import time
import mtcnn
import cv2
import imutils
import numpy as np
from imutils.video import WebcamVideoStream
from keras.applications.mobilenet_v2 import preprocess_input
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from queue import Queue
from threading import Thread, enumerate

import random

def detect_and_predict_mask(frame, mask_net):
    faces = []
    locs = []
    preds = []

    detector = mtcnn.MTCNN()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_h, img_w = rgb.shape[:2]
    bboxes = detector.detect_faces(rgb)
    for bbox in bboxes:
        (x, y, w, h) = bbox['box']
        padding = 35
        (crop_x0, crop_x1) = (x - padding if x > padding else 0, x + w + padding if x + w + padding < img_w else img_w)
        (crop_y0, crop_y1) = (y - padding if y > padding else 0, y + h + padding if y + h + padding < img_h else img_h)
        face = rgb[crop_y0:crop_y1, crop_x0:crop_x1]
        face = cv2.resize(face, (224, 224))

        face = img_to_array(face)
        face = preprocess_input(face)

        faces.append(face)
        locs.append((x, y, x + w, y + h))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = mask_net.predict(faces, batch_size=32)

    return (locs, preds)

def set_saved_video(input_video, output_video, size):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    video = cv2.VideoWriter(output_video, fourcc, fps, size)
    return video

def video_capture(frame_queue):
    while cap.isOpened():
        print('put frame')
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (video_width, video_height),
                       interpolation=cv2.INTER_LINEAR)
        frame_queue.put(frame_resized)
    cap.release()

def inference(frame_queue, detections_queue, fps_queue):
    frame = frame_queue.get()
    prev_time = time.time()
    (locs, preds) = detect_and_predict_mask(frame=frame, mask_net=maskNet)
    detections_queue.put((locs, preds))
    fps = int(1/(time.time() - prev_time))
    fps_queue.put(fps)
    print("FPS: {}".format(fps))
    cap.release()


def drawing(frame_queue, detections_queue, fps_queue):
    random.seed(3)  # deterministic bbox colors
    video = set_saved_video(cap, 'output.avi', (640, 480))

    while cap.isOpened():
        fps = fps_queue.get()
        frame = frame_queue.get()
        (locs, preds) = detections_queue.get()
        # loop over the recognized faces
        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            (label, color) = ("Mask", (0, 255, 0)) if mask > withoutMask else ("No Mask", (0, 0, 255))
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.imshow('Inference', frame)
        video.write(frame)
        if cv2.waitKey(fps) == 27:
            break
    cap.release()
    video.release()
    cv2.destroyAllWindows()

def load_stream():
    print("[INFO] starting video stream...")
    cap = cv2.VideoCapture(r'sample2.mp4')

    return cap

if __name__ == '__main__':
    display_option = 1

    cap = load_stream()
    # threading.Thread(target = timetout, args=(30,)).start()
    maskNet = load_model(r'../../saved_model/model')

    frame_queue = Queue()
    # darknet_image_queue = Queue(maxsize=1)
    detections_queue = Queue(maxsize=1)
    fps_queue = Queue(maxsize=1)
    video_width = 640
    video_height = 480
    Thread(target=video_capture, args=(frame_queue,)).start()
    Thread(target=inference, args=(frame_queue, detections_queue, fps_queue)).start()
    Thread(target=drawing, args=(frame_queue, detections_queue, fps_queue)).start()
