# USAGE
# python detect_mask_video.py

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

def detect(frame, faceNet, maskNet):
	# frame shape and blob 
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(105.0, 178.0, 124.0))

	# input the blob in model
	faceNet.setInput(blob)
	objects = faceNet.forward()

	# variables
	faces = []
	locs = []
	preds = []

	# detection loop
	for i in range(0, objects.shape[2]):
		# probability of detection
		conf = objects[0, 0, i, 2]

		# filter low uality detections
		if conf > args["confidence"]:
			# coordinates of bounding box
			boundin_box = objects[0, 0, i, 3:7] * np.array([w, h, w, h])
			(x_i, y_i, x_j, y_j) = boundin_box.astype("int")

			# restrict the box within frame size
			(x_i, y_i) = (max(0, x_i), max(0, y_i))
			(x_j, y_j) = (min(w - 1, x_j), min(h - 1, y_j))

			# region of interest
			face = frame[y_i:y_j, x_i:x_j]
			if face.any():
				face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
				face = cv2.resize(face, (224, 224))
				face = img_to_array(face)
				face = preprocess_input(face)

				
				faces.append(face)
				locs.append((x_i, y_i, x_j, y_j))

	# edge case that therre has to be one detection
	if len(faces) > 0:
		# performing multiple predictions simultaneously
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)


	return (locs, preds)


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--face", type=str,
	default="face_detector")
parser.add_argument("-m", "--model", type=str,
	default="detect.model")
parser.add_argument("-p", "--probability", type=float, default=0.5)
args = vars(parser.parse_args())

# load model from directory

prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)


maskNet = load_model(args["model"])

# start the video 
video = VideoStream(src=0).start()
time.sleep(5.0)

# get frames from stream
while True:
	
	frame = video.read()
	frame = imutils.resize(frame, width=400)

	# classification as mask or no mask
	(locs, preds) = detect(frame, faceNet, maskNet)

	for (boundin_box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(x_i, y_i, x_j, y_j) = boundin_box
		(mask, withoutMask) = pred

		
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
			
		# accuracy
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		
		cv2.putText(frame, label, (x_i, y_i - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (x_i, y_i), (x_j, y_j), color, 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# exit when e is pressed
	if key == ord("e"):
		break

# close everything
cv2.destroyAllWindows()
video.stop()
