# USAGE
# # To read and write back out to video:
# python3 People_counter_final.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
# --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input output/testhour1.avi \
# --output output/output_e.avi
#
# To read from webcam and write back out to disk:
# python3 People_counter_final.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
# --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel \ 
# --output output/webcam_final2.avi

# import the necessary packages
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video.pivideostream import PiVideoStream
from imutils.video import VideoStream
from imutils.video import FPS
from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
from csv import writer
import csv

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-i", "--input", type=str,
	help="path to optional input video file")
ap.add_argument("-o", "--output", type=str,
	help="/home/pi/Downloads/people-counting-opencv")
ap.add_argument("-c", "--confidence", type=float, default=0.4,
	help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=90,
	help="# of skip frames between detections")
# ap.add_argument("-pi", "--picamera", type=int, default=-1,
# 	help="whether or not the Raspberry Pi camera should be used")
ap.add_argument("-n", "--num-frames", type=int, default=324000,
	help="# of frames to loop over for FPS test")
ap.add_argument("-d", "--display", type=int, default=-1,
	help="Whether or not frames should be displayed")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# if a video path was not supplied, grab a reference to the webcam
if not args.get("input", False):
	print("[INFO] starting video stream...")
# 	camera = PiCamera()
#     camera.resolution = (320, 240)
#     camera.framerate = 32
#     rawCapture = PiRGBArray(camera, size=(320, 240))
#     stream = camera.capture_continuous(rawCapture, format="bgr",
# 	use_video_port=True)
	vs = PiVideoStream(resolution=(304,208)).start()
	time.sleep(2.0)
# usePiCamera=args["picamera"] > 
# otherwise, grab a reference to the video file
else:
	print("[INFO] opening video file...")
	vs = cv2.VideoCapture(args["input"])

csvresult = open("/home/pi/results8.csv","w")
with csvresult:
	fnames=["exit","entrance","hall","office","frame","time"]
	writer1 = csv.DictWriter(csvresult,fieldnames=fnames)
	writer1.writeheader()


# initialize the video writer (we'll instantiate later if need be)
writer = None

# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)
W = None
H = None

# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
totalDown = 0
totalUp = 0
hall = 0
office = 0

# start the frames per second throughput estimator
fps = FPS().start()

# loop over frames from the video stream
while fps._numFrames < args["num_frames"]:
	# grab the next frame and handle if we are reading from either
	# VideoCapture or VideoStream
	frame = vs.read()
	frame = frame[1] if args.get("input", False) else frame

	# if we are viewing a video and we did not grab a frame then we
	# have reached the end of the video
	if args["input"] is not None and frame is None:
		break

	# resize the frame to have a maximum width of 500 pixels (the
	# less data we have, the faster we can process it), then convert
	# the frame from BGR to RGB for dlib
	frame = imutils.resize(frame, width=400)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# if the frame dimensions are empty, set them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# if we are supposed to be writing a video to disk, initialize
	# the writer
	if args["output"] is not None and writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(W, H), True)

	# initialize the current status along with our list of bounding
	# box rectangles returned by either (1) our object detector or
	# (2) the correlation trackers
	status = "Waiting"
	rects = []

	# check to see if we should run a more computationally expensive
	# object detection method to aid our tracker
	if totalFrames % args["skip_frames"] == 0:
		# set the status and initialize our new set of object trackers
		#status = "Detecting"
		trackers = []

		# convert the frame to a blob and pass the blob through the
		# network and obtain the detections
		blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
		net.setInput(blob)
		detections = net.forward()

		# loop over the detections
		for i in np.arange(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated
			# with the prediction
			confidence = detections[0, 0, i, 2]

			# filter out weak detections by requiring a minimum
			# confidence
			if confidence > args["confidence"]:
				# extract the index of the class label from the
				# detections list
				idx = int(detections[0, 0, i, 1])

				# if the class label is not a person, ignore it
				if CLASSES[idx] != "person":
					continue

				# compute the (x, y)-coordinates of the bounding box
				# for the object
				box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
				(startX, startY, endX, endY) = box.astype("int")

				# construct a dlib rectangle object from the bounding
				# box coordinates and then start the dlib correlation
				# tracker
				tracker = dlib.correlation_tracker()
				rect = dlib.rectangle(startX, startY, endX, endY)
				tracker.start_track(rgb, rect)

				# add the tracker to our list of trackers so we can
				# utilize it during skip frames
				trackers.append(tracker)

	# otherwise, we should utilize our object *trackers* rather than
	# object *detectors* to obtain a higher frame processing throughput
	else:
		# loop over the trackers
		for tracker in trackers:
			# set the status of our system to be 'tracking' rather
			# than 'waiting' or 'detecting'
			status = "Tracking"

			# update the tracker and grab the updated position
			tracker.update(rgb)
			pos = tracker.get_position()

			# unpack the position object
			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())

			# add the bounding box coordinates to the rectangles list
			rects.append((startX, startY, endX, endY))

	# draw a horizontal line in the center of the frame -- once an
	# object crosses this line we will determine whether they were
	# moving 'up' or 'down'
	cv2.line(frame, (0, (3*H // 4)), (W, (3*H // 4)), (0, 255, 255), 2)
	#cv2.line(frame, (W // 5, 0), (W // 5, H), (0, 255, 255), 2)
	#cv2.line(frame, ((4*W//5), 0), ((4*W//5), H), (0, 255, 255), 2)
	#cv2.line(frame, (0, (4*H // 5)), (W, (4*H // 5)), (0, 255, 255), 2)
	
	pts = np.array([[W // 4, 0], [W // 4, ((3*H // 4))], [(3*W // 4), ((3*H // 4))], [(3*W // 4), 0]])
	isClosed = False
	cv2.polylines(frame, [pts], isClosed, (0, 255, 255), 2)

	# use the centroid tracker to associate the (1) old object
	# centroids with (2) the newly computed object centroids
	objects = ct.update(rects)

	# loop over the tracked objects
	for (objectID, centroid) in objects.items():
		# check to see if a trackable object exists for the current
		# object ID
		to = trackableObjects.get(objectID, None)

		# if there is no existing trackable object, create one
		if to is None:
			to = TrackableObject(objectID, centroid)

		# otherwise, there is a trackable object so we can utilize it
		# to determine direction
		else:
			# the difference between the y-coordinate of the *current*
			# centroid and the mean of *previous* centroids will tell
			# us in which direction the object is moving (negative for
			# 'up' and positive for 'down')
			#x = [d[1] for d in to.centroids]
			#directionx = centroid[1] - np.mean(x)
			#to.centroids.append(centroid)
			
			y = [c[1] for c in to.centroids]
			x = [c[0] for c in to.centroids]
			directiony = centroid[1] - np.mean(y) # (x,y) c(3,4), p(3,2) -> 4 - 2 = 2 positive is down, for right previous x - current x, for left previous x + current x
			directionx = centroid[0] - np.mean(x)
# 			leftdirect = centroid[0] + np.mean(x)
			to.centroids.append(centroid)
            
			# check to see if the object has been counted or not
			if not to.upc:
				# if the direction is negative (indicating the object
				# is moving up) AND the centroid is above the center
			#	line, count the object
				if directiony < 0 and centroid[1] < ((3*H // 4)) and (W // 4) < centroid[1] < (3*W // 4):
					totalUp += 1
					to.upc = True
#                     if centroid[1] < W // 4 and centroid[1] > (3*W // 4):
#                         totalUp+= 0
#                         to.upc = False
			if not to.downc:
				# if the direction is positive (indicating the object
				# is moving down) AND the centroid is below the
				# center line, count the object
				if directiony > 0 and centroid[1] > ((3*H // 4)):
					totalDown += 1
					to.downc = True
#                     if centroid[1] < W // 4 and centroid[1] > (3*W // 4):
#                         totalDown+= 0
#                         to.upc = False 
				# om is editing
			if not to.hallc:
                
				if directionx < 0 and centroid[0] < W // 4 and centroid[1] < ((3*H // 4)):
					hall += 1
					to.hallc = True
					
			if not to.officec:
				if directionx > 0 and centroid[0] > (3*W // 4) and centroid[1] < ((3*H // 4)):
					office += 1
					to.officec = True

		# store the trackable object in our dictionary
		trackableObjects[objectID] = to

		# draw both the ID of the object and the centroid of the
		# object on the output frame
		text = "ID {}".format(objectID)
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
	reporttime = (time.strftime("%H:%M:%S"))
	csvresult = open("C:/Users/einav/Dropbox/winter semester 2020/Study Project CS Image Recognition/counter/counter/results1.csv","a")
	#fnames=["exit","entrance","hall","office","frame","time"]
	writer1 = csv.DictWriter(csvresult,fieldnames=fnames)
# 	writer1.writeheader()
	with csvresult:     
			writer1.writerow({"exit":totalDown ,"entrance":totalUp,"hall":hall,"office":office,"frame":totalFrames,"time":reporttime})
	# construct a tuple of information we will be displaying on the
	# frame
	info = [
		("Entrance", totalUp),
		("Exit", totalDown),
		#("Status", status),
        ("Hall", hall),
        ("Offices", office)
	]

	# loop over the info tuples and draw them on our frame
	for (i, (k, v)) in enumerate(info):
		text = "{}: {}".format(k, v)
		cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

	# check to see if we should write the frame to disk
	if writer is not None:
		writer.write(frame)

	# show the output frame
	# check to see if the frame should be displayed to our screen
# 	if args["display"] > 0:
# 		cv2.imshow("Frame", frame)
# 		key = cv2.waitKey(1) & 0xFF
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# increment the total number of frames processed thus far and
	# then update the FPS counter
	totalFrames += 1
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# check to see if we need to release the video writer pointer
if writer is not None:
	writer.release()

# if we are not using a video file, stop the camera video stream
if not args.get("input", False):
	vs.stop()

# otherwise, release the video file pointer
else:
	vs.release()

# close any open windows
cv2.destroyAllWindows()



