from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import dlib
import cv2

ap=argparse.ArgumentParser()
ap.add_argument("-s","--shapePredictorPath",required=True,help="Path to the shape predictor path of the dlib library")
args=vars(ap.parse_args())
print("[INFO] Camera starting up")
vs=VideoStream(-1).start()
print("[INFO] Face detector and Landmarks Detector loading up")
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor(args["shapePredictorPath"])

while True:
	frame=vs.read()
	frame=imutils.resize(frame,width=400)
	gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	rects=detector(gray,1)
	for rect in rects:
		shape=predictor(gray,rect)
		shape=face_utils.shape_to_np(shape)
		for(x,y) in shape:
			cv2.circle(frame,(x,y),1,(0,0,255),-1)
	cv2.imshow("Frame",frame)
	key=cv2.waitKey(1) & 0xFF
	if key==ord("q"):
		break
