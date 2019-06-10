from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

#argparse is used to parse through various command-line arguments which we are going to pass.
ap=argparse.ArgumentParser()

# add_argument functions adds the argument which the system expects to be passed as the argument. Here the argument is named as shapePredictorPath
# The short form of the argument is designated as -s. The help section tells us what the argument is.
# The required section tells us that this argument is absolutely required when set True.
ap.add_argument("-s","--shapePredictorPath",required=True,help="Path to the trained facial landmarks predictor in the dlib library")
ap.add_argument("-i","--imagePath",required=True,help="Path to the input image on which we want to perform facial Landmarks detection")

# parse_args() parses through the arguments passed.
# vars converts the argument passed into dictionary where the key is the name of the argument which was given to the arguments 
# when it was added using add_argument.
args=vars(ap.parse_args())

#get_frontal_face_detector helps us to get access of the face detection tool inside the dlib library. The instance of it is stored in the detector.
detector=dlib.get_frontal_face_detector()

#shape_predictor helps us to get access to the landmarks detection tool in the specified path passed as the argument.
predictor=dlib.shape_predictor(args["shapePredictorPath"])

#But before the face detection and the facial landmarks detection we need to load the image.
# cv2.imread loads the image in the image variable.
# Whereas the resize function resizes the image and the cvtColor converts the color image to the gray-scale image.
# We donnot need to necessarily convert the image to the gray-scale image.
image=cv2.imread(args["imagePath"])
image=imutils.resize(image,width=500)
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# detector which stores the frontal face detection of the dlib lib helps to detect the face in the passed argument image. It returns the rect object
# which stores the co-ordinates of the detected faces. Multiple faces can also be detected.
rects=detector(gray,1)


# The for loop, loops through the various faces in the image which is detected. i is the auto -incrementing variable.
for (i,rect) in enumerate(rects):

#The shape variable stores the 68 co-ordinates of the landmarks of the face predicted.
	shape=predictor(gray,rect)

# the shape_to_np converts the shape variable to the numpy array where the co-ordinates are stored as x-y cordinated of each of the 68 coordinates.
	shape=face_utils.shape_to_np(shape)
	(x,y,w,h)=face_utils.rect_to_bb(rect)
	cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
	cv2.putText(image, "Face {}".format(i+1),(x-10,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
	for(x,y) in shape:
		cv2.circle(image,(x,y),1,(0,0,255),-1)

#Outputs the image which we have modified i.e added the representation of the landmarks and showed the faces predicted.
cv2.imshow("Output",image)
cv2.waitKey(0)
