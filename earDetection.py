import cv2
import numpy as np
import dlib
import imutils
from imutils import face_utils
from scipy.spatial import distance as dist
import os

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear

def ear_detect():
    ears = []
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    print("Loading facial landmark predictor Completed")
    # grab the indexes of the facial landmarks for the left and
    # right eye, respectively
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    ls = os.listdir("/content/Testimoney/Frames")
    ls.sort()
    for i in range(0, len(ls)-2):
        frame = ls[i]
        # print(frame)
        path = '/content/Testimoney/Frames/'+frame
        frame = cv2.imread(path,0)
        frame = imutils.resize(frame, width=450)
        rects = detector(frame, 0)
        for rect in rects:
            shape = predictor(frame, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            ears.append(ear)
    return (ears)