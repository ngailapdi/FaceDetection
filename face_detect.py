import cv2
import sys
import numpy as np

# Get user supplied values
cascPath = "haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)
cap = cv2.VideoCapture(0)
samples = np.loadtxt('./ml_data/face_samples.data',np.float32)
responses = np.loadtxt('./ml_data/face_response.data',np.float32)
model = cv2.KNearest()
model.train(samples,responses)
gender = {0: 'male', 1: 'female'}
while cap.isOpened():
	ret, image = cap.read()
	#image = cv2.imread('./img/woman1.jpg')
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(
    	gray,
    	scaleFactor=1.1,
    	minNeighbors=5,
    	minSize=(30, 30),
    	flags = cv2.cv.CV_HAAR_SCALE_IMAGE
	)
	print("Found {0} faces!".format(len(faces)))

	# Draw a rectangle around the faces 
	crop_img = []
	#while (len(faces) == 0):
	for (x, y, w, h) in faces:
		cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
		crop_img = image[x:x+w, y:y+h]
	#gr = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
	cv2.imshow("Faces found", image)          
	roismall = cv2.resize(gray,(10,10))
	roismall = roismall.reshape((1,100))
	roismall = np.float32(roismall)
	retval, results, neigh_resp, dists = model.find_nearest(roismall, k = 1)
	print str(dists)
	digit = int((results[0][0]))
	print digit 
	print gender[digit]
		#cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

	cv2.imshow("Faces found", image)

	k = cv2.waitKey(10)
	if k == 27:
		break
# # Read the image
# image = cv2.imread(imagePath)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Detect faces in the image
# faces = faceCascade.detectMultiScale(
#     gray,
#     scaleFactor=1.1,
#     minNeighbors=5,
#     minSize=(30, 30),
#     flags = cv2.cv.CV_HAAR_SCALE_IMAGE
# )

# print("Found {0} faces!".format(len(faces)))

# # Draw a rectangle around the faces
# for (x, y, w, h) in faces:
#     cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# cv2.imshow("Faces found", image)
# while 1:
# 	k = cv2.waitKey(10)
# 	if k == 27:
# 		break

