import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import numpy as np
import cv2
import time

samples =  np.empty((1,100))
responses = []
cap = cv2.VideoCapture(0)
imgPath = ['./img/man2.jpg', './img/men1.jpg','./img/men2.jpg', './img/woman.jpg','./img/women1.jpg', './img/women2.jpg']
label = [0, 0, 0, 1, 1 ,1]
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
count = 0
for img in imgPath:
	image = cv2.imread(img)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(
   		gray,
   		scaleFactor=1.1,
   		minNeighbors=5,
   		minSize=(30, 30),
   		flags = cv2.cv.CV_HAAR_SCALE_IMAGE
	)
	print("Image: {0}".format(img))
	print("Found {0} faces!".format(len(faces)))

	# Draw a rectangle around the faces
	crop_img = []
	for (x, y, w, h) in faces:
		cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
		crop_img = image[x:x+w, y:y+h]
	gr = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
	cv2.imshow("Faces found", image)
	key = cv2.waitKey()
	print key
	time.sleep(5)
	responses.append((key-48))
	count += 1
	image1 = cv2.resize(gr,(10,10))
		#cv2.imshow("Face", image1)
		#print str(image1)
	sample = image1.reshape((1,100))
	samples = np.append(samples, sample, 0)
print responses
responses = np.array(responses,np.float32)
responses = responses.reshape((responses.size,1))

print "training complete"

np.savetxt('./ml_data/face_samples.data',samples)
np.savetxt('./ml_data/face_response.data',responses)
