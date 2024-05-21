import cv2
import numpy as np
import face_recognition

# we have bgi image and we want to convert it to
imgELon= face_recognition.load_image_file("images/elon-musk-81fi31lzmqbcg80q.jpg")
imgELon= cv2.cvtColor(imgELon,cv2.COLOR_BGR2RGB)

imgTest= face_recognition.load_image_file("images/elon-musk-81fi31lzmqbcg80q.jpg")
imgTest= cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

#Main image
faceloc=face_recognition.face_locations(imgELon)[0]
encodeElon=face_recognition.face_encodings(imgELon)[0]
cv2.rectangle(imgELon,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,0),2)

 #Test image
facelocTest=face_recognition.face_locations(imgTest)[0]
encodeTest=face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(facelocTest[3],facelocTest[0]),(facelocTest[1],facelocTest[2]),(255,0,0),2)

results= face_recognition.compare_faces([encodeElon],encodeTest)

#We find face distance to find how much the image is far from the real image
#less is the distance value,more it is matched

faceDist= face_recognition.face_distance([encodeElon],encodeTest)

print(results,faceDist)






cv2.imshow('Elon Musk',imgELon)
cv2.imshow('Elon Test',imgTest)

cv2.waitKey(0)