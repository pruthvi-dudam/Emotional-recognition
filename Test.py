import cv2
import numpy as np
from PIL import Image
import os
#import pandas as pd
#import json as js

arr=['anger','happy','neutral','sad','surprise']
length = len(arr)
cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# #STORING IMAGES OF THIER RESPECTIVE EMOTION IN DATASET
#
# with open('/Users/pruthvirajdudam/Downloads/facial_expressions-master/surprise.txt','r') as f:
#     img = [line.strip() for line in f]
# for image in img:
#    loadedImage = cv2.imread('/Users/pruthvirajdudam/Downloads/facial_expressions-master/images/'+image)
#    cv2.imwrite('/Users/pruthvirajdudam/Downloads/facial_expressions-master/data_set/surprise/'+image,loadedImage)





# STORING FACE OBJECTS FOUND FROM IMAGES IN DATA_SET
#for i in range(0,length):
 #   with open('/Users/pruthvirajdudam/Downloads/facial_expressions-master/'+arr[i]+'.txt', 'r') as f:
  #      images = [line.strip() for line in f]

   # face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    #count = 0

    #for image in images:
     #   img = cv2.imread('/Users/pruthvirajdudam/Downloads/facial_expressions-master/data_set/'+arr[i]+'/'+image)
      #  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
       # faces = face_detector.detectMultiScale(gray, 1.3, 5)

        #for (x, y, w, h) in faces:
         #   cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
          #  count += 1
#           # Save the captured image into the datasets folder
 #           cv2.imwrite("/Users/pruthvirajdudam/Downloads/facial_expressions-master/dataset/" + str(i) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])
#
 #   print("\n Done creating face data")


# Path for face image database
# path = '/Users/pruthvirajdudam/Downloads/facial_expressions-master/dataset'
# recognizer = cv2.face.LBPHFaceRecognizer_create()
# detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml');
#
# # function to get the images and label data
# def getImagesAndLabels(path):
#    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
#    faceSamples=[]
#    ids = []
#
#    for imagePath in imagePaths:
#
#       PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
#       img_numpy = np.array(PIL_img,'uint8')
#
#       id = int(os.path.split(imagePath)[-1].split(".")[1])
#       faces = detector.detectMultiScale(img_numpy)
#
#       for (x,y,w,h) in faces:
#          faceSamples.append(img_numpy[y:y+h,x:x+w])
#          ids.append(id)
#
#     return faceSamples,ids
#
# print ("\n [INFO] Training faces....")
# faces,ids = getImagesAndLabels(path)
# recognizer.train(faces, np.array(ids))

# Save the model into trainer/trainer.yml
#recognizer.write('/Users/pruthvirajdudam/Downloads/facial_expressions-master/trainer/trainer.yml')

# Print the number of Emotions trained and end program
#print("\n [INFO] {0} Emotions trained. Exiting Program".format(len(np.unique(ids))))

recognizer = cv2.face.LBPHFaceRecognizer_create()
# Please input Absolute directory path of trainer.yml
recognizer.read('/Users/pruthvirajdudam/PycharmProjects/Emotion-Recognition/venv/lib/facial_expressions/trainer/trainer.yml')

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml');

font = cv2.FONT_HERSHEY_SIMPLEX

# iniciate id counter
id = 0

# Emotions related to ids: example ==> Anger: id=0,  etc

names = ['Anger', 'Happy', 'Neutral', 'Sad', 'Surprise']*60

img = cv2.imread('/Users/pruthvirajdudam/PycharmProjects/Emotion-Recognition/venv/images_to_test/test1.png') # please input Absolute directory path of image to test.

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height
# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.2,
    minNeighbors=5,
    minSize=(int(minW), int(minH)),
)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

    # Check if confidence is less them 100 ==> "0" is perfect match
    if (confidence < 100):
        id = names[id]
        confidence = "  {0}%".format(round(100 - confidence))
    else:
        id = "unknown"
        confidence = "  {0}%".format(round(100 - confidence))

    cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
    cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

cv2.imshow("output", img)
cv2.waitKey(0)

# Initialize and start realtime video capture (TEST FOR REAL_TIME PROCESSING)

# cam = cv2.VideoCapture(0)
# cam.set(3, 640)  # set video widht
# cam.set(4, 480)  # set video height
#
#     # Define min window size to be recognized as a face
# minW = 0.1 * cam.get(3)
# minH = 0.1 * cam.get(4)
#
# while True:
#     success, img = cam.read()
#     # ret, img = cam.read()
#     # img = cv2.imread('/Users/pruthvirajdudam/Desktop/t5.png')
#     img = cv2.flip(img, 1)  # Flip vertically
#
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     faces = faceCascade.detectMultiScale(
#         gray,
#         scaleFactor=1.2,
#         minNeighbors=5,
#         minSize=(int(minW), int(minH)),
#     )
#
#     for (x, y, w, h) in faces:
#
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
#         id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
#
#             # Check if confidence is less them 100 ==> "0" is perfect match
#         if (confidence < 100):
#             id = names[id]
#             confidence = "  {0}%".format(round(100 - confidence))
#         else:
#             id = "unknown"
#             confidence = "  {0}%".format(round(100 - confidence))
#
#         cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
#         cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)
#
#     cv2.imshow("video", img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# print("\n [INFO] Done detecting and Image is saved")
# cam.release()
# cv2.destroyAllWindows()

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver


#
#cv2.imshow("output",img1)
#cv2.waitKey(0)