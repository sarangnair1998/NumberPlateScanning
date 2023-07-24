#take an image and process it
#take a cascade of number plates from the internet
#Use that as a form to detect number plates
#draw a bounding box around it
#put a text saying "Number plate"


import cv2
import numpy as np


imageurl = "4.jpeg"
img = cv2.imread(imageurl)
classifier = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
number_plates = classifier.detectMultiScale(imgGray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

for (x, y, w, h) in number_plates:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    text = "Number Plate"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_x = x + (w - text_size[0]) // 2
    text_y = y - 10  # Adjust the text position above the bounding box
    cv2.putText(img, text, (text_x, text_y), font, font_scale,
                (0, 255, 0), font_thickness)

cv2.imshow("NUMBERPLATES",img)

cv2.waitKey(0)
cv2.destroyAllWindows()









