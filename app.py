# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 03:09:44 2024

@author: LENOVO
"""
import cv2 as cv
import sys

# charger le classificateur pré-entraîné
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# charger l'image
img = cv.imread('obama.jpg')
img_gray=  cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# exécuter la détection de visage

faces = face_cascade.detectMultiScale(img_gray, 1.1, 8)

# Vérifier le nombre de visages
if len(faces) != 2:
    sys.exit('La photo doit avoir exactement deux visages, rééssayez...')

#récupération des dimensions de chaque visage
x1, y1, w1, h1 = faces[0]
x2, y2, w2, h2 = faces[1]

# extraction des deux visages de l'image

face1 = img[y1:y1+h1, x1:x1+w1]
face2 = img[y2:y2+h2, x2:x2+w2]

# redimensionner face2 aux dimensions de face1 et vice versa
face2 = cv.resize(face2, (w1, h1))
face1 = cv.resize(face1, (w2, h2))

#aux=face1
# remplacer face2 par face1
#face1=face2
img[y1:y1+h1, x1:x1+w1]=face2
# remplacer face1 par face1
img[y2:y2+h2, x2:x2+w2]=face1
#face2=aux

# afficher  l'échange de visage
 
cv.imshow('echange', img)
cv.waitKey(0)
cv.destroyAllWindows()   