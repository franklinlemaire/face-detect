# -*- coding: utf-8 -*-
import cv2 as cv

# charger les classificateurs en cascade pré-entrainés
face_cascade = cv.CascadeClassifier("Haarcascade_frontalface_default.xml")

# charger les images 
img = cv.imread('obama.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # Transformer l'image en niveau de gris

# exécution de la détection de visage
# detectMultiscale(image, scale factor, number of neighbors)
faces = face_cascade.detectMultiScale(gray, 1.1, 3)
# affichage des visages
for face in faces:
    print(face)