import dlib
import skimage.io
import scipy.io
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks_dlib.dat')

root_dir = '/home/icl/chenxin/dlib/demo/'
result_dir = '/home/icl/chenxin/dlib/result/'

for file in os.listdir(root_dir):
    print("Processing image {}".format(file))
    (fileName, ext) = os.path.splitext(file);
    print("{}".format(fileName))
    img = skimage.io.imread(root_dir + file)
    faces = detector(img, 1)
    for i, box in enumerate(faces):
        print("Face {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            i, box.left(), box.top(), box.right(), box.bottom()))
        # Get the landmarks/parts for the face in box
        landmarks = predictor(img, box)
        # Draw landmarks
        h, w = img.shape[:2]
        circle_radius = w / 300
        for j in range(68):
            cv2.circle(img, (landmarks.part(j).x, landmarks.part(j).y), circle_radius, (255,0,0), -1)
        # Show image
        # plt.figure()
        # plt.imshow(img)
        # plt.show()
        # Save image
        skimage.io.imsave(result_dir + fileName + '_detected.png', img)
        # Save to txt
        fd = open(result_dir + fileName + '.txt', 'w')
        for j in range(68):
            fd.write(str(landmarks.part(j).x) + ' ' + str(landmarks.part(j).y) + '\n')
        fd.close()