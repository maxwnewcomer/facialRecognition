import cv2
import os
import numpy as np
import pickle
from imutils import paths
import imutils
import argparse
from tqdm import tqdm

def extract_embeddings(datasetPath, embeddingsPath, detectorPath, embedding_model, confidenceLim):
	print("[INFO] loading face detector...")
	protoPath = os.path.sep.join([detectorPath, "deploy.prototxt"])
	modelPath = os.path.sep.join([detectorPath, "res10_300x300_ssd_iter_140000.caffemodel"])
	detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

	print("[INFO] loading face recognizer...")
	embedder = cv2.dnn.readNetFromTorch(embedding_model)

	print("[INFO] quantifying faces...")
	imagePaths = list(paths.list_images(datasetPath))
	knownEmbeddings = []
	knownNames = []
	total = 0

	for (i, imagePath) in tqdm(enumerate(imagePaths), unit = 'Faces', total=len(imagePaths), smoothing = .5, desc = "[EMBEDDING]"):
		name = imagePath.split(os.path.sep)[-2]
		image = cv2.imread(imagePath)
		image = imutils.resize(image, width=600)
		(h, w) = image.shape[:2]
		imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
		detector.setInput(imageBlob)
		detections = detector.forward()
		if len(detections) > 0:
			i = np.argmax(detections[0, 0, :, 2])
			confidence = detections[0, 0, i, 2]
			if confidence > confidenceLim:
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
				face = image[startY:endY, startX:endX]
				(fH, fW) = face.shape[:2]
				if fW < 20 or fH < 20:
					continue
				faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
				embedder.setInput(faceBlob)
				vec = embedder.forward()
				knownNames.append(name)
				knownEmbeddings.append(vec.flatten())
				total += 1

	print("[INFO] serializing {} encodings...".format(total))
	data = {"embeddings": knownEmbeddings, "names": knownNames}
	f = open(embeddingsPath, "wb")
	f.write(pickle.dumps(data))
	f.close()
