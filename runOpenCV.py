from extract_embeddings import extract_embeddings
from recognize_video import recognize_video
from recognize import recognize
from train_model import train

projectPath = "/Users/max/Desktop/Programming/OpenCV"
datasetPath = projectPath + "/dataset"
embeddingsPath = projectPath + "/output/embeddings.pickle"
detectorPath = projectPath + "/face_detection_model"
recognizerPath = projectPath + "/output/recognizer.h5"
embedding_model = projectPath + "/openface_nn4.small2.v1.t7"
label = projectPath + "/output/l.pickle"
imagePath = projectPath + "images/test.jpg"
confidenceLim = .5

extract_embeddings(datasetPath, embeddingsPath, detectorPath, embedding_model, confidenceLim)
train(embeddingsPath, recognizerPath, label, projectPath)
recognize_video(detectorPath, embedding_model, recognizerPath, label, confidenceLim, projectPath)
