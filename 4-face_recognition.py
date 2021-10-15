import numpy as np
from PIL import Image
from mtcnn import MTCNN
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
#from keras.models import load_model
import cv2



pessoa = ["Deiski", "Nikoru"]
num_classes = len(pessoa)
cap = cv2.VideoCapture(0) # para usar o vídeo da camera do pc

detector = MTCNN() # reconhece a face na imagem
facenet = load_model("facenet_keras.h5") #cria embedding da face
model = load_model('faces.h5')


def extrair_face(image, box, required_size= (160, 160)):

    pixels = np.asarray(image)

    x1, y1, width, height = box
    x2, y2 = x1 + width, y1 + height

    face = pixels[y1:y2, x1:x2]

    image = Image.fromarray(face) # Reconvert to pillow
    image = image.resize(required_size) # Redimention the image

    return np.asarray(image)


def get_embedding(facenet, face_pixels):

    face_pixels = face_pixels.astype('float32')

    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std

    samples = np.expand_dims(face_pixels, axis=0)

    yhat = facenet.predict(samples)
    return yhat[0]


while True:

    _, frame = cap.read()

    faces = detector.detect_faces(frame)

    for face in faces:

        confidence = face['confidence']*100

        if confidence >= 98:
            x1, y1, w, h = face['box']
            face = extrair_face(frame, face['box'])

            face = face.astype('float32')/255

            emb = get_embedding(facenet, face)

            #transforma o vetor de 128 posições em uma matrix de 1 linha com 128 colunas
            tensor = np.expand_dims(emb, axis=0)

            #classe = model.predict_classes(tensor)[0]
            predict_x = model.predict(tensor)
            classe = np.argmax(predict_x, axis=1)
            prob = predict_x[0][classe] * 100

            user = str(pessoa[classe[0]]).upper()
            color = (192, 255, 119)

            cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), color, 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5

            cv2.putText(frame, user, (x1, y1-10), font, fontScale=font_scale, color=color, thickness=1)


    cv2.imshow("FACE_RECOGNTION", frame)

    key = cv2.waitKey(1)

    if key == 27: #ESC
        break

cap.release()
cv2.destroyAllWindows()