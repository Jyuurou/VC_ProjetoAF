from mtcnn import MTCNN
from PIL import Image
from os import listdir
from os.path import isdir
from numpy import asarray

detector = MTCNN() # para enchergar as faces dentro de 1 imagem


def extrair_face(arquivo, size=(160, 160)):

    img = Image.open(arquivo) #caminho completo

    img = img.convert('RGB')
    array = asarray(img)
    results = detector.detect_faces(array)

    x1, y1, width, height = results[0]['box']

    x2, y2 = x1 + width, y1 + height

    face = array[y1:y2, x1:x2]

    image = Image.fromarray(face) # Reconvert to pillow
    image = image.resize(size) # Redimention the image

    return image

# Aumento de Dados (Data Augment) method "flip" (espelhado)
def flip_image(image):
    img = image.transpose(Image.FLIP_LEFT_RIGHT)
    return img

def load_fotos(diretorio_source, diretorio_target):

    for filename in listdir(diretorio_source):
        path = diretorio_source + filename
        path_tg = diretorio_target + filename
        path_tg_flip = diretorio_target + filename.replace(".", "_flip.")

        try:
            face = extrair_face(path)
            flip = flip_image(face)
            face.save(path_tg, "JPEG", quality=100,  progressive=True)
            flip.save(path_tg_flip, "JPEG", quality=100, progressive=True)
            print("Imagens salvas")
            print(format(path_tg))
            print(format(path_tg_flip))
        except:
            print("Erro na imagem ()", format(path_tg))


def load_dir(diretorio_source, diretorio_target):

    for subdir in listdir((diretorio_source)):

        path = diretorio_source + subdir + "\\"

        path_tg = diretorio_target + subdir + "\\"

        if not isdir(path):
            continue

        load_fotos(path, path_tg)


load_dir("C:\\Users\\Deiski\\Desktop\\Projeto Identificação faces\\fotos\\",
             "C:\\Users\\Deiski\\Desktop\\Projeto Identificação faces\\faces\\train\\")
