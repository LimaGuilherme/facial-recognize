import cv2
import numpy as np
import dlib
import matplotlib.pyplot as plt


initial_image = cv2.imread('images/image9.jpg')
imagem = cv2.cvtColor(initial_image, cv2.COLOR_BGR2RGB)


classificador_dlib = dlib.shape_predictor('classifier/shape_predictor_68_face_landmarks.dat')
detector_face = dlib.get_frontal_face_detector()

def anotar_rosto(imagem):
    retangulos = detector_face(imagem, 1)
    if len(retangulos) == 0:
        return None

    for k, d in enumerate(retangulos):
        cv2.rectangle(imagem, (d.left(), d.top()), (d.right(), d.bottom()), (255, 255, 0), 2)
    return imagem


def ponots_marcas_faciais(imagem):
    retangulos = detector_face(imagem, 1)
    if len(retangulos) == 0:
        return None

    marcos = []

    for ret in retangulos:
        marcos.append(np.matrix([[p.x, p.y] for p in classificador_dlib(imagem, ret).parts()]))

    return marcos


def anotar_marcos_facias(imagem, marcos):
    for marco in marcos:
        for idx, ponto in enumerate(marco):
            centro = (ponto[0, 0], ponto[0, 1])
            cv2.circle(imagem, centro, 3, (255, 255, 0), -1)
            cv2.putText(imagem, str(idx), centro, cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 2)



if __name__ == '__main__':
    imagem_anotada = imagem.copy()
    imagem_anotada = anotar_rosto(imagem_anotada)
    marcos = ponots_marcas_faciais(imagem_anotada)
    anotar_marcos_facias(imagem_anotada, marcos)
    plt.figure(figsize=(20, 10))
    plt.imshow(imagem_anotada)
    plt.show()
    cv2.imwrite('result-images/aaaaaa2aa.png', imagem_anotada)