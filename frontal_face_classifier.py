import cv2
import matplotlib.pyplot as plt

def va():
    group_of_people_image = cv2.imread('images/image7.jpg')
    frontal_face_classifier = cv2.CascadeClassifier('classifier/haarcascade_frontalface_default.xml')

    image_in_gray_scale = cv2.cvtColor(group_of_people_image, cv2.COLOR_BGR2GRAY)
    faces = frontal_face_classifier.detectMultiScale(image=image_in_gray_scale, scaleFactor=1.3, minNeighbors=6)
    print(len(faces))

    for (x, y, w, h) in faces:
        img = cv2.rectangle(group_of_people_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # roi_gray = image_in_gray_scale[y:y + h, x:x + w]
        # roi_color = img[y:y + h, x:x + w]
        # eyes = eye_cascade.detectMultiScale(roi_gray)
        # for (ex, ey, ew, eh) in eyes:
        #     cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    plt.imshow(group_of_people_image)
    plt.show()
    cv2.imwrite('result-images/aaaaaasdasdaa.png', group_of_people_image)

    # cv2.imshow('img', group_of_people_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def b():
    group_of_people_image = cv2.imread('images/image6.jpg')
    image_in_gray_scale = cv2.cvtColor(group_of_people_image, cv2.COLOR_BGR2GRAY)
    frontal_face_classifier = cv2.CascadeClassifier('classifier/haarcascade_frontalface_default.xml')
    faces = frontal_face_classifier.detectMultiScale(image=image_in_gray_scale, scaleFactor=1.3, minNeighbors=5)

    print(len(faces))
    for (x, y, w, h) in faces:
        cv2.rectangle(group_of_people_image, (x,w), (x+w, y+h), (255, 255, 0), 2)
    plt.figure(figsize=(50, 50))
    plt.imshow(group_of_people_image)
    plt.show()

    face_imagem = 0
    for (x,y,w,h) in faces:
        face_imagem += 1
        imagem_roi = group_of_people_image[y:y+h, x:x+w]
        imagem_roi = cv2.cvtColor(imagem_roi, cv2.COLOR_RGB2GRAY)
        # cv2.imwrite("face_" + str(face_imagem) + ".png", imagem_roi)
        cv2.imwrite('aaaaaasdasdaa.png', group_of_people_image)


if __name__ == '__main__':
    va()
