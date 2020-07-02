import cv2
import matplotlib.pyplot as plt


def process():
    initial_image = cv2.imread('images/image10.jpg')
    frontal_face_classifier = cv2.CascadeClassifier('classifiers/haarcascade_frontalface_default.xml')

    image_in_gray_scale = cv2.cvtColor(initial_image, cv2.COLOR_BGR2GRAY)
    faces = frontal_face_classifier.detectMultiScale(image=image_in_gray_scale, scaleFactor=1.3, minNeighbors=6)

    for (x_axis, y_axis, weight, height) in faces:
        cv2.rectangle(initial_image, (x_axis, y_axis), (x_axis + weight, y_axis + height), (255, 0, 0), 2)

    plt.imshow(initial_image)
    plt.show()
    cv2.imwrite('result/front_face.png', initial_image)


if __name__ == '__main__':
    process()
