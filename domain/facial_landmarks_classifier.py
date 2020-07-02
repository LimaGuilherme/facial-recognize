import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt


def process():
    initial_image = cv2.imread('images/image9.jpg')
    image_in_rgb_scale = cv2.cvtColor(initial_image, cv2.COLOR_BGR2RGB)
    reference_image = image_in_rgb_scale.copy()

    dlib_classifier = dlib.shape_predictor('classifiers/shape_predictor_68_face_landmarks.dat')
    frontal_face_detector = dlib.get_frontal_face_detector()
    
    face_image = detect_face(frontal_face_detector, reference_image)
    landmarks = landmark_points(dlib_classifier, frontal_face_detector, face_image)
    create_facial_landmarks(face_image, landmarks)
    
    plt.figure(figsize=(20, 10))
    plt.imshow(face_image)
    plt.show()
    cv2.imwrite('result-images/landmarks.png', face_image)


def detect_face(frontal_face_detector, image):
    rectangles = frontal_face_detector(image, 1)
    if len(rectangles) == 0:
        return None

    for k, dimension in enumerate(rectangles):
        cv2.rectangle(image, (dimension.left(), dimension.top()), (dimension.right(), dimension.bottom()), (255, 255, 0), 2)

    return image


def landmark_points(dlib_classifier, frontal_face_detector, image):
    rectangles = frontal_face_detector(image, 1)
    if len(rectangles) == 0:
        return None

    landmarks = []

    for rectangle in rectangles:
        landmarks.append(np.matrix([[p.x, p.y] for p in dlib_classifier(image, rectangle).parts()]))

    return landmarks


def create_facial_landmarks(image, landmarks):
    for landmark in landmarks:
        for index, point in enumerate(landmark):
            center = (point[0, 0], point[0, 1])
            cv2.circle(image, center, 3, (255, 255, 0), -1)
            cv2.putText(image, str(index), center, cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 2)


if __name__ == '__main__':
    process()
