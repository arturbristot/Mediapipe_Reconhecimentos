import cv2
import mediapipe as mp

webcam = cv2.VideoCapture(0)

# Rosto
solucao_reconhecimento_rosto = mp.solutions.face_detection
reconhecedor_rostos = solucao_reconhecimento_rosto.FaceDetection()
desenho = mp.solutions.drawing_utils

# Mão
hand = mp.solutions.hands
Hand = hand.Hands(max_num_hands=2)

while True:
    verificador, img = webcam.read()

    # Para a mão!
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = Hand.process(imgRGB)
    handPoints = results.multi_hand_landmarks

    if not verificador:
        break

    lista_rostos = reconhecedor_rostos.process(img)

    if lista_rostos.detections is not None:
        for rosto in lista_rostos.detections:
            desenho.draw_detection(img, rosto)

    if handPoints:
        for points in handPoints:
            desenho.draw_landmarks(img, points, hand.HAND_CONNECTIONS)
            # print(points)

    cv2.imshow("WEBCAM", img)

    if cv2.waitKey(1) == 27:  # Esc key
        break

webcam.release()
cv2.destroyAllWindows()
