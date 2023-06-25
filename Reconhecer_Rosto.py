import cv2
import mediapipe as mp

webcam = cv2.VideoCapture(0)

solucao_reconhecimento_rosto = mp.solutions.face_detection
reconhecedor_rostos = solucao_reconhecimento_rosto.FaceDetection()
desenho = mp.solutions.drawing_utils

while True:
    verificador, frame = webcam.read()

    if not verificador:
        break

    lista_rostos = reconhecedor_rostos.process(frame)
    if lista_rostos.detections:
        for rosto in lista_rostos.detections:
            desenho.draw_detection(frame, rosto)

    cv2.imshow("WEBCAM", frame)

    if cv2.waitKey(3) == 27:
        break

webcam.release()