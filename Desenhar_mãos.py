import cv2
import mediapipe as mp

webcam = cv2.VideoCapture(0)

hand = mp.solutions.hands
Hand = hand.Hands(max_num_hands=1)
desenho = mp.solutions.drawing_utils

while True:
    check, img = webcam.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = Hand.process(imgRGB)
    handPoints = results.multi_hand_landmarks


    
    if handPoints:
        for points in handPoints:
            desenho.draw_landmarks(img, points, hand.HAND_CONNECTIONS)
            #print(points)
        


    cv2.imshow("imagem", img)
    if cv2.waitKey(3) == 27:
        break

webcam.release()