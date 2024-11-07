import cv2
import numpy as np

cap = cv2.VideoCapture(0) 

while True:
    ret, frame = cap.read()
    if not ret:
        break


    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, (0, 100, 100), (30, 255, 255))
    red_mask = mask
    value, thresholded_mask = cv2.threshold(red_mask,50,255,cv2.THRESH_BINARY)

    # red_frame = frame
    # red_frame[thresholded_mask == 0] = [0, 0, 0]

    # Морфологические преобразования
    morf = np.ones((5, 5), np.uint8)

    # Применяем операцию открытия (эрозия + дилатация)
    opened = cv2.erode(mask, morf, iterations=1)
    opened = cv2.dilate(opened, morf, iterations=1)

    # Применяем операцию закрытия (дилатация + эрозия)
    closed = cv2.dilate(mask, morf, iterations=1)
    closed = cv2.erode(closed, morf, iterations=1)

 
    moments = cv2.moments(closed)
    if ['m00'] != 0:
        x = int(moments['m10'] / moments['m00'])
        y = int(moments['m01'] / moments['m00'])
        x, y, w, h = cv2.boundingRect(red_mask)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
        print(f'Площадь объекта: {moments['m00']}')

    cv2.imshow('Original Frame', frame) # Выводим оригинальное изображение и площадь найденных обьектов
    # cv2.imshow('Threshold', thresholded_mask) # Фильтрация
    cv2.imshow('Opening', opened) # Открытие (эрозия + дилатация)
    cv2.imshow('Closing', closed) # Закрытие (дилатация + эрозия)
    # cv2.imshow('Red Frame', red_frame) # Фрейм с красным

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()