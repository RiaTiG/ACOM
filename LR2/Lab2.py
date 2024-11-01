import cv2
import numpy as np

cap = cv2.VideoCapture(0) 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Переводим кадр в формат HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    red_mask = mask1 | mask2
    red_frame = cv2.bitwise_and(frame, frame, mask = red_mask)

    # Морфологические преобразования
    kernel = np.ones((5, 5), np.uint8)
    
    # Применяем операцию открытия (эрозия + дилатация)
    opened_image = cv2.erode(mask, kernel, iterations=1)
    opened_image = cv2.dilate(opened_image, kernel, iterations=1)

    # Применяем операцию закрытия (дилатация + эрозия)
    closed_image = cv2.dilate(mask, kernel, iterations=1)
    closed_image = cv2.erode(closed_image, kernel, iterations=1)

    # Применяем операции открытия и зарытия cv2
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    for cnt in red_frame:
        M = cv2.moments(cnt)
        if M['m00'] > 0:
            area = M['m00']
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            print(f'Площадь объекта: {area}')

            x, y, w, h = cv2.boundingRect(red_mask)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)

    cv2.imshow('Original Frame', frame) # Выводим оригинальное изображение и площадь найденных обьектов
    cv2.imshow('Threshold', mask) # Фильтрация
    cv2.imshow('Opening', opened_image) # Открытие (эрозия + дилатация)
    cv2.imshow('Closing', closed_image) # Закрытие (дилатация + эрозия)
    # cv2.imshow('Red Frame', red_frame) # Фрейм с красным
    # cv2.imshow('Opening', opening) # Открытие 
    # cv2.imshow('Closing', closing) # Закрытие 

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()