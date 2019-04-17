"""
detection automobile amin'ny video, mampiasa opencv sy fichier xml haar cars.xml

"""
import cv2


car_cascade = cv2.CascadeClassifier('./datasets/haar/cars.xml')

video = cv2.VideoCapture('./datasets/video.avi')

while video.isOpened():
    ret, frame = video.read()
    if ret:
        temps = cv2.getTickCount()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cars = car_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4)
        for x, y, w, h in cars:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)

        fps = cv2.getTickFrequency()/(cv2.getTickCount() - temps)
        cv2.putText(frame, "FPS: {:05.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        cv2.imshow('Voitures', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
video.release()
cv2.destroyAllWindows()

