"""
Code mitady tarehy (detection visage) ao anaty flux vidéo, na avy amin'ny webcam na avy amin'ny film.
Mampiasa opencv sy ny fichier haar cascade hita ao amin'ny github ny opencv
"""

#importena ny librairie ampiasaina, eto dia opencv fa rehefa installer lay izy dia lasa cv2 no anarany
import cv2

tarehy_xml_url = 'https://raw.githubusercontent.com/octavebenil/opencv/master/datasets/haarcascade_frontalface_alt2.xml'

tarehy_cascade = cv2.CascadeClassifier(tarehy_xml_url)

video = cv2.VideoCapture(1)

print(video)


while video.isOpened():
  ret, frame = video.read()
  if ret:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    tarehy = tarehy_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4)
    
    for x, y, w, h in tarehy:
      cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow('Tarehy', frame)
    
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
    
video.release()
cv2.destroyAllWindows()
