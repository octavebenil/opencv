"""
Code tsotra indrindra raha hitady tarehy (detection visage) ao anaty flux vidéo (na avy amin'ny webcam na avy amin'ny vidéo na 
sary tsotra).
Mampiasa opencv sy ny fichier haar cascade hita ao amin'ny github ny opencv.
Alohan'ny hampiasana an'ity programme ity dia hamarino tsara fa efa vita ny installation ny opencv. Ny librairie ffmpeg ihany koa raha
te hampiasa vidéo fa tsy webcam
--------------------------------------------------------------------------------------------------------------------------------------
Dingana :

1. mamaky an'ilay fichier xml izay misy ny information  momba ny objet ho detectena (fichier izay vokatrin'ny entrainement efa natao
taloha) amin'ny alalan'ny fonction 'cv2.CascadeClassifier(chemin_vers_le_fichier_xml)'

2. maka ny sary amin'ny webcam na avy amin'ny fichier vidéo (film, vidéo mp4...) na sary tsotra

3. miantso ny fonction 'detectMultiScale(canvas, scaleFactor,minNeighbors)' : 
	- Ny canvas eto dia ilay frame na sary tiana hanaovana detection (moramora kokoa ny miasa amin'ny sary noir et blanc ka ovana 
    arak'izany ilay sary na frame ampiasaina. 
	- Ny scaleFactor kosa dia paramètre manambara ny coefficient de changement d'echelle entre deux images  hampiasain'ilay algorithme 
    de dectection (satria ny algorithme de detection eto dia mijery ny sary de gauche à droite avy eo de haut à bas mba hitadiavany
    ny objet izay voafaritra ao amin'ny fichier xml eto ambony (visage), ka ilay scaleFactor eto dia manambara hoe impiry izy manao 
    izay scan izay). Matetika dia entre 1.1 sy 1.5 ny valeur izy io.
	- Ny minNeighbors kosa dia paramètre ahafahana mamaritra ny isan'ny couche voisin ho verifiena mba hi-confirmena hoe misy marina
    ilay objet ho detectena (zany hoe: raha minNeighbors=2, dia jerena ny couche voisin anakiroa raha misy an'ilay objet ka raha hita 
    ao ilay objet dia azo heritreretina fa misy. Matetika dia entre 3 et 5 ny valeurs izy io.

ny scaleFactor sy minNeighbors dia inversement proportionnel

Misy paramètres optionnel koa afaka ampidirina amin'ny 'detectMultiScale' dia ny minSize sy maxSize mba hanalana ny faux positifs
ka raha ohatra ka ambanin'ny minSize ny rectangle anakiray dia tsy izy izay

Ny methode detectMultiScale dia mamerina valeur izay misy zavatra 04 (coordonnées x, y ; ny largeur ; ny hauteur ; ilay objet detecter
ao amin'ny sary. Amin'ny alalan'ireo valeurs 04 ireo dia afaka manao sary rectangle mba hanambarana ny objet hitany na voadetecteny
"""

#importena ny librairie ampiasaina, eto dia opencv fa rehefa installer lay izy dia lasa cv2 no anarany
import cv2

#Dingana 1 : Chargena ilay fichier cascade
tarehy_cascade = cv2.CascadeClassifier('../datasets/haar/haarcascade_frontalface_alt2.xml')

"""
Dingana faha 2 :
maka sary avy amin'ny webcam, ny 0 eto satria webcam intégré amin'ny ordinateur ilay izy. Raha misy webcam hafa 
ohatra dia ovaina ny valeur 0 io hoe 1, etc. Raha vidéo kosa no tiana hampiasaina dia solona ny chémin sy ny anaran'ilay video io
0 amin'ny paramètre ny fonction VideoCapture io ; ohatra : cv2.VideoCapture('video.avi') 
"""
cap = cv2.VideoCapture(0)

#miodina eo foana ny programme raha mbola misokatra ilay capture
while cap.isOpened():
  #mamaky an'ilay flux vidéo ka avadika ho sary tsirairay (na frame) ilay vidéo
  ret, frame = cap.read()
  #jerena raha ohatra ka misy erreur
  if ret:
    #avadika ho noir et blanc ilay frame azo mba hanamorana ny fikirakirana azy
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #Etape 3 : detection amin'izay
    tarehy = tarehy_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4)
    
    #avy eo alaina ny coordonée ny tarehy
    for x, y, w, h in tarehy:
      #faritana ny zavatra hitany
      cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
      
    #affichena avy eo ny frame
    cv2.imshow('Tarehy', frame)
    
  #rehefa manindry ny touche q dia azanona ny programme  
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

#libérena ny ressource rehetra    
video.release()
#hidina daholo ny fenêtre rehetra
cv2.destroyAllWindows()
