"""
Manao reconnaissance faciale ao anatina vidéo. Mampiasa opencv sy face_recognition (pip install face_recognition)

Eto dia épisode iray ao amin'ny serie Elementary no anaovana reconnaissance faciale (Watson sy sherlock no tiana ho
fantarina ao anatin'ilay vidéo)

Otrany mitady zavatra amin'ny moteur de recherche dia mila manome requette hanaovana recherche. Eto dia ny
sarin'ireo acteurs ireo no ho ampiasaina.

"""
import cv2
import face_recognition

video = cv2.VideoCapture('../datasets/elementarys03e19.avi')

#alaina ny sary ho ampiasaina
watson_image = face_recognition.load_image_file('../datasets/images/watson.jpg')
#avy eo ovana ny encodage mba hitovy amin'ny encondage ampiasain'ny algorithme izay ampiasana eto
watson_encoding = face_recognition.face_encodings(watson_image)[0]

sherlock_image = face_recognition.load_image_file('../datasets/images/sherlock.jpg')
sherlock_encoding = face_recognition.face_encodings(sherlock_image)[0]

#tabilao misy anireo tahery fantatra (izay ho hentina hanaovana famatarana ao amin'ilay vidéo)
tarehy_fantatra = [
    watson_encoding,
    sherlock_encoding
]

coord_tarehy = []
tarehy = []
anarana_tarehy = []

while video.isOpened():
    ret, frame = video.read()
    if ret:
        #ny face_recognition dia mampiasa espace RGB raha BGR no ampiasain'ny opencv
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #tadiavina ny coordonnées ny tarehy ao amin'ny frame
        coord_tarehy = face_recognition.face_locations(rgb_frame)
        #avy eo ny encodage
        tarehy = face_recognition.face_encodings(rgb_frame, coord_tarehy)

        anarana_tarehy = []
        for tarehy_video in tarehy:
            #jerena raha misy tarehy mitovy amin'ny tadiavina ao anatin'ilay vidéo, misy tolerance, eto dia
            #50% io tolérance io. Raha tiana ho précis kokoa dia afaka ampiakarina io tolérance io
            """
                io fonction compare_faces io dia mamerina tabilao mitovy alehibe amin'ny tabilao tarehy_fantatra
                ao anatin'ny valeur averin'ilay fonction dia misy valeur anakiro True na False. Ka raha eo amin'ny
                index 0 ohatra ny True dia manambara fa ny tarehy ao amin'ny index 0 ny tabilao tarehy_fantatra no hitany
                zany hoe Watson
                
            """
            mit = face_recognition.compare_faces(tarehy_fantatra, tarehy_video, tolerance=0.50)

            anarana = None
            if mit[0]:
                anarana = "Watson"
            elif mit[1]:
                anarana = "Sherlock"

            anarana_tarehy.append(anarana)

        for (ambony, akavanana, ambany, akavia), anarana in zip(coord_tarehy, anarana_tarehy):
            if not anarana:
                #jerena raha tsy misy anarana (zany ho tsy fantatra ilay tarehy
                continue

            #asiana sary rectangle ny tarehy fantatra
            cv2.rectangle(frame, (akavia, ambany), (akavanana, ambony), (0,255,0),2)
            #soratana ny anarany
            cv2.putText(frame, anarana, (akavia, ambony-5), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)


        key = cv2.waitKey(1)

        if  key == ord('q'):
            #ajanona rehefa manindry touche q
            break

        if key == ord('a'):
            #avance de 50 frame rehefa manindry touche a
            for i in range(50):
                ret, frame = video.read()

        cv2.imshow("Vidéo", frame)

#libérena ny ressource ary hidina ny fenêtre rehetra
video.release()
cv2.destroyAllWindows()