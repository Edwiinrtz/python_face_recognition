import cv2
import face_recognition as fr
import pickle
import numpy as np

known = pickle.loads(open("./dataset/known.pkl","rb").read())

haar_cascade = cv2.CascadeClassifier("/home/edwiinrtz/.local/lib/python3.8/site-packages/cv2/data/haarcascade_frontalface_alt.xml")
font = cv2.FONT_HERSHEY_SIMPLEX

video = cv2.VideoCapture(0)



knownEncodings = []
knownNames = []

for data in known:
    knownEncodings.append(data["code"])
    knownNames.append(data["name"])

while True:
    ret, frame = video.read()
    #frame = cv2.imread("./muchas.jpg")

    ##convertir la imagen a grey scale
    gris_image = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    
    ##Encontrar los rostros usando el face_recognition
    rostros = fr.face_locations(rgb_image)

    codificaciones = fr.face_encodings(rgb_image,rostros)

    ##detectar nombres
    nombres=[]
    for code in codificaciones:
        iguales = fr.compare_faces(knownEncodings,code)
        name = "desconocido"
        
        
        face_distances = fr.face_distance(knownEncodings, code)

        best_match_index = np.argmin(face_distances)
        print(iguales)
        if iguales[best_match_index]:
            name = knownNames[best_match_index]
        '''
        
        if True in iguales:
            matchedIdxs = [i for (i, b) in enumerate(iguales) if b]
            counts = {}
            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                name = knownNames[i]
                counts[name] = counts.get(name, 0) + 1
            # determine the recognized face with the largest number of
            # votes (note: in the event of an unlikely tie Python will
            # select first entry in the dictionary)
            print(counts)
            name = max(counts, key=counts.get)
        '''
        nombres.append(name)
        
    ##dibujar los rostros.
    for (face,nombre) in zip(rostros,nombres):
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame,nombre,(face[3]-10,face[0]-10),font,1,(0,255,0),2)

    ##Mostrar imagen/video
    cv2.imshow('img original',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()