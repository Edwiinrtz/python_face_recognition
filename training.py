import face_recognition as fr 
import cv2
import pickle


data = pickle.loads(open("./dataset/known.pkl","rb").read())
archivo = open("./dataset/known.pkl","wb")

name = input("Ingrese nombre: ")
haar_cascade = cv2.CascadeClassifier("/home/edwiinrtz/.local/lib/python3.8/site-packages/cv2/data/haarcascade_frontalface_alt.xml")

video = cv2.VideoCapture(0)

for i in range(100):
        xxx,image = video.read()

        gris_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ##faces = haar_cascade.detectMultiScale(gris_image, scaleFactor=1.2, minNeighbors=5)
        faces = fr.face_locations(rgb_image,model="hog")

        
        codificaciones = fr.face_encodings(rgb_image,faces)

        for codificacion in codificaciones:
                info = {"code":codificacion,"name":name}
                data.append(info)
        
        for (top,right,bottom,left) in faces:
                cv2.rectangle(image,(left,top),(right,bottom),(0,255,0))
                
        cv2.imshow("img",image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

archivo.write(pickle.dumps(data))
archivo.close() 



    



