import cv2

#initial recognize face
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('recognizer/trainner.yml')

video = cv2.VideoCapture(0)

def block(img):
    centerHeight = img.shape[0] //2
    centerWidth = img.shape[1] //2
    sizeH,sizeW = 400,300
    cv2.rectangle(img,(centerWidth - sizeW//2, centerHeight-sizeH//2),
    (centerWidth +sizeW//2, centerHeight+sizeH//2 ),(0,255,0),3)

def detectFace(img):
        # centerHeight = img.shape[0] //2
        # centerWidth = img.shape[1] //2
        # sizeH,sizeW = 400,300
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # gray = gray[centerHeight-sizeH//2:centerHeight+sizeH//2, centerWidth - sizeW//2:centerWidth +sizeW//2]
        faces = faceDetect.detectMultiScale(gray,1.3,5)
        print(gray.shape)
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2)
            # detect who are you
            id, dist = recognizer.predict(gray[y:y+h,x:x+w])
            fontface = cv2.FONT_HERSHEY_SIMPLEX
            fontscale = 1
            fontcolor = (0,255,0)
            fontcolor1 = (0,0,255)
            if  dist <=70:
                if id >=100 or id <200:
                    cv2.putText(img, "Name: " + str('Nhat'), (x,y+h+30), fontface, fontscale, fontcolor ,2)
                    print(id)
                elif id>200 :
                    cv2.putText(img, "Name: " + str('Nguyen'), (x,y+h+30), fontface, fontscale, fontcolor ,2)

            else:
                cv2.putText(img, "Name: " + 'ai do....', (x,y+h+30), fontface, fontscale, fontcolor ,2)


              


while (True):
    ret, img = video.read()
    if ret:
        img = cv2.flip(img,1)

        #define pattern navigate customer take face on
        block(img)

        #detect face in the shape
       
        detectFace(img)




        #show image
        cv2.imshow('Face',img)
        if cv2.waitKey(1) ==ord('q'):
            break

video.release()
cv2.destroyAllWindows()