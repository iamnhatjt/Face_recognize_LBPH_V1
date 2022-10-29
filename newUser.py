import cv2
import os

#initial recognize face
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video = cv2.VideoCapture(0)
num = 0


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
        # print(gray.shape)
        return faces,gray

def informationPerson():
    msv = input('Nhập MSV: ')
    name = input('Nhập Họ Tên Sinh Viên: ')
    print("điều chỉ Khuôn mắt vào khung chỉ định")
    return msv, name

msv, name = informationPerson()
filename = 'dataset/'+str(msv)+'_'+ '-'.join(name.strip().split()) +'.'


def cutImageFaceAndSave(img):
    #cut face and save into dataset
    faces,gray = detectFace(img)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2)
        global num
        num = num +1 
        # print(num)
        cv2.imwrite(filename+str(num)+'.jpg',gray[y:y + h, x:x + w])
        
        



while (True):
    ret, img = video.read()
    if ret:
        img = cv2.flip(img,1)

        #define pattern navigate customer take face on
        block(img)

        #detect face in the shape
        cutImageFaceAndSave(img)
        




        #show image
        cv2.imshow('Face',img)
        if cv2.waitKey(1) ==ord('q'):
            break
        elif num == 50:
            break

video.release()
cv2.destroyAllWindows()