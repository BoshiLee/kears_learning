import cv2
from PIL import Image
from easygui import fileopenbox, msgbox
from datetime import datetime


inputfile=fileopenbox(msg='Select an image file...',title='Face Detection',default='./*.jpg')

pictPath = r'haarcascade_frontalface_default.xml'
face_cascade= cv2.CascadeClassifier(pictPath)
img = cv2.imread(inputfile)

t1= datetime.now()#测试起始时间
faces = face_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=3, minSize=(20,20))
t2 = datetime.now()#测试结束时间
print('time spend for face_detection: ', t2-t1)#显示总的时间开销

# 標註右下角底色黃色
cv2.rectangle(img, (img.shape[1]-120, img.shape[0]-20), (img.shape[1], img.shape[0]), (0,255,255),-1)
# 標註找到多少人臉
cv2.putText(img, 'Find '+ str(len(faces)) + ' face', (img.shape[1]-110,img.shape[0]-5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 1)



num=1
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    filename = 'face' + str(num) +'.jpg'
    image = Image.open(inputfile)
    imageCrop = image.crop((x,y,x+w,y+h))
    imageResize = imageCrop.resize((150,150), Image.ANTIALIAS)
    # imageResize = imageCrop.resize((150,150), Image.LANCZOS)
    imageResize.save(filename)
    num +=1
cv2.namedWindow('FACE',cv2.WINDOW_NORMAL)
cv2.imshow('FACE',img)
cv2.waitKey(0)