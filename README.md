# project-Handwritten-digit-recognition
# Developed a machine learning project which is helpful in recognising the handwritten digits used in bank cheques, deposit or withdrawl forms.

from PIL import Image, ImageEnhance
from numpy import asarray
import numpy as np
import matplotlib.pyplot as pt

import pandas as pd
import cv2
from sklearn.tree import DecisionTreeClassifier

data=pd.read_csv('mnist_train.csv').values
clf=DecisionTreeClassifier()
xtrain=data[0:60000, 1:]

train_label=data[0:60000, 0]
clf.fit(xtrain, train_label)


video=cv2.VideoCapture(0)
a=0
while True:
    a=a+1
    check,frame=video.read()
    cv2.imshow("Capture the image by pressing m key", frame)
    key=cv2.waitKey(1)
    if key== ord('m'):
        break
showPic=cv2.imwrite("2.jpg",frame)
print(showPic)
video.release()
cv2.destroyAllWindows()

name="2.jpg"
im=Image.open(str(name)).convert("L")

w,h=im.size
if(w>h):
    cut=(w-h)//2
    img=im.crop((cut,0,(w-cut),h))
elif(w<h):
    cut=(h-w)//2
    img=im.crop((0,cut,w,(h-cut)))
else:
    img=im
img.save("10crop.jpg")

Max=(28,28)
img.thumbnail(Max)

data=asarray(img)
len,br=(data.shape)

arr=np.array([])
arr2=np.array([x for x in range(0,784,1)])
for x in range(len):
    for y in range(br):
        arr=np.append(arr,(255-data[x][y]))
combine=np.vstack((arr2,arr))
np.savetxt('array.csv',combine,delimiter=',', fmt='%d')

made=pd.read_csv('array.csv').values
d=made[0]
d.shape=(28,28)

pt.imshow(255-d,cmap='gray')



print("the predicted result for given image is:")
print(clf.predict([made[0]]))
pt.show()












