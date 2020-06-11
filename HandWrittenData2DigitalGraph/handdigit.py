import numpy as np
import cv2
import imutils
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
model=tf.keras.models.load_model("digitRecognization.h5")

def plot(X,Y):
    zipped=sorted(zip(X,Y))
    X=[i for i,_ in zipped]
    Y=[i for _,i in zipped]
    print(X,Y)
    plt.plot(X,Y)
    plt.scatter(X,Y)
    for a,b in zip(X,Y):
        plt.text(a,b,str(f"({a},{b})"))
    plt.show()
def prepare_image(img):
    return img.reshape(-1,28,28,1)
def arrange(x,coordinate,co_x,num):
    threshold=15
    for i in x:
        if abs(coordinate-threshold)<i<abs(coordinate+threshold):
            x[i].append([co_x,str(num)])
            return x
    x[coordinate]=[[co_x,str(num)]]
    return x
def sort_datapoints(x):
    #print(x)
    for i in x:
        c=sorted(x[i],key=lambda j:j[0])
        x[i]=[j[1] for j in c]
    dp=[]
    for i in sorted(x):
        dp.append(int(''.join(x[i])))
    return dp
        
image_name='example_image.jpg'  #ImagePath
im = cv2.imread(image_name)

im=cv2.resize(im,(512,512),interpolation=cv2.INTER_AREA)

im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (1, 1), 0)
ret, im_th = cv2.threshold(im_gray, 135, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("th",im)
ctrs, hier = cv2.findContours(im_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
i=0
rects = [cv2.boundingRect(ctr) for ctr in ctrs if 8<cv2.contourArea(ctr)<800]
X,Y={},{}
if len(rects)==0:
    print("Sorry No Data Found. Either Image is unclear or the background is dark")
else:
    for rect in rects:
        if rect[2]>50 or rect[3]>50:
            continue
        if rect[2]<10 and rect[3]<10:
            continue
        cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
       
        # Make the rectangular region around the digit
        leng = int(rect[3] * 1.6)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        try:
            roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
            roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            roi = cv2.dilate(roi, (3, 3))
            roi=np.array(roi)
            roi=roi/255.0
            roi=roi.reshape(-1,28,28,1)
            x=model.predict(roi)
            p=list(x[0])
            predicted_num=p.index(max(p))
            cv2.putText(im,str(predicted_num),(rect[0]-10,rect[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
            if rect[0]<512/2.0:
                X=arrange(X,rect[1],rect[0],predicted_num)
            else:
                Y=arrange(Y,rect[1],rect[0],predicted_num)
        except:
            continue
    X=sort_datapoints(X)
    Y=sort_datapoints(Y)
    cv2.imshow("Resulting Image with Rectangular ROIs", im)
    if len(X)!=len(Y):
        print("Plot numbers are diffrent")
    else:
        plot(X,Y)
    print(X,Y)
    cv2.waitKey()
