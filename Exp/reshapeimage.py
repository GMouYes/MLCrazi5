import cv2
import os

w = 640
h = 640

def getimage(file_dir):
    images = {}
    for root, dirs, files in os.walk(file_dir):
        for name in files:
            images[name] = os.path.join(root, name)
    return images


if __name__ == '__main__':
    n = -1
    aa = os.getcwd()
    dirpath = os.path.join(aa, 'cvimage')    #cvimage should be replaced by the folder that contains origin image
    imagedic = getimage(dirpath)
    print(aa)
    print(type(aa))
    try:
        for key, value in imagedic.items():
            img = cv2.imread(value)
            img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img2 = cv2.resize(img1, (w, h))
            #cv2.imwrite('pos' + str(n + 1).rjust(3, '0') + '.jpg', img2)
            imgname = aa + '\grayimage\pos' + str(n+1) +'.jpg'       #grayimage is the folder that you wish to save the image
            #print(imgname)
            cv2.imwrite(imgname, img2)
            n += 1
    except KeyboardInterrupt:
        print('wait')
