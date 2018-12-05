from SelectiveSearch import regionGenerate
#from ClassifierModel import TrainingModel
from FeatureCompute import FeatureCompute
import matplotlib.pyplot as plt
import ClassifierModel as CM

import numpy as np
import cv2 as cv
import datetime
import os
import operator as op

def nms():
    pass


if __name__ == '__main__':
    while (1):
        object = input("detect object:")
        if (op.eq(object, 'Lakers') == 1):
            para = 'Lakers'
        elif (op.eq(object, 'Toy') == 1):
            para = 'Toy'
        elif (op.eq(object, 'Exit') == 1):
            print("Exit")
            break;
        else:
            print("Illegal input")
            continue;
        generator = FeatureCompute(300,70)
        dir = os.getcwd() + '/TestData/'+ para
        for count in range(len(os.listdir(dir))-1):
            filename = dir + '/' + str(count) + '.jpg'
            img = cv.imread(filename)
            print(filename)
            cv.imshow('show', img)
            cv.waitKey(0)
            regions = regionGenerate(img)
            data = np.float32([]).reshape(0,generator.wordCnt)
            for rect in regions:
                phi = generator.generatePhi(rect[1], para)
                data = np.append(data, phi, axis=0)
            ModelName = 'SVM.sav' # temporarily fixed
            # please choose models listed in ClassifierModel
            data = np.mat(data)
            TM = CM.TrainingModel()
            # print(data.shape)
            result = TM.Predict(data, ModelName)
            # result is a one dimension vector
            i = np.argmax(result)
            location = regions[i][0]
            cv.rectangle(img, (location[0],location[1]),
                         (location[0] + location[2],location[1] + location[3]),
                         (0,255,0), 1)
            cv.imshow('show', img)
            cv.waitKey(0)
            cv.imwrite(('res' + str(count) + '.jpg'), img)
        cv.destroyAllWindows()
