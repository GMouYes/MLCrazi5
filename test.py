import cv2 as cv
import os
import numpy as np
from ClassifierModel import TrainingModel
from FeatureCompute import FeatureCompute

if __name__ == '__main__':
    classifier = TrainingModel()
    generator = FeatureCompute(300,70)
    dir = os.getcwd() + '/bbs_toy'
    data = np.float32([]).reshape(0,generator.wordCnt)
    for count in range(1000):   #len(os.listdir(dir)) - 1):
        filename = dir + '/' + str(count) + '.jpg'
        print(filename)
        img = cv.imread(filename)
        # if img.shape[0] < 100 or img.shape[1] < 100:
        #     continue
        if img is None:
            continue
        phi = generator.generatePhi(img, 'Toy')
        data = np.append(data, phi, axis=0)
    data = np.mat(data)
    print(data.shape)
    print(type(data))
    np.save('why5.npy', data)
    response = classifier.Predict(data, 'SVM.sav')
    print(response)
    print(type(response))
    i = np.argmax(response)
    print(i)
    # print(data.shape)
    # print(type(data))
    # a = np.save('tes.npy', data)
