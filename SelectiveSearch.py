import cv2 as cv

maxReg = 1200# max # of bounding box

def regionGenerate(img):
    rects = imgSegment(img)
    res = []
    for i, rect in enumerate(rects):
        if i < maxReg:
            x,y,w,h = rect
            if (w > 64 and h > 96):
                bb = img[y:y+h,x:x+w]
                temp = ([x,y,w,h],bb)
                res.append(temp)
            else:
                continue
        else:
            break
    return res


def imgSegment(img):
    # multithreads of opencv to speed-up
    cv.setUseOptimized(True);
    cv.setNumThreads(4)
    # create Selective Search Segmentation Object using default parameters
    ss = cv.ximgproc.segmentation.createSelectiveSearchSegmentation()
    fill = cv.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyFill()
    color = cv.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyColor()
    texture = cv.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyTexture()
    strategy = cv.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyMultiple()
    weight = 1.0
    #strategy.addStrategy(fill, weight)
    strategy.addStrategy(color, weight)
    strategy.addStrategy(texture, weight)
    ss.addStrategy(strategy)
    ss.setBaseImage(img)
    # high recall but slow Selective Search method
    ss.switchToSelectiveSearchQuality()

    # fast but low recall Selective Search method
    #ss.switchToSelectiveSearchFast()


    # run selective search segmentation on input image
    rects = ss.process()
    return rects

if __name__ == '__main__':
    filename = '4.jpg'
    img = cv.imread(filename)
    print(img.shape)
    regions = imgSegment(img)
    i = 0
    for i, rect in enumerate(regions):
        if i < maxReg:
            x,y,w,h = rect
            if w < 100 or h < 100:
                continue
            bb = img[y:y+h,x:x+w]
            a = cv.imwrite(('bbs_toy/' + str(i) + '.jpg'),bb)
            print(a)
