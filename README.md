# Object Detection


## Project Abstract
This project addresses the problem of detecting object location in object recognition. We implements general classifiers with special feature detection algorithm to make a framework used to detect specific object locations in the image. Instead of using deep learning frame such as RCNN or CNN, we combine SIFT and BOW, Selective Search and general classifier models into a single framework to improve time cost/performance. More specifically, we use NBA baskerball team Lakers logo and a special toy as the target object, meanwhile, dividing our framework into three different parts: feature selection part, tuning classifier part and capturing location part.


## Project Introduction

<p align="center">
  <img src="flowchart.png" height="500" width="750"><br/>
  <h6 align="center">Figure 1: Flowchart</h6><br/>
</p><br/>

Our work is divided into three steps, which is illustrated in Figure 1:
* Step1 (Feature Selection): We use k pure images including Lakers logo or a special toy without any noise as input, then use SIFT algorithm to detect and describe local features in images (transform k pure images into k x m feature vectors). By applying kmeans on the collection of feature vectors, we generate a specific vector with c vocabularies.
* Step2 (Tuning Classifier): The input datasets in this part are n images including any objects. The image only containing Lakers logo or the target toy is marked as 1, otherwise 0. We still apply SIFT for generating n x m features, then transferring them into n x c sample datasets based on the vocabulary vector obtained from Step1, where n is sample size and c is the number of features. In the consideration of the best classifier in our framework, we consider several state-of-the-art techniques, including SVM, KNN, Logistic, Decision Tree, RandomForest, Adaboost and XGboost. We divide the sample datasets into training and testing, tuning each method and then choosing the best one for next step.
* Step3 (Capturing location): In this step, we aim at detecting the target location of any new photos. Unlike the first two steps, the images used in this step have no labels (may or may not have targets) and also contain a lot of noise. For a new photo, We generate a segmentation, which aims for a unique partitioning of the image through Selective Search. After that, the output  (1500 sub-graphs) is transferred into a 1500 x c matrix through SIFT and the vocabulary vector of Step 1. Then the best classifier obtained in Step2 can tell us which sub-image contains the target object. 


## Dataset
In the experiment of detecting NBA team logo, we google 40 pure photos containing Lakers logo without any noise for Step1, 500  photos only containing Lakers logo and 500 photes containing any team logo for Step2. 
In the expreiment of detecting toy, we take 100 pure photos containing the target toy for Step1, 500 pure photos containing the target and 500 without the target for Step2.
The target in each experiment is shown in Figure 2 and 3, respectively.

<p align="center">
  <img src="0.jpg" height="300" width="400"><br/>
  <h6 align="center">Figure 2: The target in the experiment of detecting logo</h6><br/>
</p><br/>

<p align="center">
  <img src="1.jpg" height="350" width="400"><br/>
  <h6 align="center">Figure 3: The target in the experiment of detecting the toy</h6><br/>
</p><br/>



## Methodology
* SIFT and BOW: A class generate feature vector for image or sub-image by SIFT and BOW. Scale-invariant feature transform (SIFT) is a feature detection algorithm in computer vision to detect and describe local features in images. We build Difference Of Gaussian Pyramid based on Gaussian Pyramid(O,L). After normalizing these DoG image, we can clearly see the features contained in each different image, and some features exist in different blurring degrees and scales. These features are the "stable" features that Sift is trying to extract. We found empirically that 128 SIFT feature points works well in practice. For BOW, it is a K-Means aggregate essentaly.
* Selective Search: Selective Search is a region proposal algorithm used in object detection. It is designed to be fast with a very high recall. It is based on computing hierarchical grouping of similar regions based on color, texture, size and shape compatibility.

## Evaluation
First, in the comparative experiment of Step2, we tune each classifier and compare them based on precision, recall and F1-Score. In practice, we prefer choosing the one with highest F1-Score rather than other metrices, because the F1 score is the harmonic average of the precision and recall.  
The comparative result is shown in Fig.4 and Fig.5:
<p align="center">
  <img src="result_lakers.png" height="150" width="400"><br/>
  <h6 align="center">Figure 4: The result in the comparative experiment of detecting team loge</h6><br/>
</p><br/>

<p align="center">
  <img src="result_toy.png" height="250" width="400"><br/>
  <h6 align="center">Figure 5: The result in the comparative experiment of detecting toy</h6><br/>
</p><br/>

Secondly, for a new photos in the Step3, we consider the potential location of the object based on the 'confidence' of each sub-graph labeled 1. For SVM, we interpret the 'confidence' as the distance of the sub-graph to the separating hyperplane. For others, it is the estimated probability of the sub-graph aligned for class 1.





## DEMO Implement


## Project Output

<p align="center">
  <img src="result.png" height="350" width="400"><br/>
  <h6 align="center">Figure 6: Output of Detecting Lakers Logo</h6><br/>
</p><br/>

<p align="center">
  <img src="result_toy.jpg" height="350" width="200"><br/>
  <h6 align="center">Figure 7: Output of Detecting toy</h6><br/>
</p><br/>

