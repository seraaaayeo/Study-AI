<h1 align="center">AI fundamental stage</h1>
<p align="center">
    <img alt="algorithm image" src="https://user-images.githubusercontent.com/53554014/89105614-82385b80-d45d-11ea-9221-91d367bb1059.jpg" width=50% height=50% />
</p>
<p align="center">
  :computer:Data Science, Machine Learning, Deep Learning 이론 구현하기:computer:
</p>

* * *

## Description
여러 온라인 강좌를 통해 데이터 분석, 머신러닝/딥러닝 이론을 구현하고 연습하기

## Reference: online courses
* [elice](https://elice.io/)
* [edwith](https://www.edwith.org/)
* [Coursera : Machine Learning class by Andrew Ng](https://www.coursera.org/learn/machine-learning/home/welcome)

## Project list
|  <center>Number</center> |  <center>About</center> |  <center>Concept</center> |  <center>Description</center> |  <center>Shortcut</center> |    
|:--------|:--------:|:--------:|:--------:|:--------:|
|**1** |  <center>News data word cloud</center> | <center>Crawling, Wordcloud</center> |  <center>네이버 뉴스 데이터를 크롤링하고, 수집한 데이터를 정제하여 워드클라우드 만들기</center> | <center>[crawling&wc](https://github.com/seraaaayeo/Study-AI/tree/master/crawling%26wc)</center> |
|**2** | <center>Sentiment Classifier</center> |  <center>Naive Bayes</center> | <center>나이브 베이즈 분류기를 학습시켜 네이버 영화 리뷰 감정 분류하기</center> | <center[Naive Bayes Classifier](https://github.com/seraaaayeo/Study-AI/blob/master/elice/3_Naive_Bayes/Naive_Bayesian_(5)_Sentiment%20Classifier.ipynb)></center> |
|| <center>Sentiment Classifier</center> |  <center>DNN</center> | <center>tensorflow를 이용하여 모델을 쌓고 학습시켜 네이버 영화 리뷰 감정 분류하기</center> | <center>[DNN Classifier](https://github.com/seraaaayeo/Study-AI/blob/master/elice/6_ANN/(4)_Sentiment%20calssifier.ipynb)</center> |
|**3** | <center>Classification</center> |  <center>PCA, Kmeans clustering</center> | <center>sklearn을 이용하여 13차원 와인 데이터를 pac 차원축소하고 K-means 클러스터링으로 분류하기</center> | <center>[Wine data clustering](https://github.com/seraaaayeo/Study-AI/blob/master/elice/4_KMeans_Clustering/test_KMeans.ipynb)</center> |
|**4** | <center>Classifier</center> |  <center>Perceptron</center> | <center>퍼셉트론 선형 분류기로 sklearn iris 데이터 분류하기</center> | <center>[iris data classification](https://github.com/seraaaayeo/Study-AI/blob/master/elice/5_Perceptron/perceptron_iris.ipynb)</center> |
|**5** | <center>MNIST</center> |  <center>CNN, Data augmentation</center> | <center>여러가지 기법을 적용하여 최적의 MNIST CNN 만들기</center> | <center>[Best CNN-MNIST](https://github.com/seraaaayeo/Study-AI/blob/master/edwith/Basic_DL/CNN_MNIST_best.ipynb)</center> |
| | <center>MNIST</center> |  <center>CNN</center> | <center>데이터셋을 직접 준비하여 MNIST CNN 만들기</center> | <center>[MNIST-Rock scissor papper](https://github.com/seraaaayeo/Study-AI/blob/master/aiffel/prj1-rock_scissor_papper.ipynb)</center> |
| | <center>MNIST</center> |  <center>DNN, Fashion MNIST</center> | <center>tensorflow를 이용하여 DNN 모델을 쌓고 Fashion MNIST 데이터셋 학습하기</center> | <center>[DNN-Fashion MNIST](https://github.com/seraaaayeo/Study-AI/blob/master/elice/6_ANN/(5)_Fashion-MNIST.ipynb)</center> |


## Contents
### Base
* Library
    * [Numpy](https://github.com/seraaaayeo/Study-AI/tree/master/elice/1_Numpy)

### Machine Learning 
#### 선형대수학 방법
* [Gradient Descent](https://github.com/seraaaayeo/Study-AI/blob/master/edwith/Gradient%20descent/gradient_descent_test1.ipynb)
* Linear regression
    * [Linear regression]()
    * [Multi-variable Linear regression](https://github.com/seraaaayeo/Study-AI/blob/master/elice/2_Regression/Linear_regression_(2)_multi%20linear%20regression.ipynb)
    * [Poly Linear regression](https://github.com/seraaaayeo/Study-AI/blob/master/elice/2_Regression/Linear_Regression_(3)_test_%EB%8B%A4%ED%95%AD%EC%8B%9D%20%ED%9A%8C%EA%B7%80%20%EB%B6%84%EC%84%9D.ipynb)

#### 통계학적 방법
* Naive bayes
    * [암 검사 키트](https://github.com/seraaaayeo/Study-AI/blob/master/elice/3_Naive_Bayes/Naive_Bayesian_(2)_%EC%95%94%20%EA%B2%80%EC%82%AC%20%ED%82%A4%ED%8A%B8.ipynb)
    * [영화 감정 분석](https://github.com/seraaaayeo/Study-AI/blob/master/elice/3_Naive_Bayes/Naive_Bayesian_(5)_Sentiment%20Classifier.ipynb)

#### with Library
* Linear Regression
    * [Linear regression - sklearn](https://github.com/seraaaayeo/Study-AI/blob/master/elice/2_Regression/Linear_regression_(1)_basic.ipynb)
    * [Linear regression - tensorflow](https://github.com/seraaaayeo/Study-AI/blob/master/elice/6_ANN/(1)_Linear_Regression.ipynb)
    * [Nonlinear regression](https://github.com/seraaaayeo/Study-AI/blob/master/elice/6_ANN/(2)_Nonlinear_Regression.ipynb)
    * [Multi-variable Linear regression](https://github.com/seraaaayeo/Study-AI/blob/master/elice/6_ANN/(3)_Multi_linear_regression.ipynb)
* Logistic Regression
    * [Softmax function](https://github.com/seraaaayeo/Study-AI/blob/master/edwith/Basic_ML/Softmax_classificatoin.ipynb)
* Clustering
    * [PCA](https://github.com/seraaaayeo/Study-AI/blob/master/elice/4_KMeans_Clustering/Clustering_(1)_PCA.ipynb)
    * [K means](https://github.com/seraaaayeo/Study-AI/blob/master/elice/4_KMeans_Clustering/Clustering_(2)_KMeans.ipynb)
* 

### Deep Learning
* CNN
    * [CNN 모델 구축하는 방법](https://github.com/seraaaayeo/Study-AI/tree/master/edwith/Basic_DL)


## Stack
* Python 3.7
* Tensorflow 2.2

## Pre-requisties
|  <center>Requirement</center> |  <center>Description</center> |  
|:--------|:--------:|
|**Jupyter notebook** | <center>We use jupyter nootbook</center> |
|**Colab** | <center>If you don't have GPU, use colab for GPU</center> |
|**Git** | <center>We follow the Github flow</center> |
