# Numpy 기초
* 행렬이 왜 필요한가? -> 머신러닝에서 대부분의 데이터는 행렬로 표현됨.
* 행렬을 어떻게 표현할까? -> numpy array

## Numpy Array
```
import numpy as np
A = np.array([[1, 2], [3, 4]])
```
* 산술연산: 큰 데이터를 다룰 때 편의를 위해 만들어진 기능이다.
* 비교연산
* 논리연산
    ```
    a = np.array([1, 1, 0, 0], dtype=bool) //TTFF
    ```
* norm(normalization)
    - 원소의 합이 1이 되도록 한다.
    - 벡터를 벡터 원소의 합으로 나눈다.
    - A = A/np.sum(A)
* reduction
    - argmin/argmax: 최소/최대값의 **인덱스**를 반환
    - min/max: 최소/최대값을 반환
* logical reductions
    - all: array내의 모든 값이 True인가?
    - any: array내의 값이 하나라도 True인가?
* Statistical reductions
    - mean
    - median
    - std: standard devation, 표준편차
* dot product
    ```
    np.dot(x, y)
    ```
* 전치행렬: 행과 열을 뒤집은 행렬.
    ```
    A = np.array([][])
    
    print(A.transpose())
    print(A.T)
    ```
* 역행렬: Numpy의 선형대수학 세부패키지 linalg를 사용.
    ```
    np.linalg.inv()
    ```

## Numpy 그림 그리기
1. 캔버스
    * 그래프 공간(캔버스) 설정
    ```
    xrange=[1, 3] //x축 범위
    yrange=[2, 4] //y축 범위
    ```
2. 그림 그리기
    * 어떤 함수 f와 매우 작은 숫자 threshold에 대해
    * 캔버스 내에 점 P = (x, y)를 임의로 생성
    * f(P) < threshold라면 점을 찍는다.
    * 이를 100,000회 반복한다.
3. 원 그리기
    * (0, 0)이 중심이고 반지름 1인 원을 그리는 방정식은 다음과 같다.
    ```
    x^2+y^2=1
    ```
    * 원을 그리는 함수를 circle이라 정의하자.
    * 정확히 원 위에 있는 점들에 대해 circle(P)는 0이어야 한다.
    
***

# 기계학습

1. 지도학습(supervised)
    * regression
    * classification
2. 비지도학습(unsupervised)
    * clustering
3, 강화학습(reinforced)

## Linear regression

### 모델의 학습 목표
각 데이터의 실제 값과 모델이 예측하는 값을 최소한으로 한다.

### Loss function
* 실제값과 예측값의 차이의 제곱의 합. MSE(Mean Square Error)등으로 배워왔다.
```
Y ~ 기울기*X + 절편
```
* 전체 모델의 차이: sum{y_i (b0 * x_i + b1))^2}

## Multi-linear regression
* 페이스북 광고 뿐만 아니라 인스타그램, TV광고도 하기로 결정했다. 이 때 각 매체가 얼마나 효율적인지 알아내야 한다. 
    * => 다중선형회귀분석 사용
```
Y ~ B0X1 + B1X2 + B2X3 + B3
```
* X1: facebook, X2: Instagram, X3: TV
* B3: 하나도 광고를 하지 않아도 얻는 이익
* 이 차이를 최소로 하는 B0, B1, B2, B3를 구한다.

## 다항식 회귀 분석
* 입력값과 출력값 사이의 모델이 직선이 아닌 곡선 형태를 띈다.
* 판매량과 광고비의 관계를 2차식으로 표현해 보자.
    - 광고비의 제곱을 새로운 데이터로 치환하여 다중회귀분석과 동일하게 사용한다.
    ```
    Y = B0X1 + B1X2 + B2
    ```
    - X1 = X^2(광고비의 제곱)
    - X2 = X(광고비)
    - Y = 판매량
    
***

# 확률과 나이브베이즈(Naive Bayesian)

## 확률론
### 빈도주의자 vs 베이즈주의자
* 빈도주의자: 동전을 충분히 크게 던졌을 때 앞면이 50%, 뒷면이 50% 나온다. 
    - => 사건이 무한히 크게 발생할 때 사건의 확률을 정확히 정할 수 있다.
* 베이즈주의자: 동전 던지기의 결과가 앞면이 나올 것이라는 **확신/믿음**이 50%이다.
    - => 동전을 던지기 전에, 그 동전에 대한 정보를 알고있다면 사전 정보를 확률에 반영할 수 있다.
* 고등과정에서 배우는 확률은 빈도주의자적인 확률이다.

## 베이즈법칙
```
P(A|X) = P(X|A)P(A) / P(X)
```
* P(X|A): P(X) given A, A라는 조건이 있을 때 X일 확률.
* 예제: 암 검사 키트
* <img src=https://user-images.githubusercontent.com/53554014/88652777-38224380-d106-11ea-98d0-080164e361e0.jpg width=70% height=70% title="확률 계산"></src>

## 나이브 베이즈 분류기
* 분류(classification): 주어진 데이터가 어떤 클래스에 속하는지 알아내는 작업
* <img src=https://user-images.githubusercontent.com/53554014/88652598-04471e00-d106-11ea-8ef8-504848887b35.png width=70% height=70% title="기계학습"></src>
* 두 사건이 있을 때 P(A|X)와 P(B|X) 중 어느 것이 더 큰가?
```
P(A|X) : P(B|X) = P(X|A)P(A) / P(X) : P(X|B)P(B) / P(X)
```
* ![image](https://user-images.githubusercontent.com/53554014/88670176-b6d7aa80-d11f-11ea-87d4-acd61457112a.png)
* P(X|B): likelihood
* P(B): 사전(prior)확률
* P(B|X): 사후(posterior)확률

### 실습: 나이브 베이즈를 이용한 감정분류 모델 학습
* 긍정적인 문서 2000개와 부정적인 문서 2000개가 있다. *마음이 따뜻해지는 최고의 영화*와 같은 특정 단어가 입력되었을 때, 해당 문장이 긍정적인지 부정적인지 계산.
    1. Traning: 각각의 문서 셋들에서 나오는 단어의 빈도 수 측정
    2. 부정과 긍정 모델에서 '최고의'와 같이 특정 단어가 단어가 나올 확률 계산.
    3. 같은 방법으로 문장을 이루는 모든 단어의 확률을 계산
    4. 나이브 베이즈 계산
    5. 결론적으로 해당 문장이 긍정인지 부정인지 계산할 수 있다.
* 어떤 문장의 단어가 모델에 존재하지 않을 경우?
    - 해당 단어를 0은 아니지만 아주아주 작은 수의 상수로 둔다.(ex.0.000000000001)
    
***

# 비지도학습(Unsupervised Learning)

## 지도학습 VS 비지도학습
* 지도학습: 얻고자 하는 답으로 구성된 데이터
  - 강아지 고양이 분류기
  - 강아지 데이터는 강아지, 고양이 데이터는 고양이임을 알려주며 학습.
* 비지도학습: 답이 정해져 있지 않은 데이터에서 숨겨진 구조를 파악
  - 비슷한 데이터끼리 모은다.
  - 예를 들어 강아지는 어떤 특성(구조)을 가지고 있고 고양이는 어떤 특성을 가지고 있기 때문에 군집화가 가능.

## 비지도학습: Clustering
* 클러스터: 비슷한 데이터의 그룹

### Hard Clustering: 데이터 포인터들은 비슷한 것들끼리 뭉쳐있다.
* 예시
  * 고양이 사진은 100% 고양이이고 0% 강아지이다.
* 종류
  * Hierarchical Clustering
  * **K-Means**
  * DBSCAN
  * OPTICS

### Soft Clustering: 한 개의 데이터 포인트는 숨겨진 클러스터들의 결합이다.
* 각각의 데이터 포인트가 여러 클러스터의 **확률**적 집합이다.
* 예시
  * "사이언스 픽션" 장르의 책은 60% 과학이고 35% 판타지, 5% 역사이다.
  * "역사 판타치" 장르의 책은 55% 판타지이고 45% 역사이다.
* 종류
  * Topic Models
  * FCM
  * **Gaussian Mixture Models(EM)**
  * Soft K-Means
  
## Hard Clustering
### K 결정하기
1. 눈으로 확인
2. 모델이 데이터를 얼마나 잘 설명하는가

### 고려할 것들
* 데이터의 특성
  * 어떻게 만들어진 데이터인가?
  * 데이터 포인트 외 다른 feature
* 분석 결과로 얻고자 하는 것
  * 예: 고양이vs개 분류, 사람들의 행동 분석, 가격 대비 효율성 분석 등

### 차원축소
* ??를 모를 때 K를 결정하기 위한 방법.
#### PCA(Principal Component Analysis)
* 사용 이유
  - 고차원의 데이터를 저차원으로 줄이기 위해.(예: 13차원의 특성을 갖는 데이터를 2차원으로 시각화)
  - **데이터 정제**
* 데이터를 한 개의 축으로 투사했을 때 분산이 가장 높아지도록 데이터를 조정한다.

### K-means
1. Centroid: 각 클러스터의 중심
2. Distance: 중심과 데이터 포인트와의 거리

* 각 클러스터와 거리 차이의 분산을 최소화하는 방식으로 동작.
* 알고리즘 flow
  - K값에 따라 초기 중심점을 임의로 설정.(K=2이면 중심 2개)
  - 각각의 데이터 포인트에 대해 가까운 중심점 계산하여 해당 클러스터에 할당
  - 1차적으로 만들어진 클러스터에 대해 중심점을 다시 계산.
    * 중심점: 해당 클러스터 내 데이터 포인터 위치의 무게중심(평균)
  - 각각의 데이터 포인트에 대해 새로운 중심점과의 거리를 계산하여 클러스터에 할당
  - 중심점 업데이트와 클러스터 할당을 반복하다가 어떠한 데이터 포인트에 대해서도 새로운 할당이 일어나지 않을 때 알고리즘 종료
* K means는 시작점에 민감한 알고리즘이다.
  - 시작점을 어떻게 설정할 것인가?

***

# 퍼셉트론(Perceptron)
* 인공신경망 시스템. 입력이 있을 때, 어떤 함수에 따라서 activation이 일어나고 출력을 하는 시스템.
* 퍼셉트론 -> 인공 신경망 -> 인공지능 으로 진행된다.
* 구조
    ```
    출력 = activation(w1x1 + w2x2 + B) #w1, w2 = 가중치, B = bias
    ```
    - activation function(step function) : 특정 조건을 만족하면 1을 출력(활성화). 이외에는 0을 출력.
    - 퍼셉트론은 논리 게이트의 역할을 한다.
* 논리게이트
    - AND
    - NAND(Not-And): 0/0 -> 1, 1/0 -> 1, 0/1 -> 1, 1/1 -> 0
* 단층 퍼셉트론: 선형 분류기
* 다층 퍼셉트론: 비선형 분류기
    - 퍼셉트론을 여러개 쌓아서 만든다.
    - OR와 NAND 게이트를 결합하여 비선형인 XOR 게이트를 만들어 구역을 분리할 수 있다.
    - 여기서 NAND 게이트와 OR 게이트를 hidden layer라고 한다.
    - 이처럼 여러 개의 layer를 이용하면 세분화된 분리가 가능하다.
    - **Hidden layer가 3층 이상이면 Deep Neural Network(DNN, 딥러닝)이라 한다.**
    
***

# Tensorflow
```
conda install tensorflow
```
* 전세계에서 가장 많이 쓰이는 딥러닝 프레임워크.
* Tensor = 다차원 array = Data
    - 딥러닝에서 텐서는 다차원 배열로 나타내는 데이터이다.
    - ex.RGB 이미지는 3차원 배열로 나타나는 텐서.
* Flow: 데이터의 흐름. 
    - 입력 tensor가 있을 때, operation node에 의해 결과값이 나오는 텐서의 흐름을 tensorflow
    - 텐서플로우에서 계산은 데이터 플로우 그래프로 수행한다.
    * 모델을 만든다 = graph를 만든다.

## Tensorflow 사용

### 상수 선언하기
```
import tensorflow as tf

tensor_constant = tf.constant(value, dtype=None, shape=None, name=None)
```
* value: 반환되는 상수값
* shape: tensor의 차원(optional)
* dtype: 반환되는 tensor 타입(optional_)
* name: 상수 이름(optional)
* 예
    - 모든 원소 값이 0인 tensor 생성
    ```
    tensor_zero = tf.zeros(shape, dtype=tf.float32, name=None)
    shape에는 차원의 튜플을 넣는다. 예를 들어 2X2 텐서를 생성하고 싶으면 (2,2) 입력.
    ```
    - 모든 원소 값이 1인 tensor 생성
    ```
    tensor_one = tf.ones(shape, dtype=tf.float32, name=None)
    ```

### 시퀀스 선언하기
* start에서 stop까지 증가하는 num 개수 데이터
    ```
    import tensorflow as tf

    tensor_seq = tf.linspace(start, stop, num, name=None)
    ```
    * start: 시작 값
    * stop: 끝 값
    * num: 생성할 데이터 개수
    * name: 시퀀스 이름
* start에서 stop까지 delta씩 증가하는 데이터
    ```
    tensor_seq2 = tf.range(start, limit=None, delta=None, name=None)
    ```
    * limit: 끝 값
    * delta: 증가량

### 난수 선언하기
* 보통 난수를 통해서 모델을 초기화한다.
* 주로 정규분포 사용.
```
import tensorflow as tf

#정규분포 생성
tensor_ = tf.random.normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name='normal')

#균등분포 생성
tensor2_ = tf.random.uniform(shape, minval=0, maxval=None, dtype=tf.float32, seed=None, name='uniform')
```
* seed: seed값에서 특정한 알고리즘을 통해 순차적으로 난수 생성.
* 균등분포: 일정한 범위 내의 확률은 모두 같다.

### 변수 선언하기
```
#정규분포 생성
tensor_val = tf.Variable(value, name=None)

#일반적인 퍼셉트론의 가중치와 bias 생성
weight = tf.Variable(10)
bias = tf.Variable(tf.random.normal([10, 10]))
```
* 모델을 트레이닝해서 얻고싶은 값이 무엇이냐
* 초기화된 변수에 트레이닝을 통한 값이 assign 가능.
* 가중치를 variable로 선언하면 모델이 training하면서 가중치가 update 가능.

### Tensor 연산자
* 단항 연산자
    * tf.negative(x) #숫자 
    * tf.logical_not(x) #!x, Boolean
    * tf.abs(X)
* 이항 연산자
    * add, subtract, multiply
    * 나누기: truediv
    * 몫: mod
    * 제곱: pow
* tensor에 .numpy()를 붙이면 넘파이 배열로 변환.

## Tensorflow로 딥러닝 구현하기
### 데이터 용어 정리
* Epoch: 한 번의 epoch는 전체 데이터 셋에 대해 한 번 학습을 완료한 상태를 뜻한다.
* Batch: 나눠진 데이터 셋. iteration은 epoch를 나누어서 실행하는 횟수를 뜻한다. 
    - 데이터셋의 양이 굉장히 많기 때문에 데이터셋을 여러 작은 데이터로 쪼개서 학습을 진행하며, 이 작은 데이터를 batch, 쪼갠 데이터의 양을 batch size라 한다.
    - 예: 1000개의 데이터를 batch size 100으로 학습을 시킬 경우, 1 epoch는 10 iteration

### 데이터 준비하기
* Dataset API를 사용하여 데이터 준비.
```
data = np.random.sample((100, 2)) #shape이 (100,2)
labels = np.random.sample((100, 1))

#numpy array로부터 데이터셋 생성
dataset = tf.data.Dataset.from_tensor_slices((data, lables))
dataset = dataset.batch(32)
```

### Keras
딥러닝 모델을 만들기 위한 고수준의 API 요소를 제공하는 모델 수준의 라이브러리

* Keras API
    - multi-backend
    - cpu, gpu 모두 구동

### 딥러닝 모델 생성 함수
* 인공신경망 Sequential 모델을 만들기 위한 함수
    ```
    tf.keras.models.Sequential()
    ```
* 신경망 모델의 layer 구성에 필요한 함수
    ```
    tf.keras.layers.Dense(units, activation)
    ```
    - unit: 레이어 안의 Node 수
    - activation: 적용할 활성함수
* 예시 **tf.keras.layers를 추가하여 hidden layer를 쌓는다.**
    ```
    model = tf.keras.models.Sequential([
     tf.keras.layers.Dense(10, input_dim=2, activation='sigmoid'),
     tf.keras.layers.Dense(10, activation='sigmoid'),
     tf.keras.layers.Dense(1, activation='sigmoid'),
    ]
    ```
    - input node는 2개
    - 첫 번째 layer의 node는 10개
    - 두 번째 layer의 node는 10개
    - 세 번째 layer의 node는 1개.
    - 각 unit은 weight로 연결된다.
* 모델 학습하기
    - loss function, optimizer(loss function을 줄이는 전략이 무엇인가) 필요.
    ```
    #모델 구성
    model.compile(loss='mean_squared_error', optimizer='SGD') #학습 방법 설정
    model.fit(dataset, epochs=100) #모델 학습
    
    #데이터셋 생성
    dataset_test = tf.Dataset.from_tensor_slices((data_test, labels_test))
    dataset_test = dataset_test.batch(32)
    
    #모델 성능 평가
    model.evaluate(dataset_test)
    predicted_labels_test = model.predict(data_test) #학습된 모델로 예측값 생성
    ```
    - SGD: Statistic(??) Gradient Descent

### 분류모델: 네이버 영화 댓글 평점 데이터
* Data -> Preprocessing -> Input layer
    * Preprocessing: Tokenizing&Bow, Encoding
* softmax function을 통해 긍정인지 부정인지 경향성을 볼 수 있다.
* 최적화 기법: 선형회귀에서는 GD가 쓰이지만, 인공신경망에서는 다양한 형태의 GD 기법들이 사용된다.
    - **SGD**, **Adam**, Momentum, AdaGrad, RMSProp 등

***

## 아나콘다와 텐서플로우
* [아나콘다 가상환경 생성](https://zvi975.tistory.com/65)

























