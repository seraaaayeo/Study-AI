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

1. 지도학습
    * regression
    * classification
2. 비지도학습
    * clustering
3, 강화학습

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