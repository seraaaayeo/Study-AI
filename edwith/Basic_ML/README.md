# Basic Machine Learning

* Hypothesis(가설함수) = Wx + b
* Cost function: 가설함수와 실제 데이터의 차의 제곱의 평균
* 머신러닝의 핵심은 cost가 최소화되는 W(가중치)를 찾는 과정이다.
    - Gradient descent

## Multi-variable Linear Regression
* 많은 변수를 표현하기 위해 Matrix를 사용한다.
* w1x1 + w2x2 + w3x3 +...
    - (x1 x2 x3)와  (w1 w2 w3)의 dot product
    - H(X) = XW
* 데이터의 건수 = instance
    - ![image](https://user-images.githubusercontent.com/53554014/89309773-ccfce200-d6ae-11ea-97ae-028da231da79.png)
    - 인스턴스(데이터) 5개, feature 3개
* matrix 연산은 행과 열의 개수가 매우 중요
    - 앞 행렬의 열과 뒷 행렬이 행이 일치해야 함.
    - (5X3) dot (3X1) = (5X1)
    - 즉 feature(변수)의 수와 weight의 수는 일치해야 한다.

## Logistic Regression
* **여기 설명 뭐라는지 잘 모르겟음... 복습할 것!**
* Binary Classification: 0(positive) 1(negative)
* Logistic vs Linear
    - Logistic: discrete(counted)
    > 신발사이즈
    - Linear: Continous(mesured)
    > 시간, 몸무게
* Logistic function(sigmoid)
    - logistic function의 출력값은 0과 1 사이이다.
    - 

## Softmax Regression
* hypothesis를 0과 1사이의 값으로 압축하면 좋겠다..!
* sigmoid(logistic) = 1 / 1+e^-2
* sigmoid 함수를 통과하면 0과 1사이의 출력값을 가진다.
* Logistic?
    - 두 데이터(혹은 그 이상)를 분류하기 위해 사용.
* Multinomial classificaion
    - 여러개의 binary classfication을 결합하여 찾을 수 있다.
* softmax
    - 각 클래스의 출력값은 **확률**이다.
    > 0과 1사이의 값이며, 각각의 출력값의 합은 1
    - one-hot encoding: 가장 큰 값을 1로, 나머지 값은 0으로 예측한다.
* Cross entropy