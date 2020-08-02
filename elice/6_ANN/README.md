## 선형회귀, 비선형회귀 모델 구현하기
**NOTE** 해당 코드는 코랩 환경에서 테스트되었음.


## 다중 선형 회귀분석 모델 구현하기
**NOTE** 해당 코드는 코랩 환경에서 테스트되었음.

### 소개
* Facebook, TV, Newspaper의 값에 따라 예상되는 Sales값을 예측하는 모델 생성.
* MSE를 1 이하로 낮추는 모델 생성.

## 네이버 영화평 감정분석 모델 구현하기
[데이터 출처: Naver Sentiment Movie Corpus v1.0](https://github.com/e9t/nsmc)

### 소개
* 입력 받은 영화평이 긍정/부정일 확률을 구하는 감정분석기 구현을 위해 인공신경망 모델을 구현하여 학습시킨다.
* **Note** [Naive bayes] 디렉터리의 감정분석은 나이브 베이즈 분류기를 학습시켰다.
* 원본 데이터는 총 10만개의 부정 리뷰, 10만개의 긍정 리뷰로 구성되어 있음.
### 학습 데이터
* 랜덤하게 추출한 100개의 데이터 사용.

## Fashion-MNIST 데이터 분류하기
[데이터 출처 : zalando research](https://research.zalando.com/welcome/mission/research-projects/fashion-mnist/)

### 소개
* Fashion-MNIST 데이터는 의류, 가방, 신발 등의 패션 이미지들의 데이터셋으로 6만 개의 학습 데이터셋과 만 개의 테스트 데이터셋으로 이루어져 있다.
* 각 이미지들은 28X28, 흑백 이미지
* 총 10개 클래스로 분류됨
  - Label0: 반팔
  - Label1: 바지
  - Label2: 맨투맨
  - Label3: 나시
  - Label4: 외투
  - Label5: 샌들
  - Label6: 티셔츠
  - Label7: 운동화
  - Label8: 에코백
  - Label9: 구두
### 학습 데이터
* 사용한 데이터는 모델 학습을 위해 (28X28) 크기의 다차원 데이터를 1차원 배열로 전처리한 데이터이다.
* 4000개의 학습 데이터와 1000개의 테스트 데이터를 랜덤으로 추출해서 사용.
