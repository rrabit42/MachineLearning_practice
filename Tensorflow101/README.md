# 생활코딩 - Tensorflow101  
*원래 아이패드에다가 필기해서 굉장히 rough함*  

* 실습 환경 - [Google Colaboratory](https://colab.research.google.com/)  

## Basic  
* 인간의 판단 능력을 기계에게 위임 => AI  
* 이 강의에서 tensorflow로 해결하려는 것은 지도학습(회귀, 분류)  
*지도학습(machine learning 중 하나), 회귀-숫자 예측, 분류-범주형 데이터 예측*  

* machine learning algorithm을 이용해 문제 해결  
대표적으로 Decision Tree, Random Forest, KNN, SVM, **Neural Network**(우리가 사용할 알고리즘) 등  
<img width="801" alt="1" src="https://user-images.githubusercontent.com/46364778/104095329-9704ff00-52d9-11eb-8e8d-c9816ce9d9e9.png">  

## Neural Network = 인공신경망 = Deep Learning  
* 인간의 신경망 모방  
<img width="210" alt="2" src="https://user-images.githubusercontent.com/46364778/104095327-95d3d200-52d9-11eb-9a90-8367b4948fe5.PNG">  

* 우리는 Deep learning을 **library**를 이용해서 할 것이다.  
* library 종류에는 **tensorflow**, pytorch, caffe2, theano 등이 있다.  
```import tensorflow as tf```  

## 지도학습의 빅픽쳐  
1) 과거의 데이터를 준비한다.  
2) 모델의 구조를 만든다.  
3) 과거의 데이터로 모델을 학습(Fit)한다.  
4) 모델을 이용한다.  

## Pandas  
* 표를 다루는 도구, library  
* *tensorflow와는 무관하다*  
```import pandas as pd```  

---
## 1. 데이터 준비하기  
1. 데이터를 불러온다.  
2. **종속변수**와 **독립변수**로 분리한다. (따로 분리 필요, column으로 선택)  

### 실습을 통해 배울 도구들  
* **파일 읽어오기**: ```read_csv('./경로/파일명.csv')```  
① csv는 콤마(,)로 데이터가 분리되어 있다 => 이걸 excel에서 읽어들이면 표의 형태로 보여줌  

* **모양 확인하기**: ```데이터.shape```  
② 파일들이 변수에 잘 담겨졌는지 확인 => 그 중 모양으로 확인(row 개수=순수 data 개수, column 개수)

* **칼럼 선택하기**: ```데이터[['칼럼명1', '칼럼명2', '칼럼명3']]```  
④ ```독립변수 = 데이터[  ]``` ```종속변수 = 데이터[  ]``` 로 분리 => 확인은 shape으로!  

* **칼럼 이름 출력하기**: ```데이터.columns```  
③ 칼럼 선택하기 전에 이름 확인  

* **맨 위 5개 관측치 출력하기**: ```데이터.head()```  
테이블 형태로 보여준다.  

## 2. ML 모델 설계  
* 데이터가 준비되었으니 ML 모델을 만들어 보자!
```
X = tf.keras.layers.Input(shape=[1])  // 1은 독립변수의 개수(column 1개) ex) 독립 = 데이터[['온도']]
Y = tf.keras.layers.Dense(1)(X)       // 1은 종속변수의 개수(column 1개)
model = tf.keras.models.Model(X,Y)
model.compile(loss='mse')
```  
* NN으로 구성 => 매우 간단한 code, 뉴런이 1개, 학습 전(아기)  

## 3. 학습(fit)  
* 데이터로 모델을 학습(fit) 시키자!  
```model.fit(독립, 종속, epochs=1000)```  
* epoch: 전체 데이터를 몇번 반복하여 학습시킬건지(현재는 1000번)  
* 이 코드의 결과로 우리는 충분히 학습된 ```model```을 get하게 된다.  
* verbose=0: 출력을 끄고 학습시키기(화면에 loss 등 표시 X)  
```model.fit(독립, 종속, epochs=1000, verbose=0```  

#### Loss란?  
* 학습이 얼마나 진행되었는지를 나타내는 지표  
* 즉, 학습이 끝날 때 마다 그 시점의 모델이 얼마나 정답에 가까이 맞추고 있는지  
* loss가 **0**에 가까울 수록 정확도가 높다는 뜻(학습이 잘된 모델)  
* 종속변수와 예측한 값 비교 => (예측-결과)^2의 average가 loss  
**매 Epoch마다 loss를 확인하자!**  

## 4. 모델을 이용  
```print("Predictions: ", model.predict([[15]]))    // 15는 우리가 준비한 독립변수```  
* 이 함수로 독립변수로 예측한 종속변수를 출력하게 된다.  

---  

































