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
+) csv를 다운 받아서 colab에 업로드 => 경로 복사해서 사용해도 됨  

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

## 깨알 통계 지식  
* **이상값**(다른 data에 비해 지나치게 높거나 낮은 값)으로 인해 **'평균'** 의 대표성이 무너지면  
그 대안으로 사용하는 값이 **'중앙값'**  

## 수식과 퍼셉트론  
```
X = tf.keras.layers.Input(shape=[13])   // 13개의 입력층으로 구성됨
Y = tf.keras.layers.Dense(1)(X)         // 1개의 출력층
model = tf.keras.models.Model(X,Y)
model.compile(loss='mse')
```

<img width="375" alt="3" src="https://user-images.githubusercontent.com/46364778/104096143-0da3fb80-52de-11eb-841d-ea0949b31381.PNG">  

* ```y = w1x1 + w2x2 + ... + w13x13 + b``` : **뉴런 1개(perceptron)**  
* ```w1, w2, ... w13``` : **가중치(weight)**  
* ```b``` : **편향(bias)**  
* Dense layer는 ```y = w1x1 + w2x2 + ... + w13x13 + b``` 이 수식을 만듦.  
컴퓨터는 학습과정에서 입력 data를 보고 이 수식의 w들과 b를 찾는 것!  
```
ex) 종속변수가 2개 => 퍼셉트론 2개가 병렬로 연결
y1 = w1x1 + w2x2 + ... + w13x13 + b
y2 = w'1x1 + w'2x2 + ... + w'13x13 +b'

입력은 x1~x13이고 출력은 y1,y2 일때
찾아야 하는 가중치는 12+12=24개, bias는 1+1=2개 => 총 26개의 숫자
```

### 원핫인코딩(one hot encoding)  
* ```인코딩 = pd.get_dummies(데이터)```  
* 수식의 결과는 숫자인걸?? => **해당하는 범주의 column을 1로 초기화, 나머지 해당 안되는건 0**으로 만들어준다.  
* 이 방식을 **원핫인코딩**이라고 함  
* ```데이터``` 내의 범주형 데이터만 one hot encoding 해줌  
<img width="374" alt="4" src="https://user-images.githubusercontent.com/46364778/104098570-eb12e200-52df-11eb-80c2-1e8b6e2c9f4b.PNG">  

* 출력 layer는 3개 <= y1, y2, y3 (각 column에 대한 수식 필요, 1인지 0인지)  

### 소프트맥스(softmax) & corssentropy  
* ```Y = tf.keras.layers.Dense(3, activation='softmax'(X)```  
* ```model.compile(loss='categorical_crossentropy')```: 문제 유형에 맞게 **loss** 지정한 것  
**회귀**는 ```mse```, **분류**는 ```crossentropy```  
* 0% ~ 100% 확률 값으로 분류 표현함 => 이렇게 하도록 만들어주는 도구가 Sigmoid와 **Softmax**  
* 즉, 비율로 예측하기 위해 softmax 사용  
<img width="390" alt="5" src="https://user-images.githubusercontent.com/46364778/104101623-d2ef9280-52e0-11eb-9f11-82702efa8e4c.PNG">  

* 수식을 <b>softmax()</b>로 감쌈 => 원래 수식 결과는 ```-무한대~ 무한대``` 범위였음. but 이를 통해 분류 모델에 맞는 ```0 ~ 1```사이의 값으로만 출력이 됨!  

<img width="336" alt="7" src="https://user-images.githubusercontent.com/46364778/104102656-37125680-52e1-11eb-8396-92555bdcfec6.PNG">  

* **f**는 퍼셉트론의 출력이 어떤 형태로 나가야하는지 조절  
* 이런 함수를 **활성화 함수(Activation)** 이라고 함  


## 회귀(수치 데이터) 실습  
* [레모네이드 판매 예측](https://github.com/rrabit42/study_MachineLearning/blob/main/Tensorflow101/lemonade.py)  
* [보스턴 집값 예측](https://github.com/rrabit42/study_MachineLearning/blob/main/Tensorflow101/boston.py)  

## 분류(범주형 데이터) 실습  
* [아이리스 품종 분류](https://github.com/rrabit42/study_MachineLearning/blob/main/Tensorflow101/iris.py) 

