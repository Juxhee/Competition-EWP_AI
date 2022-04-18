## 제2회 EWP 발전 빅데이터 AI 경진대회 - TEAM SLH

https://ewp.co.kr/popup/20201211/popup.asp

#### 미반응 암모니아 최소화를 위한 SCR 출구측, 굴뚝 Nox량 등 예측

- 석탄 연소 시 Nox, Sox 발생하는데, 이 중 Nox량을 기준량 이하로 배출하기 위해 SCR에서 암모니아 공급
- SCR에서 공급하는 암모니아량은 SCR출구 측, 굴뚝의 Nox량과 암모니아 slip에 의해 결정됨
- 주어진 운전 데이터로 이러한 값들을 예측하는 모델을 만들고자 함
</br>
<img src="https://user-images.githubusercontent.com/60679596/163787682-3c0658a2-82e3-47ff-b062-2a4795db5845.png" width="600" height="200">

- 데이터셋 : two pass type 센서에서 측정된 시계열 데이터

</br>
</br>

## Task 1
#### 50분 치 시계열 데이터를 입력으로 받아 그 다음 10분 치의 5개 y값 예측하는 모델 구축

`model.py`

</br>

<img src="https://user-images.githubusercontent.com/60679596/163787193-45ca63e3-e9e8-405d-bd3f-7039c630133a.png" width="700" height="200">

<img src="https://user-images.githubusercontent.com/60679596/163789939-524b4130-69f4-4ff9-ad37-0b95e365b306.png" width="800" height="360">

</br>

## Train

`python stage1_main.py`
</br>

## Test

`python stage1_test.py`

</br>
</br>

## Task 2
#### Nox량과 암모니아 slip을 최소화하는 최적의 암모니아 투입량을 산출하는 알고리즘 구축

</br>

<img src="https://user-images.githubusercontent.com/60679596/163788884-d145fdb5-b61f-4b39-84f3-6fed0cf39b82.png" width="700" height="250">

<img src="https://user-images.githubusercontent.com/60679596/163788023-0dfb3166-6392-470d-9ba6-715aeda338e5.png" width="700" height="300">

- framework

<img src="https://user-images.githubusercontent.com/60679596/163788257-b0da3a50-171a-4d23-afa8-5361f5b330d8.png" width="700" height="300">

<img src="https://user-images.githubusercontent.com/60679596/163788350-933e4e34-ba6b-4ac3-aed7-6d9b026e5ebd.png" width="700" height="250">

</br>

- 모델링 결과

<img src="https://user-images.githubusercontent.com/60679596/163788490-656f5317-433a-40f1-886e-a56f6bc94f40.png" width="700" height="300">

</br>

## How to use

`python stage2_main.py`

