# PotHole Detection Project

## Introduction
+ 딥러닝 기반 포트홀(PotHole) 감지 모델 개발
+ 전방 카메라 이미지로 포트홀을 감지하는 문제
    + Image Classfication
    + Label : [포트홀 없음, 포트홀, 보수완료]

## Data
+ AI Hub의 개방 데이터를 이용
    + https://aihub.or.kr/aidata/34112/download
+ 실제 차량의 전방 카메라로 수집한 이미지 데이터
    + https://drive.google.com/file/d/1byD-TlXFXVJysi7q3K8y-W_96qbbfF5L/view
+ Data Unbalance 때문에, Kaggle에서 열린 비슷한 TASK의 이미지를 직접 수집
+ 전처리 과정에서 서로 다른 크기의 이미지를 320x320크기의 Color 이미지로 처리함

### Class 0 : 포트홀 없음
![Image_3654](https://user-images.githubusercontent.com/28241676/155121803-516b9912-81b9-4727-a944-77df90746bde.png)

### Class 1 : 포트홀
![Image_16138](https://user-images.githubusercontent.com/28241676/155122239-5e65a021-f38a-4d32-a84c-1a7cd9bf8061.png)

### Class 2 : 보수완료 포트홀
![Image_23301](https://user-images.githubusercontent.com/28241676/155122397-c8df5c90-634e-451f-bb41-97c71bed6b6f.png)

## Model

### Model 1 ~ 3
+ model 1
    + Simple Convolution Network
    + 단순히 합성곱 연산을 쌓은 네트워크
+ model 2 :
    + model 1 + Channel-Attnetion
+ model 3 :
    + model 2 + self-Attention(on FC Layer)
### Model 4~6
+ model 4
    + deep Convolution Network
    + VGG의 컨셉으로한 Network
+ model 5 :
    + model 4 + Channel-Attention
+ model 6 :
    + model 5 + Self-Attention(On FC Layer)
    
## Result

## Conclusion