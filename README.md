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
+ model 1 :
    + Simple Convolution Network
    + 단순히 합성곱 연산을 쌓은 네트워크
+ model 2 :
    + model 1 + Channel-Attnetion
+ model 3 :
    + model 2 + Self-Attention(on FC Layer)
### Model 4~6
+ model 4 :
    + deep Convolution Network
    + VGG의 컨셉으로한 Network
+ model 5 :
    + model 4 + Channel-Attention
+ model 6 :
    + model 5 + Self-Attention(on FC Layer)

### Model 7~
+ model 7 :
    + VVG 컨셉의 Model 4에 잔차연결을 추가한 컨셉의 모델
    + model 4 + Residual Connection
+ model 8 :
    + model 7 + Channel-Attention
+ model 9 :
    + Model 8 + Self-Attention(on FC Layer)
    
## Result
### Performance table.
|  Model  |                              description                             |  Loss  | Accuracy |
|:-------:|:--------------------------------------------------------------------:|:------:|:--------:|
| Model_1 | Simple CNN                                                           | 0.7505 |   66.4%  |
| Model_2 | Simple CNN + Attention(Channel)                                      | 0.4351 |   80.7%  |
| Model_3 | Simple CNN + Attention(Channel) + Self-Attention(on FC Layer)        | 1.0835 |   69.7%  |
| Model_4 | VGG16 Concept CNN                                                    | 0.6275 |   75.6%  |
| Model_5 | VGG16 Concept CNN + Attention(Channel)                               | 0.5021 |   79.0%  |
| Model_6 | VGG16 Concept CNN + Attention(Channel) + Self-Attention(on FC Layer) | 0.4266 |   84.9%  |
| Model_7 | VGG16 + Residual Connection                                          | 0.5739 |   84.9%  |
| Model_8 | VGG16 + Residual Connection + Attention(channel)                     | 0.7012 |   81.5%  |
| Model_9 | VGG16 + Residual Connection + Attention(Channel, Self-Attention)     | 0.6299 |   77.3%  |

### Result Summary
![Model_Summary](https://user-images.githubusercontent.com/28241676/155266429-971597a2-49ea-4bf5-ac85-f656ccb9538d.png)
<details>
<summary>모델 Inference 결과 Shell</summary>

<div markdown="1">

```shell
Inference MyModel_1.pt
2it [00:03,  1.74s/it, loss=0.639662, acc=0.664]
	loss : 0.750558
	acc : 0.664


Inference MyModel_2.pt
2it [00:00,  2.34it/s, loss=0.298573, acc=0.807]
	loss : 0.435107
	acc : 0.807


Inference MyModel_3.pt
2it [00:00,  2.39it/s, loss=0.676090, acc=0.697]
	loss : 1.083568
	acc : 0.697


Inference MyModel_4.pt
2it [00:00,  2.15it/s, loss=0.559131, acc=0.756]
	loss : 0.627527
	acc : 0.756


Inference MyModel_5.pt
2it [00:00,  2.24it/s, loss=0.396172, acc=0.790]
	loss : 0.502188
	acc : 0.790


Inference MyModel_6.pt
2it [00:00,  2.19it/s, loss=0.202595, acc=0.849]
	loss : 0.426644
	acc : 0.849


Inference MyModel_7.pt
2it [00:00,  2.22it/s, loss=0.395379, acc=0.849]
	loss : 0.573950
	acc : 0.849


Inference MyModel_8.pt
2it [00:00,  2.21it/s, loss=0.763084, acc=0.815]
	loss : 0.701221
	acc : 0.815


Inference MyModel_9.pt
2it [00:00,  2.19it/s, loss=0.351030, acc=0.773]
	loss : 0.629991
	acc : 0.773
```

</div>
</details>

## Conclusion
Attention의 성능 향상을 체감할 수 있는 좋은 프로젝트였다.<br>
CNN단(Feature Encoding)에서는 Channel-Attention을 통해 집중해서 봐야할 영역을 스스로 학습시켜서 학습의 향상을 보였다.<br>
또한 분류기(FC Layer)단에서는 Transformer의 Self-Attention 기법을 통해 정제된 특징벡터에 대한 관계를 학습을 유도했다.

개인적으로 Model 7~9가 가장 성능이 좋을 것으로 예상했는데, 생각보다 낮았던 이유는 과적합이 발생한 것으로 생각된다.<br>
  *(실제로 그래프를 보면, 특정 Epoch부터 Loss가 늘어난 것을 볼 수 있다.)*<br>

Paper with Code에 보면, 여러가지 Attention Method가 소개되고 있는데, 이번 계기를 통해서 성능을 향상 시킬 수 있는 여러가지 Attention 기법을 찾아보고 공부해봐야겠다.

