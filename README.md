# PotHole Detection Project

## Model 1 ~ 3
+ Simple Convolution Network
+ 단순히 합성곱 연산을 쌓은 네트워크
+ model 1 :
    + loss : 0.750558
	+ acc : 0.664
+ model 2 :
    + model 1 + Channel_Attnetion
    + loss : 0.435107
	+ acc : 0.807
+ model 3 :
    + model 2 + self_Attention(fc_layer)
    + loss : 1.083568
	+ acc : 0.697


## Model 4~6
+ deep Convolution Network
+ VGG의 컨셉으로한 Network
+ model 4 :
    + test loss : 0.643273
    + test acc : 0.756
+ model 5 :
    + test loss : 0.498135
    + test acc : 0.790
+ model 6 :
    + test loss : 0.434023
    + test acc : 0.849