---
title: "[Paper Review, KR] MotionStream: Real-Time Video Generation with Interactive Motion Control"
date: 2026-09-07
categories:
  - Other Fields
tags:
  - distillation
  - real-time generation
  - autoregressive
---

> **Paper Information** \\
> **Title:** MotionStream: Real-Time Video Generation with Interactive Motion Control \\
> **Authors:** Joonghyuk Shin, Zhengqi Li, Richard Zhang, Jun-Yan Zhu, Jaesik Park, Eli Shechtman, Xun Huang \\
> **Venue:** ICLR 2026 Oral \\
> **Link:** [[Paper](https://joonghyuk.com/motionstream-web/paper.pdf)], [[Project](https://joonghyuk.com/motionstream-web/)], [[GitHub](https://github.com/alex4727/motionstream)]

## Teaser Image (Poster)

<p align="center">
  <img src="/assets/images/posts/2026-09-07-motionstream/1789522129230.png" width="100%">
</p>

## Introduction

Motion-controlled video generation은 사용자가 객체, 인물, 카메라의 움직임을 직접 지정할 수 있도록 발전해 왔다. 초기에는 구조(structure), 카메라, subject, audio 등 다양한 조건을 활용하는 controllable video generation 연구가 진행되었고, 이후 motion 자체가 영상의 동역학을 직접 표현하는 중요한 conditioning signal로 자리 잡으면서 optical flow, 2D/3D trajectory, bounding box, semantic segmentation 등 다양한 motion representation이 활용되기 시작했다. 최근 video diffusion model들은 이러한 조건을 이용해 높은 화질과 정교한 trajectory following을 달성했지만, 대부분 전체 영상과 전체 motion **condition을 한 번에 처리하는 bidirectional diffusion 구조에 기반**한다. 따라서 **사용자가 미래의 trajectory를 모두 미리 지정해야 하며, 생성 중간에 결과를 확인하거나 motion을 수정하기 어렵다**. 또한 **diffusion의 반복적인 denoising 과정 때문에 생성 속도가 매우 느린데**, 논문에서는 Motion Prompting의 경우 5초 영상을 생성하는 데 약 12분이 소요되는 사례를 들고 있다. 결국 **기존 motion-controlled video generation은 높은 품질과 제어 능력에도 불구하고 느린 생성 속도, non-causal processing, 짧은 생성 길이 때문에 실시간 상호작용에는 적합하지 않았다**.

한편 video generation의 시간적 생성 방식은 초기 GAN 기반 autoregressive/parallel video synthesis에서 시작하여, 이후 denoising objective를 사용하는 video diffusion model과 next-token prediction 기반 autoregressive video model로 발전해 왔으며, 최근에는 두 방식을 결합해 causal하면서도 고품질인 AR-diffusion 모델을 만드는 연구가 활발하다. 특히 느린 bidirectional diffusion teacher를 빠른 autoregressive student로 distillation하는 방식이 등장하면서 real-time generation 가능성이 커졌지만, 이러한 **모델들은 학습한 sequence 길이를 넘어 장시간 생성할 경우 color drift나 품질 저하가 누적되거나, 이를 방지하기 위해 복잡한 long-video fine-tuning이 필요하다는 한계**가 있다. 동시에 Genie 계열과 같은 interactive video world model 연구도 실시간 user interaction을 목표로 발전하고 있지만, 많은 방법이 **매우 큰 inference compute를 요구하거나 Minecraft·게임과 같은 closed-domain 또는 synthetic environment에 제한**되어 있다. 따라서 기존 연구들은 각각 controllability, causal generation, real-time interaction 중 일부는 달성했지만, **open-domain photorealistic video에서 motion control을 유지하면서 장시간 안정적으로 실시간 생성하는 문제는 여전히 해결되지 않은 과제**로 남아 있다.

## Method & Technical Details

<p align="center">
  <img src="/assets/images/posts/2026-09-07-motionstream/1789524542899.png" width="70%">
</p>

#### Adding Motion Controls to BiDirectional Teacher Models

해당 Technique의 핵심은 

> 기존 Wan 계열의 bidirectional video diffusion model에 motion trajectory를 조건으로 넣을 수 있는 기능을 추가해서 teacher model을 만드는 것

이다. 

즉, motion을 정확히 잘 따르는 고품질 teacher를 만드는 단계라고 보면 된다.

먼저, 입력 motion을 어떻게 표현할지부터 생각을 해보아야 한다. 즉, **Track Representation**을 생각해보자. 

일단, 입력 motion은 여러 개의 **2D point trajectory**로 표현된다. 어떤 점 $n$이 시간에 따라

$$
(x_t^n, y_t^n)
$$

처럼 움직인다고 생각하면 된다. 각 track에는 고유한 $d$-dimensional embedding $\phi_n$을 부여한다. 이 embedding 은 학습 가능한 ID embedding이 아닌, random ID에 sinusoidal positional encoding을 적용해서 만든 값이다. 즉,

- track 1 $\rightarrow$ $\phi_1$
- track 2 $\rightarrow$ $\phi_2$
- track 3 $\rightarrow$ $\phi_3$

처럼 각 trajectory를 서로 구분할 수 있는 identity가 생긴다. 그리고 각 프레임 $t$에서 그 track의 위치에 해당 embedding을 놓는다. *(자 이렇게 보면 어려운 기술 용어들이 나와서 어렵다. 아래의 detail을 눌러서 좀 더 자세하게 확인해보는게 좋다)*

<details markdown="block">
<summary>Track Representation</summary>
<div markdown="1" style="border-left: 4px solid #0969da; padding-left: 12px; margin-top: 10px;">

많은 4D Generation 논문들을 보면 track에 관한 개념이 많이 등장한다 trac 이라는 것은 비디오 프레임 흐름에 따라 하나의 점이 프레임이 바뀌었을 때 어디로 이동하는 지를 알려주는 표현 방식이라고 생각하면 된다.

그러니까 $t=1$ 일때 $(100,200)$ 에 있던 하나의 점이 $t=2$가 되니까 $(105,201)$ 에 있을 때, 이 track을 우리는 이와 같이 표현한다:

$$
\{(x_t^n, y_t^n)\}_{t=1}^T = \{(100,200), (105,201)\}
$$

MotionStream은 실제 학습 데이터에서 CoTracker3라는 모델을 통해 추적한다.
그리고 CoTracker3를 통해 2500개의 점을 추적하고, 학습 시 그중 1000~2500개를 random sampling 한다.

논문에서는 가 2D track마다 randomly sampled ID number를 하나 부여한다고 되어 있다. 그런데 scalar인 ID number를 그대로 모델에 넣는 것보다 고차원 vector로 바꾸는 것이 유용하여 ID number를 $d$-dimensional vecotr로 바꾼다. 이를 **embedding**이라고 부른다. MotionStream에서는 $d=64$로 설정하여 한 track의 ID를 64차원 vecotr로 표현한다.

따라서 예시로,

$$
ID = 17
$$

인 Track이 있을 때, 이 ID를

$$
\phi_{17} = [0.21, -0.84, 0.56, ...] \in \mathbb{R}^{64}
$$

로 바꾸는 것이다. 그런데, ID를 Vector로 바꿀 때 사용하는 방식이 바로 Sinusoidal Positional Encoding이다. ID $p$에 대해

$$
[sin(p),cos(p),sin(p/10),cos(p/10), ...]
$$

처럼 여러 frequency의 sin/cos 값을 만들어 하나의 vector로 만든다. 그래서 ID가 다르면 서로 다른 vector가 만들어진다는 것이 핵심이다. 

</div>
</details>

이 다음으로 수식이 등장한다.

$$
c_m \left[ t, \left\lfloor \frac{y_t^n}{s} \right\rfloor, \left\lfloor \frac{x_t^n}{s} \right\rfloor \right] = v[t, n]\phi_n
$$

이 수식의 의미는 **VAE latent resolution에 맞게 downsample한 위치에 embedding을 찍는 방식**을 말한다.

먼저, $c_m$은 motion conditioning tensor다. 논문에서는 shape을 $c_m \in \mathbb{R}^{T \times H/s \times W/s \times d}$로 정의한다. 축을 하나씩 보면, (시간, 세로 위치, 가로 위치, track embedding) 이다. 즉, 각 frame 마다 $H/s \times W/s$ 크기의 map이 하나 있고, 각 위치마다 $d$-dimensional vecotr를 넣을 수 있는 tensor다.

원래 track 위치는 원본 video pixel 좌표다. 예를 들어 spatial compression이 $s=8$이고, $H=480, W=832$라고 한다면, diffusion model은 원본 RGB 영상을 직접 처리하지 않고 VAE로 압축된 latent를 처리하기 때문에 $480 \times 832$가 $60 \times 104$ 정도로 작아진다. 따라서, 원본 pixel에서 track이

$$
(x,y) = (320,160)
$$

에 있다면 latent coordinate에서는

$$
(\frac{320}{8}, \frac{160}{8}) = (40,20)
$$

이다. 그래서 식에서

$$
\left\lfloor \frac{y_t^n}{s} \right\rfloor, \left\lfloor \frac{x_t^n}{s} \right\rfloor
$$

를 사용한다. 그리고 track의 visibility에 대해 이야기 하자면, 모든 track이 모든 frame에서 보이는 것이 아니다. 예를 들어 사람이 다른 사람 뒤로 가려지면 특정 point가 안 보일 수 있다. 이를 나타내는 것이

$$
v[t,n] \in \{0,1\}
$$

이다. 보이면 1의 값을 갖게 되어

$$
c_m[ \cdots ] = \phi_n
$$

이고, 가려져 안보이면 0의 값을 갖게 되어

$$
c_m[ \cdots ] = 0
$$

이 된다. 즉, visibility mask 역할을 수행하는 것이 $v[t,n]$이다. 그 다음으로 Track head에 대해 이야기해보자. 이제 만든 $c_m$을 바로 diffusion transformer에 넣는 게 아니라, 작은 network를 통과시킨다. 이게 바로 track head다.

흐름은 대략

$$
\text{2D Tracks} \rightarrow \text{Sinusoidal ID Embedding} \rightarrow c_m \rightarrow \text{Track Head}
$$

이다. Track head는 먼저 4x temporal compression을 수행한다. 이유는 Wan의 VAE가 video latent를 시간축에서도 압축하기 때문이다. 원본 track은 frame마다 존재하지만, video latent는 temporal resolution이 더 작다. 따라서 motion condition도 같은 시간 해상도로 맞춰야 한다.

그 뒤에 $1 \times 1 \times 1 \;\; Conv3D$를 사용한다. 이 convolution은 각 위치에 존재하는 feature vector를 원하는 channel dimension으로 projection하는 가벼운 layer다. 

그런데, 기존의 controllable diffusion에서는 ControlNet 스타일 구조를 많이 사용하지만 해당 논문의 경우에는 단순한 방식으로 처리한다. 이 이유에 대해 논문의 저자들은 Controlnet은 원래 backbone과 유사한 network branch를 하나 더 추가하는 방식이라 계산량이 많이 증가하기 때문이라고 답한다. 따라서, Motion Stream의 목표인 **real-time rendering**과 맞지 않아 이를 넣지 않는다. 

그래서 track head로 처리한 motion feature를 video latent에 직접 channel-wise concatenation한다:

$$
[\text{video latent} || \text{motion feature}]
$$

이를 통해 ControlNet과 같이 backbone을 복제하지 않고 motion control을 추가할 수 있었다.

다음으로 이 teacher를 어떻게 학습할지에 대해 논의한다. 논문의 저자들은 **rectified flow matching objective**로 학습을 진행한다. 입력은 개념적으로 $z_t + \text{text} + \text{motion condition}$이고, 출력은 $\text{velocity prediction}$이다. 그래서 이 teacher는 text도 따르고, track trajectory도 따르는 video generator로 학습된다.

그런데 여기서 문제가 있다. 우리가 이전 수식에서 track visibility를 표현하기 위해 $v[t,n]$을 사용했었다. 그래서 이 값이 0이면 해당 위치의 motion feature가 0이도니다. 그런데 사용자가 interaction 중에 control을 중단하게 되면 motion signal이 없어지므로 이 경우에도 0이 된다. 즉 모델 입장에서는 0이라는 값을 보고 track 가려져서 0이 된건지 사용자가 track을 더 이상 지정하지 않는 것인지 구분이 힘들다.

이는 모델이 track이 사라진 것이 object가 사라진 것이라고 판단하는 오류를 발생할 수 있다. 논문에서는 이 경우에 object가 갑자기 나타나거나 사라지는 artifact가 생길 수 있다고 설명한다. 그래서 논문은 **Stochastic mid-frame masking**이라는 방법을 사용한다.

이 방법은 일부 frame에서 일부러 motion signal을 없애는 방식이다. 논문에서는

$$
c_m[t_{rand}, :, :] = 0
$$

로 만들고 확률은 $p_{mask}=0.2$다. 이 방법의 의미는 중간에 motion condition이 갑자기 없어져도 video 자체는 자연스럽게 이어져야 한다는 것을 학습시키는 것이다. 물론, 이 방법을 처음부터 사용하지는 않고, 먼저 이 방법을 사용하지 않은 상태에서 motion track을 정확하게 따라가는 능력을 학습시키고 난 뒤에 이 방법을 추가해서 사용한다.

논문에서는 text condition을 $c_t$, motion condition은 $c_m$이라고 표현하고 이 두 condition의 역할은 다르다. 그런데, 두 condition guidance 중 하나가 너무 강하면 object가 너무 rigid하고 simplistic하고 이동하거나(motion guidance가 너무 클 때), user가 지정한 trajectory를 벗어날 수 있다(text guidance가 너무 클 경우).

따라서, 두 guidance의 균형이 필요하며 논문의 저자들은 **joint guidance**를 사용한다:

$$
\hat{v}
=
v_{\mathrm{base}}
+
w_t
\left(
v(c_t, c_m) - v(\emptyset, c_m)
\right)
+
w_m
\left(
v(c_t, c_m) - v(c_t, \emptyset)
\right)
$$

$$
v_{\mathrm{base}}
=
\alpha v(\emptyset, c_m)
+
(1-\alpha)v(c_t, \emptyset),
\qquad
\alpha
=
\frac{w_t}{w_t+w_m}
$$

MOtionStream의 teacher는 rectified flow model이라서, 매 denoising step마다 현재 noisy latent $z_t$를 보고 velocity $v(z_t, t, c_t, c_m)$ 를 예측한다. 그리고 이 velocity를 따라서 latent를 조금씩 clean video 방향으로 이동시킨다. 그런데, 그냥 $v(c_t, c_m)$을 사용하는 대신, CFG처럼 text와 motion condition의 영향을 강조해서 최종 velocity를 만든다. 이게 바로 위의 $\hat{v}$에 대한 수식이다. 따라서 흐름은

$$
\text{Model Predictions} \rightarrow \text{joint guidance를 통해 구한} \; \hat{v} \rightarrow \text{next denoising step}
$$

이 된다. 위의 $\hat{v}$의 수식을 보면, 그냥 $v(c_t, c_m)$을 사용하지 않고 좀 복잡하게 되어있다. 이는 CFG의 아이디어에서 비롯된 것인데, **이 condition을 넣었을 때와 안 넣었을 때 prediction이 얼마나 달라지는지 보고, 그 차이를 더 강하게 민다**는 컨셉이다.

여기서,

$$
v(c_t, c_m) - v(\cancel{0}, c_m)
$$

는 text effect에 대해 보여주고, 이는 **text가 velocity prediction에 추가한 방향**을 알려준다. 그리고 $w_t$를 곱해서 text의 영향을 조절한다. 반대로

$$
v(c_t, c_m) - v(c_t, \cancel{0})
$$

는 text는 같은데 motion만 다르다. 따라서 **motion trajectory가 추가한 방향**이라고 볼 수 있고, $w_m$을 곱해서 motion adherence를 조절한다. $\hat{v}$ 수식 밑에 $v_{base}$ 수식도 motion-only prediction, text-only prediction을 섞어서 만드는 것을 볼 수 있다.

그러나 이 joint guidance도 단점을 가지고 있다. joint guidance 방식을 사용하면 품질은 좋아지지만 계산량이 늘어난다. 왜냐하면 한 번의 denoising step에서

$$
v(c_t, c_m), v(\cancel{0}, c_m), v(c_t, \cancel{0})
$$

를 각각 계산해야 하기 때문에 3 NFE(Number of Function Evaluatiosn)가 필요하다. 그래서 이 teacher는 그 자체로는 real-time 모델이 아니다. 

그래서, 최종적으로 teacher model을 학습하는 방식에서 가장 중요한 기법들은 **Track Representation, Track Head, Stochastic Mid-frame Masking, Joint Text-Motion Guidance**라고 볼 수 있다.

#### Causal Distillation

<p align="center">
  <img src="/assets/images/posts/2026-09-07-motionstream/1789538226799.png" width="50%">
</p>

Teacher model은 현재 bidirectional diffusion model이다. 예를 들어 81 frame 영상을 만든다고 하면 teacher는 frame 전체를 한꺼번에 보고 attention을 한다. 그래서 10 frame 하나를 처리할 때도 전체 1~81 temporal context를 이용할 수 있다. 이 방식은 **좋은 품질을 내기에는 유리하지만 실시간 interaction에는 문제**가 있다. **사용자가 아직 미래 trajectory를 그리지 않았다면 미래 frame에 대한 control siganle 자체가 없다**. 그래서 현재까지 입력된 trajectory만 가지고 바로 영상을 만들어야 하는 real-time 상황에는 맞지 않다. 그래서

> student는 $1 \rightarrow 2 \rightarrow 3 \rightarrow ...$ 처럼 앞에서 생성한 결과만 보면서 다음 부분을 생성하는 causal model이 되어야 한다.

단순히 causal attention 방식을 사용하는 Self Forcing 방식을 그대로 사용하는 것도 문제가 있다고 설명한다. **첫 번째는 training horizon 밖으로 나가면 quality가 급격히 무너지는 것**이다. 왜냐하면, teacher가 81 frame 정도의 영상만 학습했다고 가정한다면, student도 그 정도 길이의 영상을 생성하는 것은 잘하지만, 83 frame을 넘어 생성하기 시작할 때무터 error가 누적되거나 geometry가 이상해지는 등의 문제가 발생할 수 있다.

그리고 두 번째로 **attention context가 길이에 따라 달라지면 latency가 일정하지 않다는 점**이다. full causal attention의 경우에 frame의 길이가 길어질수록 과거는 계속 쌓인다. 그리고 context도 같이 커지게 되는데, 이러면 긴 영상을 만들수록 attention cost도 커지게 된다. 이는 실시간 streaming 에서는 치명적이다.

