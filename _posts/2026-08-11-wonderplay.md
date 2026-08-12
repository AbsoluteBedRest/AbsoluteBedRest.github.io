---
title: "WonderPlay: Dynamic 3D Scene Generation from a Single Image and Actions"
date: 2026-08-11
categories:
  - 4D Vision
tags:
  - animation
  - 4D Generation
  - Interaction
---

> **Paper Information** \\
> **Title:** WonderPlay: Dynamic 3D Scene Generation from a Single Image and Actions \\
> **Authors:** Zizhang Li, Hong-Xing Yu, Wei Liu, Yin Yang, Charles Herrmann, Gordon Wetzstein, Jiajun Wu \\
> **Venue:** ICCV 2025 (Highlight) \\
> **Link:** [[Paper](https://arxiv.org/pdf/2505.18151)], [[Project](https://kyleleey.github.io/WonderPlay/)], [[GitHub](https://github.com/kyleleey/WonderPlay)]

## Teaser Image

![alt text](/assets/images/posts/2026-08-11-wonderplay/1786437605563.png)

## Introduction

읽기 전에 하나를 전체로 깔고 들어가는게 좋을 듯 싶다. \
**"WonderPlay의 경우에는 Physic Solver(유사 물리엔진)와 Video Diffusion로 구성되어 있다."** 라는 문장을 기억하자. \
(*참고로, 해당 논문 리뷰는 Physic Solver의 작동 방식이나 Detail을 다루지 않는다.*)

> 해당 논문은 여러가지 기존 방법론들 그리고 기법들이 가진 한계점을 설명한다: 
> 1. 기존의 Action-conditioned dynamic scene을 생성하는 방법들은 **강체(rigid) 및 탄성체(elastic) 등 제한적인 역학 표현 범위를 가지며, general한 움직임을 표현하는데 한계**를 가진다.
> 2. 기존의 video를 생성하는 방법들은 2D video만 생성할 뿐, **동적 3D Scene을 생성하지는 못하며**, **결과로 나타날 동역학적 궤적 자체를 사용자가 먼저 알고 입력**해주어야 그에 걸맞는 비디오를 생성한다. 또한, **동역학적 궤적을 어떻게 표현해야할지도 미지수**다.
> 3. 기존의 world model들은 단순히 **카메라 시점 제어(Camera Control)나 텍스트 입력 제어 수준에 머물러 있어** 물리적인 힘에 반응하는 행동 상호작용은 제공하지 못한다.
> 4. Dynamic 3D Scene을 생성하는 기존 방법들은 **단순한 장면에만 집중하고, 특정 행동 입력 반응하여 물리적으로 역학을 시뮬레이션하는 능력이 없다**.

이 논문은 이러한 한계점들을 파악하여 physic solver와 video diffusion를 함께 사용하여 **"Hybrid generative simulator"**를 만들어 해결하고자 한다

## Method & Technical Details

먼저, Input으로 제공하는 것은 single image와 actions이다. 그리고, 큰 흐름은 **"Physic Solver -> Video diffusion"** 으로 보면 된다.

여기서, 3D Scene을 $\mathcal{S}_t$라고 표현하고, 이 3D scene의 구성요소인 Background($$\mathcal{B}_t$$)과 Object($$\mathcal{O}_t$$)가 존재한다. \
일단 우리는 Input image 인 $I$와 action인 $f_g$(gravity), $f_w$(wind), $f_p$(3D point force)로부터 $$S_t$$를 만들어야 한다. 

#### Background Reconstruction

> Background를 재구성하는 방식은 **FLAGS(Fast Layered Gaussian Surfels)** 기법을 사용해서 표현된다. \
>해당 알고리즘의 자세한 설명은 Preliminaries Page에서 설명을 하도록 하겠다. 간단하게, Input image를 여러 layer로 분리하고 depth 추정 기술(ex. monoculuar depth estimation...)을 통해 3D 공간에 unprojection을 진행한다. 그 후에, optimization을 통해 Surfel의 매개변수를 최적화하는 과정을 FLAGS라는 기술이다.

> 그럼 여기서 **Surfel** 이라고 표현하는 것은 무엇인가? \
> -> Surfel 은 Surface Element (표현요소)의 약어이며, 3D 공간 상에서 물체의 표면을 표현하는 기본 단위다. \
> -> 이 논문에서는 Gaussian Surfel이라고 표현하며, (위치, 방향, 크기, opacity, 색상) 정보를 가진다.

자 그러면 Background를 구성하는 Gaussian Surfel(이하 내용에서는 가우시안이라고 지칭하겠다)은 $N_B$개가 존재하고, 아래의 가우시안의 파라미터 수식으로 구성된다고 생각하면 된다.

$$
\mathcal{B}_t = \{p^B, q^B, s^B, o^B, c^B_t\}
$$

여기서, 중괄호 안에 기호들은 각각 positions(위치), quaternions(회전), scales(크기), opacities(투명도), RGB colors(색상)을 의미하며, 각각이 가우시안을 표현하는 수단이라고 생각하면 된다.

#### Objects Reconstruction

Physic Solver은 Gaussian 그 자체를 인식하지 못하고, 대부분 Mesh 단위로 처리하게 된다. 따라서, Object Reconstruction을 단순 Gaussian으로 처리하지 않고, 특정 모델을 사용하여 Object를 Mesh를 변환하고 사용한다. 생성 방식은 아래와 같다.

1. 입력 이미지에서 Object 영역을 먼저 Segmentation을 진행한다. 
2. Instant Mesh라는 모델로 Object를 3D Mesh로 표현한다.
3. Mesh의 각 Vertex에 Gaussian Surfel을 binding하고 edge와 velocity 정보를 추가하여 physic solver 계산 준비를 마무리한다.

이 과정을 거쳐 시뮬레이션이 가능한 상태를 만들었을 때, 이 논문은 이를 **"topological Gaussian Surfel"**이라고 표현한다.
Background Reconstruction과 동일하게 이를 파라미터 수식으로 표현하게 되면 아래와 같다.

$$
\mathcal{O}_t = \{E, v_t,p^O_t,q^O_t,s^O_t,o^O_t,c^O_t\}
$$

Background 파라미터 수식과 유일하게 달라진 부분은 Edge($E$)랑 Velocity($v_t$)라는 파라미터가 추가된 부분뿐이다.

추가적으로, Object가 다양한 물질의 물리 법칙을 설명할 수 있도록 물질 고유의 속성 값인 material($m$)을 정의하고, 6가지 재질(강체, 탄성체, 옷감, 연기, 액체, 과립형 물질)로 VLM이 분류(추정)하여 사용한다.

**이제 우리는 다음 단계로 넘어갈 수 있는 $\mathcal{B}_0 \cup \mathcal{O_0} = S_0$와 추정된 $m$이 준비된다.**

#### 1st stage: Physic Solver

앞에서 말했듯이, 큰 흐름은 Physic Solver를 지나 Video Diffusion을 통해 refine 하는 것이다.

처음 stage에서는 **Physic Solver가 Coarse Dynamic Scene ($$\{\tilde{S}_t\}_{t=1}^T$$)를 만들기 위해** 사용된다. \
그래서 $T$ 전체의 Dynamic Scene을 만드는 과정은 아래의 수식을 반복하여 진행한다.

$$
v_{t+1} , p_{t+1}^O, q_{t+1}^O = solver(\tilde{S}_t, f_g, f_w(t), f_p(t))
$$

해당 수식을 보면 2가지를 알 수 있다. Background Gaussian 들은 Physic solver에 의해 처리되지 않는 다는 것(즉, static 함)과 current Scene $$\tilde{S}_t$$를 넣어 다음 $$t+1$$시점에서의 Scene인 $$\tilde{S}_{t+1}$$을 얻기 위해 다음 시점의 velocity, position, quarternion 을 추정한다는 것이다. 그러면 다음 시점의 Scene 수식을 이해할 수 있다.

$$
\tilde{S}_{t+1} = \mathcal{B}_0 \cup \{ E, v_{t+1}, p_{t+1}^O, q_{t+1}^O, s_{0}^O, o_{0}^O, c_{0}^O\}
$$

해당 수식처럼 Background 의 Gaussain 들과 다음 시점의 Scene에서 추정된 정보들을 기준으로 $t+1$ 시점에서의 Scene인 $$\tilde{S}_{t+1}$$을 얻는다. 근데, 여기서 scale, opacity, color 파라미터는 변하지 않고 기존 처음($$t=0$$)의 값을 사용하는 것을 볼 수 있는데, 이는 논문 저자들의 의도를 추론할 수 있다.

1. Physic solver가 잘 추정을 못하거나 추정하지 못하는 부분이거나
2. 어차피 이후에 video diffusion이 관여해 처리할 dynamic한 정보들이 아닌 visual quality 관련한 부분이거나
3. 혹은 추론 속도를 올리기위해

#### 2nd stage: Video Generator

이제 2번째 단계에서는 두 가지 개념을 토대로 Refinement(정제) 작업을 진행한다: **Motion Control, RGB Control**

> Motion Control의 경우, **Video diffusion에 사용될 Noise를 준비하는 step**이다.
> - 먼저, 앞서 1st stage에서 physic solver가 추출한 가우시안 입자들의 3D 속도 정보 $$\{ v_t \}_{t=1}^T$$를 가상 카메라 view에 projection하여, 2D optical Flow($$F$$)를 렌더링한다.
> - **첫 노이즈 상태 $N_0$인 랜덤 가우시안을 샘플링한 후, 앞서 구한 FLOW $$F$$ 벡터를 따라 픽셀들을 매 프레임 순차적으로 비틀어 정렬한 Warped Noise $$N(F)$$를 생성한다.** 수식은 아래와 같다.
> 
> $$
> N_{t+1} = warp(N_t, F_{t+1})
> $$
> 
> - 이를 통해 physic solver가 의도한 모션 궤적을 따라 생성하도록 가이드한다.

> RGB Control
> - 먼저, physic solver를 통해 임시 3D 장면 $$\{ \tilde{S}_t \}_{t=1}^T$$를 카메라 뷰에서 렌더링해서 RGB 영상 $$\tilde{V}$$를 만든다.
> 
> $$
> V_{s_1} = \alpha_{s_1} \tilde{V} + \sqrt{1- \alpha_{s_1}^2} N(F)
> $$
> 
> - 해당 수식을 통해 video diffusion의 forward process를 진행한다. 여기서 SDEdit 기법을 기반으로 denoising의 시작 타이밍을 Tricking 한다.
> - 따라서, 해당 수식을 사용해서 diffusion model의 맨 처음 단계($$S$$, 즉 최대 단계)에서 시작하지않고 중간 단계인 $$s_1$$ 단계에서 시작하여 복원을 시작한다.

> Spatially Varying
> - 중요한 부분이 있다. video diffusion의 hallucination은 피할 수 없는 문제다. 이를 해결하기 위해서 해당 논문은 Background과 Object를 분리해서 denoising을 진행한다.
> - 중요한 concept는 **Background의 경우에는 정제작업이 필요하지만, 결과물에서 최대한 static하게 변하지 않는 부분이어야 한다. 그리고 object의 경우에는 실제 물리 작용이 활발히 일어나야 하는 부분**이다. 따라서, Background에는 적은 step의 denoising이 필요하고, Object에는 Background보다 많은 step의 denoising이 필요하다고 판단한다.
> - 앞서 RGB Control에서 $$s_1$$ step을 Object가 필요한 step 수라고 한다면, Background는 그보다 작은 $$s_2(<s_1)$$ step 수만 있어도 충분하다.
> 
> $$
> \hat{V}_s{s_2} = M \odot V_{s_2} + (1-M) \odot (\alpha_{s_2}\tilde{V}+\sqrt{1-\alpha_{s_2}^2}N(F))
> $$
> 
> - 그래서 위의 수식처럼 미리 binary mask로 Background와 object를 분리하고, 일단 둘 다 있는 상태의 $$s_2$$ step에서 정제작업을 이어오다가 Background 부분은 과감히 버리고, 새로 forward process를 거쳐 Background 용 $$V_{s_2}$$를 만들어 Background와 Object 영역을 다시 합쳐 denoising을 진행한다.

그럼 2nd stage를 하나의 수식으로 표현한다면:

$$
V = g(F, \tilde{V}, I)
$$

즉, 결과물인 Video는 optical flow, coarse scene에서 렌더링된 video, original input video가 입력으로 생성된다

## Experiments



## Contributions