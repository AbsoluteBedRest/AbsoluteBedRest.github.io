---
title: "[Paper Review, KR] Streaming Autoregressive Video Generation via Diagonal Distillation"
date: 2026-09-17
categories:
  - Others
tags:
  - distillation
  - real-time generation
  - autoregressive
---

> **Paper Information** \\
> **Title:** FlashWorld: High-Quality 3D Scene Generation within Seconds \\
> **Authors:** Jinxiu Liu, Xuanming Liu, Kangfu Mei, Yandong Wen, Ming-Hsuan Yang, Weiyang Liu \\
> **Venue:** ICLR 2026 \\
> **Link:** [[Paper](https://arxiv.org/pdf/2603.09488)], [[Project](https://spherelab.ai/diagdistill/)], [[Github](https://github.com/Sphere-AI-Lab/diagdistill)]

## Teaser Image (Poster)

<p align="center">
  <img src="/assets/images/posts/2026-09-05-diagonal_distillation/1789105357999.png" width="100%">
</p>

## Introduction

최근 video generation은 diffusion model의 발전으로 높은 시각적 품질을 달성했지만, 대부분의 video diffusion model은 모든 frame이 서로를 참조하는 bidirectional attention 구조를 사용하기 때문에 전체 영상을 한 번에 생성해야 한다. 이런 방식은 offline generation에는 효과적이지만, 현재 frame을 생성할 때 미래 frame을 사용할 수 없는 streaming·real-time 환경에는 적합하지 않다. 이를 보완하기 위해 autoregressive video generation이 등장했으며, video를 frame 또는 chunk 단위로 순차적으로 생성함으로써 streaming에 더 적합한 구조를 제공한다. 이후에는 autoregressive modeling과 diffusion을 결합한 hybrid 방식들이 제안되어 기존 GPT-style AR model보다 높은 visual quality를 달성했지만, 각 chunk를 생성할 때 여러 번의 denoising이 필요해 inference latency가 여전히 큰 문제가 남아 있었다.

이러한 계산량을 줄이기 위해 diffusion distillation 연구가 활발히 진행되었으며, progressive distillation이나 consistency 계열의 deterministic 방식과 DMD, DMD2, ADD 등의 distributional/hybrid 방식이 대표적이다. 하지만 이들 방법은 주로 image generation을 중심으로 개발되어, video에 직접 적용하면 temporal dependency와 inter-frame consistency를 충분히 반영하지 못하는 한계가 있다. 특히 denoising step을 크게 줄일수록 motion이 약해지거나 temporal coherence가 떨어지고, autoregressive generation에서는 이전에 생성된 결과를 계속 condition으로 사용하기 때문에 prediction error가 장시간 누적될 수 있다. CausVid와 Self-Forcing 같은 최근 연구들이 latency와 train–inference mismatch를 줄이려 했지만, 여전히 여러 denoising step이 필요하며 long video에서는 quality degradation이나 oversaturation 같은 문제가 남아 있다.

## Method & Technical Details

#### Preliminary

다른 논문들과 같이 해당 논문은 diffusion을 설명하는 것부터 시작한다. 원본 데이터 $x \sim p_{real}$ 에 timestep $t$에 따라 noise를 넣으면 이렇게 된다:

$$
q_t(x_t | x) \sim \mathcal{N}(\alpha_t x, \sigma_t^2 I) 
$$

모델이 noisy sample $x_t$를 입력받아서 원래 claen sample에 가까운

$$
\mu_{real}(x_t, t)
$$

를 예측하도록 학습된다. 그리고 score function 식은 

$$
s_{real}(x_t, t) = \nabla_{x_t} log p_{real, t}(x_t) = - \frac{x_t - \alpha_t \mu_{real}(x_t,t)}{\sigma_t^2}
$$

이렇게 생겼으며, 지난 많은 논문에 말했듯이 직관적으로 **"현재 noisy sample을 어느 방향으로 움직여야 real data distribution 쪽으로 갈 수 있는지"**를 알려주는 것이 score라고 하는 gradient다.

그런데 일반 diffusion의 경우에는 여러 denoising step이 필요하기 때문에 느리다는 단점이 있다. 이를 위해 DMD, Distribution Matching Distillation이 등장한다. DMD의 목표는 여러 step을 사용하는 teacher diffusion model의 distribution을 few-step student generator가 따라가게 만드는 것이다. 즉, **적은 step 만으로도 teacher model의 결과물과 대등하게 나오도록 하는 것**이 목표다.

논문에서는 student를

$$
G_{\theta}(z), \; z \sim \mathcal{N}(0, I)
$$

라고 표현한다.

DMD는 student가 생성한 distribution $p_{fake}$를 real/teacher distrituion $p_{real}$과 비슷하게 만드는 것이다. 즉, 이 둘의 차이를 줄이는 것인데, 논문에는 KL divergence를 이용해서

$$
KL(p_{fake,t} || p_{real,t})
$$

를 최소화한다고 표현한다. 핵심은 다음의 sutdent $G_{\theta}$를 업데이트할 때 사용되는 식이다.

$$
s_{real} - s_{fake}
$$

이는 **teacher/real distribution의 score와 student가 만든 distribution의 score가 얼마나 다른가?**를 나타내는 식이다. 이를 통해 student를 업데이트한다. DMD에 관련한 자세한 내용은 **Flashworld(ICLR 2026 Oral)** 논문 리뷰 페이지의 Preliminary 섹션에서 확인할 수 있으니 궁금하면 확인하길 바란다.

이제 여기서 저자들은 문제 제기를 한다. DMD는 원래 image generation 중심으로 만들어진 방식이기 때문에, 

$$
L_{reg} = E_{z,y}d(G_{\theta}(z), y)
$$

같은 loss는 주로 각 frame 자체의 품질을 맞추는 데 집중한다. 즉, frame 1, frame 2, frame 3 ... 각각의 경우는 teacher와 비슷하게 만들 수는 있지만, Frame 1 $\rightarrow$ Frame 2 $\rightarrow$ ... 과 같이 사이의 motion이나 temporal consistency가 자연스럽게 이어진다는 보장이 없다.

논문에서는 이를 **"기존 DMD의 regression loss가 per-frame quality는 보장하지만 temporal coherence와 long-range dependency를 명시적으로 모델링하지 못한다"**라고 표현한다.

#### Diagonal Deonising & Diagonal Forcing

이 섹션에서는 두 가지 핵심 기술에 대해 설명한다.

> 1. Diagonal Denoising: 앞쪽 chunk는 많이 denoise하고 뒤쪽 chunk는 적게 denoise 해서 속도를 높이는 방법.
> 2. Diagonal Forcing: 뒤쪽 chunk가 적은 step으로도 안정적으로 생성될 수 있도록, 이전 chunk의 적당히 noisy한 상태를 KV cache/context로 전달하는 방법.

기존 autoregressive video diffusion이라면 모든 chunk에 같은 수의 denoising step을 사용하는 것이 자연스럽다. 그런데 저자들은 앞쪽 chunk가 이후 chunk를 위한 structural prior 역할을 한다고 본다. 

즉, 만약 앞쪽 chunk에서 이미 인물의 모습, 배경, 구조 등등이 정해지면, 뒤쪽 chunk는 이를 context로 받아 생성되므로 처음부터 그만큼 많은 계산을 할 필요가 없다는 주장이다. 그래서 저자들은 chunk가 진행될때마다 denoising step이 줄어드는 **propgressive reduction**을 설명한다:

$$
X_k = D_{s_k} (Z_k | \tilde{X}_{<k}), s_k = 5,4,3
$$

여기서 $Z_k \sim \mathcal{N} (0,I)$이므로 각 새로운 chunk 자체는 여전히 Gaussian noise에서 시작한다. $D_{s_k}$는 $s_k$번 denoising하는 distilled model이고, $\tilde{X}_{<k}$는 이전 chunk들에서 전달된 noisy contex다. 즉 **뒤의 chunk가 덜 noisy하게 시작하는 것이 아니라, 좋은 temporal context가 있기 때문에 적은 step으로도 noise를 제거할 수 있다는 아이디어**다.



