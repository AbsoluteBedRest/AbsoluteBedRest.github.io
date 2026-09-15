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



