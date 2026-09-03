---
title: "[Paper Review, KR] FlashWorld: High-Quality 3D Scene Generation within Seconds"
date: 2026-08-17
categories:
  - 3D Vision
tags:
  - 3D Generation
  - Distillation
---

> **Paper Information** \\
> **Title:** FlashWorld: High-Quality 3D Scene Generation within Seconds \\
> **Authors:** Xinyang Li, Tengfei Wang, Zixiao Gu, Shengchuan Zhang, Chunchao Guo, Liujuan Cao \\
> **Venue:** ICLR 2026 Oral \\
> **Link:** [[Paper](https://arxiv.org/pdf/2510.13678)], [[Project](https://imlixinyang.github.io/FlashWorld-Project-Page/)], [[Github](https://github.com/imlixinyang/FlashWorld)]


## Teaser Image (Poster)

<p align="center">
  <img src="/assets/images/posts/2026-08-20-flashworld/1787817427574.png" width="100%">
</p>

## Introduction

3D Generation 분야는 크게 성장하고 있는 분야이지만, **scarcity of high-quality 3D scene data**와 **exponential complexity of modeling real-world scenes**라는 두 개의 큰 장애물 때문에 어려움을 겪고 있다고 말한다.

여기에는 크게 2개의 패러다임이 있다.

먼저 multi-view-oriented(MV-oriented) 파이프라인이다. diffusion model이 텍스트나 참조 이미지로부터 여러 시점의 이미지들을 먼저 생성한 다음, 3D reconstruction을 수행하는 방식이다. 그러나 시점 합성 과정에서 명시적인 3D 제약 조건이 없어 geometric 혹은 semantic inconsistencies를 발생시킬 수 있다. 게다가 이는 상당한 computational overhead가 발생하고 생성 시간도 상당하다는 것을 알 수 있다.

diffusion model의 효율성을 높이기 위해, post-training distillation 기술들이 자주 사용된다. 이러한 distillation 기법을 직접 적용하면 프레임워크가 본질적으로 가진 한계점을 오히려 증폭시키게 될 수 있다.

다음으로 3D-oriented 패러다임이다. 이 방식은 diffusion model이랑 미분가능한 rendering을 combine하는 것이다. 이 방식은 물체나 배경의 기하학적 형태가 어긋나지 않고 물리적 일관성을 유지하나 화질이 다소 흐릿해지는 문제가 있다. 게다가 refinement stage를 추가로 필요로한다.

## Preliminary

FlashWorld의 핵심인 cross-mode post-training을 이해하기 위해서는 먼저 **Diffusion Model**과 **Distribution Matching Distillation (DMD)**에 대한 이해가 필요하다.

### Diffusion Model

Diffusion model은 일반적으로 Gaussian noise에서 시작하여 점진적으로 noise를 제거하면서 target data distribution의 sample을 생성한다.

원본 데이터 $$x$$에 timestep $$t$$에 따른 Gaussian noise를 추가하는 forward process는 다음과 같이 정의된다:

$$
x_t = F(x,t) = \alpha_t x + \sigma_t \epsilon,
\qquad
\epsilon \sim \mathcal{N}(0,I)
$$

여기서 $$\alpha_t$$와 $$\sigma_t$$는 timestep $$t$$에 따른 signal과 noise의 비율을 결정한다.

즉,

$$
x_t
=
\underbrace{\alpha_t x}_{\text{signal}}
+
\underbrace{\sigma_t\epsilon}_{\text{noise}}
$$

로 볼 수 있다.

Denoising network는 noisy sample $$x_t$$와 timestep $$t$$를 입력으로 받아 원래의 clean data $$x$$를 예측하도록 학습된다.

$$
\mathcal{L}
=
\mathbb{E}_{x,t,\epsilon}
\left[
\left\|
x-\hat{x}_\theta(x_t,t)
\right\|^2
\right]
$$

위 수식에서는 clean data $$x$$를 직접 예측하는 $$x$$-prediction을 사용하지만, diffusion model은 noise $$\epsilon$$을 예측하거나 $$x$$와 $$\epsilon$$의 선형 결합인 $$v$$를 예측하는 방식으로도 학습될 수 있다.

이러한 prediction들은 모두 denoised estimate $$\mu(x_t,t)$$로 변환할 수 있으며, 이를 이용하면 distribution의 **score**를 다음과 같이 표현할 수 있다.

$$
s(x_t,t)
=
\nabla_{x_t}\log p_t(x_t)
=
-
\frac{x_t-\alpha_t\mu(x_t,t)}
{\sigma_t^2}
$$

Score

$$
s(x_t,t)=\nabla_{x_t}\log p_t(x_t)
$$

는 현재 sample $$x_t$$가 해당 data distribution에서 probability가 더 높은 영역으로 이동하려면 어느 방향으로 움직여야 하는지를 나타내는 gradient라고 이해할 수 있다.

즉, diffusion model은 단순히 denoising 결과를 예측하는 것뿐만 아니라, 현재 sample을 data distribution에 더 가까운 방향으로 이동시키기 위한 **score field**를 제공할 수 있다.


### Distribution Matching Distillation (DMD)

**Distribution Matching Distillation (DMD)**은 많은 denoising step이 필요한 diffusion model을 적은 step만으로 generation을 수행하는 generator로 distillation하기 위한 방법이다.

기존 diffusion teacher가

$$
z
\rightarrow
x_{T-1}
\rightarrow
x_{T-2}
\rightarrow
\cdots
\rightarrow
x_0
$$

처럼 여러 번의 denoising step을 거쳐 sample을 생성한다면, DMD의 목적은 few-step student generator $$G_\theta$$가 생성하는 distribution을 teacher의 target distribution과 일치시키는 것이다.

즉,

$$
p_{\text{fake}}
\rightarrow
p_{\text{real}}
$$

이 되도록 student generator를 학습한다.

여기서, $$p_{\text{real}}$$는 teacher diffusion model이 표현하는 target distribution, $p_{\text{fake}}$$는 현재 student generator $$G_\theta$$가 생성하는 distribution

을 의미한다.

DMD에서는 randomly sampled noise $$z$$를 student generator에 입력하여

$$
x_{\text{fake}} = G_\theta(z)
$$

를 생성하고, 여기에 다시 timestep $$t$$에 해당하는 noise를 추가한다.

$$
x_t
=
F(G_\theta(z),t)
$$

이 noisy sample에 대해 real distribution과 fake distribution 각각의 score를 계산한다.

$$
s_{\text{real}}(x_t,t)
=
\nabla_{x_t}
\log p_{\text{real}}(x_t)
$$

$$
s_{\text{fake}}(x_t,t)
=
\nabla_{x_t}
\log p_{\text{fake}}(x_t)
$$

DMD의 핵심 gradient는 다음과 같이 두 score의 차이를 이용한다.

$$
\nabla \mathcal{L}_{\mathrm{DMD}}
=
-
\mathbb{E}_{t}
\left[
\int
\left(
s_{\mathrm{real}}
\left(
F(G_\theta(z),t),t
\right)
-
s_{\mathrm{fake}}
\left(
F(G_\theta(z),t),t
\right)
\right)
\frac{dG_\theta(z)}{d\theta}
\,dz
\right]
$$

여기서 핵심적인 부분은

$$
s_{\text{real}} - s_{\text{fake}}
$$

이다.

Score의 정의를 이용하면

$$
s_{\text{real}} - s_{\text{fake}}
=
\nabla_x\log p_{\text{real}}(x)
-
\nabla_x\log p_{\text{fake}}(x)
$$

이므로,

$$
s_{\text{real}} - s_{\text{fake}}
=
\nabla_x
\log
\frac{p_{\text{real}}(x)}
{p_{\text{fake}}(x)}
$$

로 볼 수 있다.

따라서 $$s_{\text{real}}-s_{\text{fake}}$$는 단순히 student에게 loss를 전달하기 위해 사용하는 것이 아니라, **현재 student의 output distribution이 real distribution과 비교했을 때 어느 방향으로 수정되어야 하는지를 나타내는 gradient**라고 이해할 수 있다.

이를 다시

$$
\frac{dG_\theta(z)}{d\theta}
$$

를 통해 generator parameter $$\theta$$까지 전달함으로써

$$
p_{\text{fake}}
\rightarrow
p_{\text{real}}
$$

이 되도록 student generator를 학습한다.


### Real Score Model and Fake Score Model

DMD에서는 $$s_{\text{real}}$$과 $$s_{\text{fake}}$$를 직접 알 수 없기 때문에 각각 diffusion model을 이용하여 score를 추정한다.

Real score의 경우 pretrained diffusion model

$$
\mu_{\text{real}}
$$

을 사용한다.

$\mu_{\text{real}}$은 target data distribution에 대해 이미 학습되어 있으므로 training 과정에서 **frozen** 상태로 유지된다.

반면 fake distribution은 student generator가 학습될 때마다 계속 변화한다.

$$
p_{\text{fake}}^{(0)}
\neq
p_{\text{fake}}^{(1)}
\neq
p_{\text{fake}}^{(2)}
\neq \cdots
$$

따라서 fake distribution의 score를 추정하는 별도의 diffusion model

$$
\mu_{\text{fake}}
$$

가 필요하다.

$\mu_{\text{fake}}$는 현재 student generator가 생성한 sample들을 이용한 diffusion loss를 통해 지속적으로 update되며, 현재의

$$
p_{\text{fake}}
$$

를 추정하도록 학습된다.

전체적인 구조는 다음과 같이 생각할 수 있다.

$$
z
\rightarrow
G_\theta(z)
\rightarrow
F(G_\theta(z),t)
$$

생성된 noisy sample은 두 score model에 입력된다.

$$
F(G_\theta(z),t)
\rightarrow
\begin{cases}
\mu_{\text{real}} \rightarrow s_{\text{real}} \\
\mu_{\text{fake}} \rightarrow s_{\text{fake}}
\end{cases}
$$

그리고

$$
s_{\text{real}}-s_{\text{fake}}
$$

를 이용하여 student generator $$G_\theta$$를 update한다.


### Why DMD Accelerates Inference

DMD의 주된 목적은 **training 자체를 빠르게 하는 것이 아니라 inference에 필요한 denoising step을 줄이는 것**이다.

일반적인 multi-step diffusion teacher가

$$
z
\xrightarrow{\text{many denoising steps}}
x
$$

를 통해 target distribution의 sample을 생성한다면, DMD는 student가

$$
z
\xrightarrow{\text{few steps}}
\hat{x}
$$

만으로도

$$
p(\hat{x})
\approx
p(x)
$$

가 되도록 학습한다.

즉 teacher의 긴 denoising trajectory를 그대로 student가 따라가는 것이 아니라, **teacher가 많은 denoising step을 통해 최종적으로 형성하는 output distribution을 few-step generator가 재현하도록 학습하는 것**이다.

따라서 distillation training에는 real score model, fake score model, student generator 등이 필요하기 때문에 학습 과정 자체가 단순해지는 것은 아니지만, 학습이 완료된 뒤 inference에서는 teacher와 fake score model이 필요하지 않으며 few-step student만 사용하면 된다.

정리하면,

$$
\boxed{
\text{DMD: Multi-step Teacher Distribution}
\rightarrow
\text{Few-step Student Generator}
}
$$

이며, FlashWorld에서는 이러한 DMD를 기반으로 높은 visual quality를 가지는 MV-oriented mode의 distribution을 3D consistency를 가지는 3D-oriented few-step generator에 전달한다.

## Method & Technical Details



