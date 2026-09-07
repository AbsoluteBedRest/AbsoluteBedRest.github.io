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

<p align="center">
  <img src="/assets/images/posts/2026-08-20-flashworld/1788745987016.png" width="70%">
</p>

해당 논문의 기법은 잘 train 되고 high-quality multi-view 를 생성할 수 있는 **"MV-oriented multi-view diffusion model"**과 few-step 만에 3D consistency를 부여하는 **"3D-oriented generator"**를 앞서 말한 DMD 기법을 통해 distillation하는 것이 목표다. 

그러기 위해서는 저자들은 두 가지 challenge를 해결해야 한다고 한다:
> 1. 3D-oriented few step generator는 충분히 robust한 prior와 강한 generative 능력이 필요하다.
> 2. high-quality multi-view dataset은 충분치 않기 때문에 기존의 style, object, camera 궤적과 같은 변수들을 handling 하는 전략을 develop 해야한다.

#### Dual-mode Pre-training

이 challenge들을 해결하기 위한 기반을 다지기 위해 framework를 먼저 제안한다.

여기서 dual-mode란 아래의 두 가지 모드를 의미한다:

> 1. MV-oriented mode: multi-view image를 예측하는 것
> 2. 3D-oriented mode: 중간 feature에서 3DGS를 직접 만들고 렌더링하는 것

먼저, training dataset에서 $X = \{X_1, X_2, ... , X_V\}$ multi-view image들을 가져온다. 그리고, $C=\{C_1, C_2, ... , C_V\}$ 각 view에 해당하는 camera parameter도 가져온다. 추가로, $y$라는 condition(text prompt, single-view image 등)이 들어간다.

이렇게 입력이 준비가 되면,

$$
Z = E(X)
$$

입력 multi-view images $X$를 VAE eoncoder $E$에 넣어서 latent로 변환한다. 그러면 아래 처럼 multi-view data 한 배치에 대한 latent 집합이 완성된다:

$$
Z = \{Z_1, Z_2, Z_2, ...\}
$$

그리고 나서, 일반적인 diffusion training처럼 random timestep $t$를 선택하고 noise를 넣는다.

$$
Z_t = \alpha Z + \sigma_t \epsilon
$$

예상할 수 있겠지만, 그러면 학습으로 사용되는 입력 multi-view image가 아닌 noisy multi-view latent인 $Z_t$를 사용하게 된다. 그러면, denoising network에 최종적으로 들어가는 입력은:

$$
(Z_t, C, y) \rightarrow \text{Denoising Network}
$$

순서대로, noisy latent, camera parameter, condition 이 3개가 들어가게 된다. 여기서 camera parameter를 표현하는 방식에는 **Plücker Coordinates raymap**를 사용하며, 이는 3D generation 분야에서 multi-view를 생성할 때 camera paramter를 많이 사용되는 방식이다.

Denoising Network는 **Diffusion Transformer (DiT)**를 기반으로 하며, **3D attention block**이 추가적으로 들어간다. 이 network는 두 가지 결과를 출력하려고 한다:

$$
\hat{Z}_{MV}, F
$$

순서대로, clean multi-view latent(MV-oriented mode), multi-view scene infromation 을 담고 있는 auxiliary feature 이다. 이 중에서 후자의 경우에는 이후에 3DGS decoder로 보내 3D Gaussian을 만든다(3D-oriented mode). **즉, 여기서부터 두 개의 branch로 나뉘어서 진행된다.**

먼저, MV-oriented mode의 경우, DiT가 $Z_t, C, y$를 받아 $\hat{Z}_{MV}$를 예측한다. 그리고 ground-truth claen latent $Z$와 비교한다:

$$
\mathcal{L}_{MV} = \mathbb{E})_{X, t, \epsilon, y, C} [\lVert Z - \hat{Z}_{MV} \rVert ^ 2]
$$

즉, noisy multi-view latent 를 clean multi-view latent로 변환하는 것을 배우는 diffsuion objective다. 중요한 건, MV-oriented mode가 실제로 만드는 건 3D representation이 아니다. **각 camera view에 해당하는 이미지를 직접 생성**하는 것이다. 그러므로, 각 view가 diffusion에 의해 이미지 공간에서 생성되므로, view 1과 view 2가 동일한 3D geometry에서 나온다고 보장되지 않는다.

> 즉, 순수 diffusion의 생성 능력에 따라 high quality를 출력할 수 있지만, multi-view inconsistency라는 문제가 생길 수 있다.

이를 해결하기 위해 또다른 branch인 3D-oriented mode가 등장한다. DiT의 intermediate/output feature인 $F$를 별도의 3DGS decoder $D_G$에 넣는다:

$$
D_G(F) = \{\tau, q, s, \alpha, c\}
$$

decoder가 내놓는 결과는 순서대로, depth, rotation quaternion, scale, opacity, spherical harmonics coefficietns다. 좀 더 간단하게, 깊이, 회전, 크기, 불투명도, 색상에 관련한 parameter를 내놓는다고 생각하면된다. 이는 3D Gaussian parameter를 제공한다고 생각하면 된다.

우리는 3DGS에 익히 알고 있다면, 원래 보통 3DGS primitives 라고 한다면, gaussian의 position인 $\mu$가 있어야 한다는 것을 눈치챌 수 있다. 그런데 위의 parameter 중에는 position에 관련한 parameter가 없고, 대신 그 자리에 depth $\tau$가 있다. 저자들은 이 position의 parameter를 바로 정하는 방식이 아니라 이 depth라는 parameter를 사용해 position parameter를 예측한다:

$$
\mu = o + \tau d
$$

여기서, $o$는 camera origin(카메라 위치, 보통 3D 공간 상에서 원점), $d$는 ray direction(카메라가 바라보는 방향, 즉 우리가 바라보는 방향), $\tau$는 예측된 depth를 의미한다. 개념적으로, 해당 수식이 각 pixel 단위에서 일어나며, 각 픽셀에 대응하는 3D gaussian들을 lifting한다.

그러면, 최종적으로 우리가 하는 Gaussian parameter를 완성할 수 있다:

$$
G = \{\mu, q, s, \alpha, c\}
$$

그러면, 우리는 3DGS renderer $R$를 이용해서 해당 novel camera view에 해당하는 장면을 rendering 할 수 있다:

$$
R(G, C_{novel})
$$

그러면 objective가 무엇인지를 생각해야한다. 정확하게는 3D Gaussian에 관련한 ground-truth가 존재하기 힘들다. depth를 추정하는 모델도 각각은 각 픽셀단위로 depth를 다르게 추정한다. 사용하는 depth model이 달라 depth 추정이 다르다면, 그에 따라 가우시안의 position은 달라질 것이다. 그런데, 결국에는 사용자가 장면을 듣고 3D처럼 보인다라는 느낌이 들면 그게 정답이다라고 볼 수 있는 것이다. 즉, 해당 관련한 파라미터들의 ground-truth가 없다:

$$
\mu_{GT}, q_{GT}, s_{GT}
$$

그래서 대부분의 3D Generation 논문들은 이 rendering(특정 view에 관한 장면을 캡처하는 것)을 사용해서 **rendering supervision**을 사용한다:

$$
\mathcal{L}_{3D} = \mathbb{E} [\lVert X_{novel} - R(G, C_{novel}) \rVert ^2]
$$

여기서, 더 특이한 점은 보통의 논문들을 camera view들(pose 1 ~ pose n)을 파이프라인 실행 전에 정해놓고, 해당 view들에 관해서만 supervision을 진행하는데 해당 논문의 경우에는 다른 방식으로 접근한다.

> novel-view에서 rendering하고 novel-view를 대상으로 supervision을 진행한다.

이렇게 하는 이유는 입력한 view에서만 reconstruction loss를 걸게 되면, 입력 view 에서만 맞게 보이고, 다른 view에서는 3D 구조가 이상할 수 있다. 그러나 입력 view 이외의 view에서도 맞게 보이게 novel-view reconstuction loss를 걸게 되면, 여러 관점에서 일관되게 배치되어야 하므로, **3D consistency constraint**가 생기게 된다. 여기서, 우리는 이전에 정해놓은(training dataset이 정해놓은, 동일한 Camera parameter $C$) 각 view에서 3D consistent한 rendering multi-view를 얻게 된다.

MV-oriented branch에서는 결과적으로 clean multi-view latent $\hat{Z}_{MV}$를 만들어냈다. **해당 3D-oriented branch 에서는 비슷하게, $\hat{Z}_{3D}$를 만들어낸다.** 이는 rendering multi-view에 단순히 VAE Encoder $E$를 거쳐 나온 latent다:

$$
\hat{Z}_{3D} = E(R(G, C))
$$

#### cross-mode post-training

이제 다음으로 논문은 distillation 방식을 통해 few-step 3D Scene을 생성할 수 있도록 학습을 하려고 한다.

우리는 앞서 preliminary 섹션에서 DMD에 대해 배웠고, 그에 대해 

$$
\mu_{real}, \mu_{fake}, G_{\theta}
$$

가 존재한다는 것을 알 수 있었다. 이 파이프라인에서는 각각 MV-oriented mode(teacher), 3D student distribution의 fake score를 추정하는 model, 3D-oriented few-step generator(student)라고 이해하면 된다.

여기서 $\mu_{real}$는 frozen 된 상태로 사용된다. 우리는 앞서 3D-oriented mode에서 많은 step으로 돌렸는데, 이를 그대로 여기서 많은 step으로 돌리는 것이 아니라, 앞의 3D-oriented mode를 지금 few-step student의 initialization으로 사용한다. **즉, 전의 dual-mode pretraining 섹션에서의 architecture에 다른 architecture가 새로 추가되는게 아니다.**

논문은 이전 섹션의 3D-oriented mode의 파이프라인을 거의 그대로 물려 받는다:

$$
(\{Z_{t_i}, t_i, y, C\} \rightarrow DiT \rightarrow F_i \rightarrow D_G \rightarrow G_i) \rightarrow R(G_i,C) \rightarrow E(R(G_i,C)) \rightarrow \text{noise injection} \rightarrow Z_{t_{i+1}}
$$

여기서, **괄호 안에 있는 파이프라인이 3D-oriented generation process의 denoising $G_{\theta, 3D}$이라고 보면 되고**, 이걸 4번의 step 만에 처리하는 걸 목표로 한다. 여기서 헷갈리면 안되는 것이 DiT의 timestep이 4라는게 아니라, 해당 프로세스가 4번의 횟수로 진행되는 것을 말한다. 해당 논문에서는 timestep이 $t_i=\{1000, 900, 759, 500\}$ 으로 이루어져 있다고 한다. 해당 프로세스의 step 한 번의 끝에는 매번 noise injection이 수행되고, 이를 다음 step $t_{i+1}$에 넘겨준다. 

이렇게 파이프라인의 흐름을 정해놓고, DMD2 기법을 사용한다. 원래 3D-oriented branch 에서는 many step을 사용했는데, 바로 4-step branch로 바꿔버리게 되면, quality가 보장되지 않는다. 그래서 이 few-step student를 다시 학습해서 few-step으로도 좋은 output distribution에 도달할 수 있도록 학습해야한다.

여기서 이제 frozen된 high-quality MV distribution의 score인 $s_{real}$을 추정하는 $\mu_{real}$을 사용하여 teacher model로 사용한다. 그리고, $\mu_{fake}$는 student인 $G_{\theta, 3D}$를 계속 추적하면서 student가 생성하는 distribution인 $p_{fake}$의 score인 $s_{fake}$를 추정한다. 당연히 student가 바뀔 때마다 $p_{fake}$도 바뀌므로 $\mu_{fake}$도 계속 update 된다.

즉,

$$
s_{MV} - s_{\text{current 3D student}} = s_{real} - s_{fake}
$$

이 gradient를 사용해서 $G_{\theta, 3D}$를 update한다. 이렇게 되면, 

$$
p_{\text{3D student}} \rightarrow p_{MV}
$$

가 되도록 한다. 추가적으로 DMD2 loss는 DMD loss에 GAN loss를 합한 버전이다:

$$
L_{DMD2} \approx L_{DMD}+\lambda_{GAN}L_{GAN}
$$

여기서, $\lambda$는 R1 regularization이다. 개념적으로 discriminator는 

$$
D(X_{real}) \rightarrow 1
D(X_{fake}) \rightarrow 0
$$

가 되도록 학습되고, generaotr $G_{\theta, 3D}$는

$$
D(X_{fake}) \rightarrow 1
$$

이 되도록 학습된다.

그런데, 문제는 이렇게 되면 결국에 3D student가 3d consistency에 대한 강점을 잃어버리고 MV-oriented 쪽으로 끌려갈 위험성이 있다는 것이다. 물론, 3d-oriented branch는 하나의 share된 3D Scene에서 렌더링되어 이 gradient가 update의 주체가 되어 3d consistency를 유지할 것처럼 보인다. 

논문의 저자들을 아래의 3개의 이유로 3d consistency가 유지된다고 말하지만, 해당 리뷰를 적는 본인은 **해당 이유만으로 증명되기는 힘들다고 본다**:

> 1. 3D supervision (이전 섹션에서 novel-view에 대한 rendering supervision을 수행하는 것)
> 2. pretrained 3D-oriented weights로 시작하는 것
> 3. 매 step 마다 계속 $G_i \rightarrow \text{Render}$ 하는 것

**여기서 이 논문의 리뷰를 적는 작성자가 생각하기에 2번이 중요해 보이는데, 이 weights가 결국에는 MV-oriented weights로 계속 변화할텐데, 그러면 3d consistency의 강점을 어떻게 안 잃어버리고 유지할 지 의문이다.** MV쪽은 고정되어 있는 상태에서 3D쪽만 update하여 진행하는 방식이기 때문이다.

추가적으로 저자들은 **Cross-Mode Consistency Loss**를 추가한다. 여기서, 3D Student가 사용하는 DiT backbone을 공유하는 MV_oriented student branch를 사용한다.

$$
\begin{aligned}
\hat{Z}_{3D} &= E(R(G_{\theta, 3D}(Z_t, t_i, y, C), C)) \\
\hat{Z}_{MV} &= G_{\theta, MV}(Z_t, t_i, y, C)
\end{aligned}
$$

를 얻어서

$$
\mathcal{L}_{CMC} = \lVert \hat{Z}_{3D} - \hat{Z}_{MV} \rVert ^2
$$

로 맞춘다. 이는 DMD2를 진행했을 때 생기는 floating artifact와 불안정한 3D prediction을 줄이기 위한 보조 loss로 MV-oriented student도 낮은 frequency로 어데이트하고, 두 mode의 prediction을 맞춘다.

결론적으로, 3D student는 해당 signal을 받게 된다:

$$
\underbrace{\mathcal{L}_{\mathrm{DMD}}}_{\text{MV teacher distribution으로 이동}}
+
\underbrace{\mathcal{L}_{\mathrm{GAN}}}_{\text{rendering을 real/high-quality하게}}
+
\underbrace{\lambda \mathcal{L}_{\mathrm{CMC}}}_{\text{3D branch 안정화}}
$$

#### Out-of-Distribution Data Co-Training

이렇게 학습을 진행하게 되면, 하나의 사소한 문제가 또 발생하게 된다. Flashworld의 경우, MVImgNet, RealEstate10K, DL3DV10K를 기반으로 학습되었다. 그런데, DiT의 경우에는 이 이외에도 대규모 image와 video data로 학습되었기 때문에 앞의 3개의 multi-view datasets을 제외한 다른 데이터가 들어와도 robust하게 대처할 수 있는 반면, 3D branch에 있는 3DGS Decoder는 3개의 multi-view datasets을 제외한 다른 데이터에 대해 robust 하지않다.

> 즉, 3DGS Decoder $D_G$는 multi-view distribution만 경험했다.

그렇다면 3DGS Decoder가 받은 input의 distribution을 넓혀주면 해결된다. 그러기 위해서, 먼저 해당 파이프라인에서 처음에 입력하는 데이터를 multi-view dataset이 아니라 single image나 text만 넘겨주게 된다.

text 혹은 single image $y$만 넘겨주는 경우에는 이에 camera trajectory $C$를 같이 넘겨주고, DMD 방식으로만 학습을 진행하게 된다. 당연히 실제 Ground truth가 없으므로 GAN Loss는 사용하지 않고 DMD와 CMC loss 만으로 학습을 진행한다:

$$
(y, C_{random}) \rightarrow DiT \rightarrow F \rightarrow D_G \rightarrow G
$$

여기서 Camera trajectory는 REalEstate10K, WorldScore 같은 이미 있는 multi-view dataset의 trajectory를 사용한다. 그리고, 해당 학습은 pre-training 단계가 아닌 post-training 단계에서 multi-view data와 ood data를 2:1의 비율로 섞어서 사용한다.





