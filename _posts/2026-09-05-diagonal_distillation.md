---
title: "[Paper Review, KR] Streaming Autoregressive Video Generation via Diagonal Distillation"
date: 2026-09-15
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

<p align="center">
  <img src="/assets/images/posts/2026-09-05-diagonal_distillation/1789460432729.png" width="70%">
</p>

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

Chunk 4 이후에는 2-step으로 고정한다.

$$
C_k = T(\tilde{X}_{k-1})
$$

$$
X_k = D_2(D_1(Z_k | C_k)| C_k)
$$

즉 이전 chunk 정보를 이용해 $T$ 인 conditioning module을 이용해서 conditioning $C_k$를 만들고, 두 번만 denoising $D_1, D_2$ 한다.

그런데, 단순히 "5 $\rightarrow$ 4 $\rightarrow$ 3 $\rightarrow$ 2" 방식으로 step 수만 줄이면 문제가 있따. Chunk $k$를 만들 때 이전 Chunk $k-1$의 완전히 clean한 결과만 condition으로 준다고 한다면, 

$$
\text{Chunk} \; k-1 \rightarrow \text{clean output} \rightarrow \text{KV cache} \rightarrow \text{Chunk k}
$$

그런데 Chunk $k$는 현재 diffusion/flow denoising의 특정 noise level에 있다. 그러면 모델은 clean context를 보면서 동시에 다음 chunk는 지금 어느 noise level 있어야 하는지까지 암묵적으로 판단해야한다. 

논문이 이에 대해 **implicit next-noise-level prediction** 문제라고 표현하고, 이 prediction에 작은 오차가 생기면 autoregressive하게 누적될 수 있다고 말한다. 그래서 저자들은 clean previous chunk를 그대로 사용하는 것이 아니라 일부러 적당한 noise를 다시 넣어서 사용한다(*정확히는 전 Chunk에서 denoising 중에 나오는 intermediate latent에 nosie를 조금 넣고 이를 KV cache에 저장하는 방식으로 진행한다*):

$$
\tilde{X}_{k-1} = \sqrt{\alpha_{k-1}} X_{k-1} + \sqrt{1-\alpha_{k-1} \epsilon}, \; \epsilon \sim \mathcal{N}(0, I)
$$

$X_{k-1}$은 이전 chunk의 clean output이고, $tilde{X}_{k-1}$은 거기에 controlled noise를 추가한 상태다. 그러면 흐름은 이렇게 바뀐다.

$$
\text{Chunk} \; X_{k-1} \rightarrow \text{noise 추가} \rightarrow \text{partially noisy} \; \tilde{X}_{k-1} \rightarrow \text{KV cache} \rightarrow \text{Chunk k}
$$

그런데 의문이 들 수 있다. 왜 noisy context가 이를 해결할 수 있는지에 대해서다. 핵심은 **현재 chunk와 이전 context의 denoising 상태를 더 잘 맞춰주는 것**이다. 

Clean context를 그대로 주게 된다면, Previous context(Clean)와 Current chunk(Noisy) 간의 noise-level mismatch가 크다. 반면 Diagonal Forcing에서는 Previous Context에 noise를 추가함으로써 Current chunk 간의 noise-level이 비슷해진다.

그래서 여기까지 본다면, **Diagonal Denoising**은 step 수를 progressive하게 reduction해서 계산량을 줄이는 쪽이라면, **Diagonal Forcing**은 이렇게 step을 줄여도 품질이 무너지지 않도록 연결해서 temporal coherence와 long-term stability를 유지하는 역할을 수행한다.

#### Flow Distribution Matching

여기서 앞의 Diagonal Denoising 때문에 발생하는 하나의 문제를 또 발견한다. 이는 **few-step으로 줄였을 때 motion이 약해지는 문제**다. 논문에서는 이를 **motion attenuation**이라고 표현하고, teacher가 충분한 denoising step을 거치면 물체가 프레임 사이에서 크게, 자연스럽게 이동하는데, student를 2-step 정도로 강하게 줄이면 spatial appearance는 그럴듯해도 움직임의 크기가 작아진다는 것이다.

저자들은 이 원인을 **denoising trajectory가 너무 짧아지면서 temporal dynamics를 충분히 복원하지 못하는 것**으로 설명한다. 그래서 단순히 DMD로는 부족하고, **motion distribution** 자체도 teacher와 맞춰야 한다라고 말한다:

$$
E_{motion} = D_{KL} (p_{teacher}(F(x) | x_t) || p_{student}(F(x) | x_t))
$$

여기서 $F(x)$는 video에서 추출한 motion flow feature다. 핵심은 **"teacher와 student가 비슷한 frame을 만드는지만 보는 것이 아닌, 두 모델이 만들어내는 motion feature의 분포도 비슷하게 만든다"**라는 것이다.

DMD는 기존에

$$
p_{student}(x) \approx p_{teacher}(x)
$$

를 맞추는 방향이라면, 여기에 추가로

$$
p_{student}(F(x)) \approx p_{teacher}(F(x))
$$

를 맞춘다고 생각하면 된다. 추가적으로 Flow Distribution Matching도 DMD 방식으로 학습한다. 

$$
\nabla_{\phi} L_{DMD}^{flow} = \nabla_{\phi} KL(p_{gen,flow,t} || p_{data,flow,t})
$$

즉, spatial domain에서 했던 distribution matching을 motion feature domain으로 확장한다. 그래서

$$
s_{data}^{flow} - s_{gen}^{flow}
$$

의 차이를 이용해 student를 업데이트한다.

기존의 DMD의 경우에는 student image dstribution과 teacher image distribution을 matching하는 방법이었다면, Flow DMD의 경우에는 student motion distribution과 teacher motion distribution을 matching하는 방법이라고 보면된다.

그런데, 여기서 중요한게 있다. 여기서 말하는 **FLOW**는 RAFT 같은 optical flow를 말하는 것이 아니다. 대신 video diffusion의 latent space에서 직접 motion feature을 뽑아 사용한다. 정확히는,

$$
(\text{Latent frame t} - \text{Latent frame t+1}) \rightarrow (X_{t+1} - X_t) \rightarrow \text{Convolution layers} \rightarrow \text{MLP} \rightarrow \text{Motion feature} \; F(x)
$$

즉, consecutive latent의 차이를 먼저 계산하고, 여기에 convolution을 적용해 local motion pattern을 추출한 뒤 MLP로 feature adaptation을 한다. 그래서 여기서의 FLOW는 **latent difference 기반의 learnable motion representation**에 가깝다.

Flow DMD의 목표는

$$
\nabla_{\phi}L_{DMD}^{flow} = \mathbb{E}[\nabla_{\phi}D_{KL}(p_{gen,flow,t} || p_{data,flow,t})]
$$

로 student가 만드는 motion-feture distribution을 real/teacher 쪽 motion-feature distribution에 가깝게 하겠다는 것이다. 그러면 flow score는

$$
s^{flow} (x_t, t) = \nabla_{x_t} log p(F(x) | x_t)
$$

로 정의한다. 이는 일반 diffusion과 다르게 **현재 noisy video latent $x_t$를 어느 방향으로 변화시키면 원하는 motion feature $F(x)$가 더 그럴듯해지는지**를 나타내는 graident라고 보면된다.

$$
\nabla_{\phi} L_{\mathrm{DMD}}^{\mathrm{flow}}
\approx
-\mathbb{E}_{t}
\left[
\int
\left(
s_{\mathrm{data}}^{\mathrm{flow}}
\left(
\Psi(G_{\phi}(\epsilon), t), t
\right)
-
s_{\mathrm{gen},\phi}^{\mathrm{flow}}
\left(
\Psi(G_{\phi}(\epsilon), t), t
\right)
\right)
\frac{dG_{\phi}(\epsilon)}{d\phi}
\, d\epsilon
\right].
$$

이를 통해 결국 $s_{data}^{flow} - s_{gen}^{flow}$, 즉 real/teacher motion distribution이 원하는 방향과 student motion distribution이 원하는 방향의 차이를 계산해서 Generator를 학습한다.

여기서 $\Psi(x,t)$는 video sample $x$를 timestep $t$의 noisy state $s_t$로 만드는 forward noising 연산이다(*아마도 기존 DMD 식에서는 forwarding noising 연산을 $F$로 표현하는데, 여기서는 flow extractor를 $F$로 표현해서 $\Psi$를 사용하는 것 같다*). 
그리고 이에 맞는 regression loss도 기존 전체 loss에 추가한다.

$$
L_{reg}^{flow} = \mathbb{E} [\lVert F(G_{\phi}^{teacher}) - F(G_{\phi}^{student}) \rVert _2^2]
$$

즉 같은 조건에서 teacher와 student가 생성한 video를 각각 teacher/student flow extractor$F(\cdot)$에 넣고, 맞추는 방법이다. 그러면 loss 최종 전체 식이

$$
L_{\mathrm{Total}}
=
\lambda_{\mathrm{spatial}}
L_{\mathrm{DMD}}^{(\mathrm{grad})}
+
L_{\mathrm{reg}}
+
\gamma
\left(
\lambda_{\mathrm{flow}}
L_{\mathrm{DMD}}^{\mathrm{flow},(\mathrm{grad})}
+
L_{\mathrm{reg}}^{\mathrm{flow}}
\right).
$$

이렇게 된다.


