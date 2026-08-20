---
title: "[Paper Review] SonoWorld: From One Image to a 3D Audio-Visual Scene"
date: 2026-08-13
categories:
  - 3D Vision
tags:
  - audio
  - 3D Generation
  - Interaction
---

> **Paper Information** \\
> **Title:** SonoWorld: From One Image to a 3D Audio-Visual Scene \\
> **Authors:** Derong Jin, Xiyi Chen, Ming C. Lin, Ruohan Gao \\
> **Venue:** CVPR 2026 \\
> **Link:** [[Paper](https://arxiv.org/pdf/2603.28757)], [[Project](https://humathe.github.io/sonoworld/)], [[GitHub](https://github.com/HuMathe/sonoworld)]

## Teaser Image

<p align="center">
  <img src="/assets/images/posts/2026-08-13-sonoworld/1786602166402.png" width="50%">
</p>

## Introduction

기존의 몰입적인 3D Scene을 만드는 방법론들 그리고 기법들은 사실적인 3D world를 singel image로부터 만들 수 있지만, **이는 청각적인 경험은 결여되어 있어 인지적으로 불완전하다**고 논문의 저자들은 주장한다.

그래서 논문의 저자들은 single image로부터 training free 방식으로 <mark><span style="color: red;">일관적인 3D Scene을 만드는 동시에 공간적인 sound field를 만들어 user interaction까지 제공</span></mark>하는 것을 목표로 한다.

이를 위해서 3가지 challenge가 존재한다:

1. scene-level의 audio generation은 점 음원(point source), 영억 음원(areal source) 그리고 주변 환경 음원 등 다양한 타입과 스케일의 음원으로 구성되어야한다.
2. 시스템은 객체가 무엇인지, 어떻게 들리는지, 얼마나 시끄러울지 등을 유추할 수 있도록 총제적이고 의미론적인 scene에 대한 이해가 필요하다.
3. 생성된 모든 소리는 이미지로부터 추론된 타당한 3D 위치에 있어야 하며, 인지적으로 실제와 같은 공간적효과로 렌더링되어야 한다.

관련된 연구 분야들에서는 아래와 같은 한계점들이 존재한다(**연구 동향 확인이 아니라면 건너뛰어도 됩니다.**):

> - 해당 논문은 일관성 있는 360도 파노라마를 outpainting하고, depth alignment와 가우시안 최적화를 통해 이를 3D 로 lifting하여 Scene을 완성하는 기존의 Panoramic method를 따라가지만, 시각 정보와 함께 **spatial audio**를 공동으로 모델링한다는 점에서 차별화된다.
> - 기존 공간 음향 생성 분야에서 진행된 연구인 Sonic4D의 경우 공간 음향 생성을 4D Dynamic Scene과 결합하는 수준까지 나아갔으나, 단일 객체, 좁은 시야각, 그리고 오프라인 처리에만 국한되어 작동한다는 한계가 있는데, SonoWorld는 이를 넘어 **점(point), 영역(areal), 주변(ambient) 음원들과 실시간적인 exploration을 지원**한다. 
> - Visual Scene에서 sound localization과 Audio-Visual Source Seperation 분야에서도 각각 단순히 시각적 장면 내에서 소리가 나는 위치를 추적하는게 아니라 VLM을 활용해서 생성된 3D 파노라마 장면 속의 잠재적인 음원들을 찾고 이를 공간 상에 위치시키거나 단순히 혼합 오디오를 개별 component 오디로로 분리하는데 그치지 않고, 생성된 3D 시각 장면 내에 위치된 모든 음원에 대한새로운 공간 음향 생성을 직접 구현하는 것으로 차별점을 명시한다.

## The Image2AVScene Task

<details markdown="block">
<summary>Ambisonics Concept</summary>

#### 1. Spatial Sound representation

**Ambisonics**는 "내 주변 360도 공간에서 어느 방향으로 어떤 소리가 들리는가"를 몇 개의 오디오 채널로 압축해서 표현하는 방법"이다. 

SonoWorld는 3D Scene 안에 놓인 sound source들의 위치를 이용해 이 Ambisonics 채널들을 직접 계산한다.

차근 차근 이해해보자.

우리가 듣는 일반적인 mono audio(공간 음향이 아님)는 

$$
a(t)
$$
처럼 시간에 따른 소리만 표현한다.

공간음향에서는 소리가 어느 방향에서 오는지도 알아야 하므로:

$$
a(\theta, \phi, t)
$$

로 표현하여 $$\theta=azimuth, 좌우방향$$, $$\phi=elevation, 상하 방향$$를 파라미터로 사용한다. 즉, 구 모양을 생각하면 편하게 이해할 수 있다:

![alt text](/assets/images/posts/2026-08-13-sonoworld/1786949404525.png)

#### 2. Spherical Harmonics

문제는 360도 모든 방향의 소리를 그대로 저장하게 되면, 실시간 소리 렌더링도 힘들 뿐더러 저장용량도 커지는 문제가 있다.
대신, Ambisonics는 몇 개의 기본적인 공간 패턴을 이용해서 전체 sound field를 표현한다.

자 여기서 기본 공간 패턴을 우리는 이렇게 표현하겠다:

$$
Y_l^m(\theta, \phi)
$$

이 식은 우리가 특정 방향에 대한 $$\theta, \phi$$ 를 넣어줬을 때, 해당 방향에 대한 가중치를 주는 함수다. 즉, 해당 한수는 360도 각 방향에 대해 서로 다른 가중치를 주는 패턴인 것이다.

물론, 이를 설명하기 위해서는 Fourier Transform이라는 걸 설명하는게 좋겠지만, 그렇게 되면 너무 길어지기 때문에 이를 그냥 **"특정한 3D 방향 패턴을 나타내는 basis"** 라고 생각해주면 좋다. basis는 단순히 기본 선형대수학에서 나오는 표현이니 넘어가도록 하겠다.

간단하게,
> Foureier Transform에서 sin/cos 이 basis 역할을 하듯, 구면 공간에서는 spherical harmonics가 basis 역할을 수행한다.

#### 3. Ambisonics coefficient

논문의 Equation (1)은 

$$
a(\theta, \phi, t) \approx \sum_{l=0}^L \sum_{m=-l}^l Y_l^m (\theta, \phi)a_{l,m}(t) = y_L(\theta, \phi)^T \mathcal{a}_L(t)
$$

이다. 

여기서 $$Y_l^m (\theta, \phi)$$ 는 공간 방향 패턴이고, $$a_{l,m}(t)$$는 그 패턴에 대응하는(Ambisonics로 인코딩된) audio waveform이다.

즉, Ambisonics 에서는 360도 sound field 전체를 직접 저장하지 않고,

$$
a_{0,0}(t), a_{1,-1}(t), a_{1,0}(t), a_{1,1}(t) ...
$$

같은 여러 개의 audio channel로 저장한다.

이것들이 논문에서 말하는 Ambisonics coefficients, 즉 Ambisonics channel이다.

단순하게, Equation (1)의 의미는

> Directional Sound = Spatial Basis (Weights) 값 X Ambisonics Channels에 담긴 waveform

라고 이해하자.

#### 4. Ambisonics order과 FOA

Ambisonics order 인 $$L$$은 공간 표현 정밀도를 결정한다.

논문에서 주로 사용하는 것은 $$L=1$$인 **First-Order Ambisonics, FOA**다.

Ambisoncis channel 수는 $$(L+1)^2$$ 개 이므로 FOA 에서는 4개의 audio channel을 사용한다(Eq(1)을 보면 알겠지만, L=1이면 $$\sum$$ 에 의해 나오는 결과값은 총 4개다).

즉 이 논문에서는 $$a_1(t) \in \mathbb{R}^4$$ 라고 생각하면 된다.

#### 5. Point source를 Ambisonics로 만들기

이 논문에서 가장 중요한 수식은 Equation (1)이 아닌 Equation (2)다.

$$
\mathcal{a}_L^{\text{single}} (t) = \sigma(d)a_{\text{src}}(t)y_L(u)
$$

* $a_{\text{src}}(t)$는 원래 sound waveform.
* $u$는 listener에서 source를 바라보는 방향.
* $\sigma(d)$는 거리에 따른 sound attenuation.

간단하게, 아래와 같이 이해하면 된다.
> **Ambisonics = 원래 소리 $\times$ 방향정보 $\times$ 거리 효과**

#### 6. 3-DOF 와 6-DOF

일반적인 Ambisoncis recording은 한 위치에서 측정한 sound field 이기 때문에

$$ 
R \in SO(3)
$$

로 yaw, pitch, roll 을 수행하는 3D rotation인 3-DOF 다.

하지만 listener가 다른 위치로 이동하면 source 까지의 방향, 거리가 달라진다.

Sonoworld는 3D 위치를 알고 있기 때문에, listener가 이동하면 새로운 위치에서 u와 d를 다시 계산하여 Equation (2)를 다시 적용할 수 있다.

따라서,
> 3 rotation DoF + 3 translation DoF = 6-DoF

인 6-DoF exploration이 가능해진다.

#### 7. 헷갈릴 수 있는 부분

Equation (2)는 실제 원본 source waveform인 mono audio에 source의 방향과 거리 정보를 반영하여 여러 개의 Ambisonics channel waveform으로 인코딩하는 식이다.

Equation (1)은 그렇게 만들어진 여러 Ambisonics channel waveform을 특정 방향 (θ,ϕ)의 Spherical Harmonics 값으로 가중합하여, 그 방향을 바라보는 virtual microphone의 directional waveform을 얻는 식이다.

방향 $\mathbf{u}_{\mathrm{query}}$를 바라보는 virtual microphone의
waveform은

$$
a_{\mathbf{u}_{\mathrm{query}}}(t)
=
\mathbf{y}_L(\mathbf{u}_{\mathrm{query}})^T
\mathbf{a}_L(t)
$$

이다.


Equation (2)를 Equation (1)에 대입하면 조금 더 쉽게 이해할 수 있는데,

$$
\mathbf{a}_L(t)
=
\sigma(d)\,
a_{\mathrm{src}}(t)\,
\mathbf{y}_L(\mathbf{u}_{\mathrm{src}})
$$

를 Equation (1)에 대입하면,

$$
a_{\mathbf{u}_{\mathrm{query}}}(t)
=
\mathbf{y}_L(\mathbf{u}_{\mathrm{query}})^T
\left[
\sigma(d)\,
a_{\mathrm{src}}(t)\,
\mathbf{y}_L(\mathbf{u}_{\mathrm{src}})
\right]
$$

$\sigma(d)$와 $a_{\mathrm{src}}(t)$는 scalar이므로 밖으로 빼면,

$$
a_{\mathbf{u}_{\mathrm{query}}}(t)
=
\sigma(d)\,
a_{\mathrm{src}}(t)\,
\mathbf{y}_L(\mathbf{u}_{\mathrm{query}})^T
\mathbf{y}_L(\mathbf{u}_{\mathrm{src}})
$$

따라서

$$
\boxed{
a_{\mathbf{u}_{\mathrm{query}}}(t)
=
\sigma(d)\,
a_{\mathrm{src}}(t)\,
\underbrace{
\mathbf{y}_L(\mathbf{u}_{\mathrm{query}})^T
\mathbf{y}_L(\mathbf{u}_{\mathrm{src}})
}_{\text{directional matching term}}
}
$$

여기서,

$$
\mathbf{y}_L(\mathbf{u}_{\mathrm{query}})^T
\mathbf{y}_L(\mathbf{u}_{\mathrm{src}})
$$

가 크면, virtual microphone이 바라보는 방향과 실제 source 방향이 Ambisonics basis 상 잘 맞아 소리가 크게 들리는 등의 영향이 일어난다.


</details>

#### Task Goal

$$
\mathcal{G}: I \rightarrow \{V(p), A(p,t)\}
$$

해당 수식을 보다시피, 특정 observer의 pose p가 주어졌을 때, visual representation인 $$V(p)$$와 $$A(p,t)$$를 추정해 framework인 $$\mathcal{G}$$를 만들어내는 것이다. 

이 $$V(p)$$ 같은 경우는 3D guassian splats 를 이용해 표현하고, $$A(p,t)$$의 경우에는 point-cloud를 기반으로 하는 ambisonics rendering을 통해서 표현한다:

$$
a_L(t) = A(p,t) \in \mathbb{R}^{(L+1)^2}
$$


## Method & Technical Details

<p align="center">
  <img src="/assets/images/posts/2026-08-13-sonoworld/1787120630529.png" width="70%">
</p>

SonoWorld의 Key Tehcnique은 총 4개로 나뉜다.

> - **첫 번째로, single image를 입력으로 받아 360도 파노라마를 생성하고 이를 3D Scene에 Lift하는 것**
> - **두 번째로, reconstructed된 3D 공간에서 소리가 있을 법한 실체를 식별하고 위치를 결정하는 것**
> - **세 번째로, spatial sound field를 생성하는 ambisonics encoder를 설계하는 것**
> - **네 번째로, 청취자 입장에서 어느 위치(pose)에서도 입체적인 binaural audio를 렌더링하는 것**

#### Panorama-Based Visual Scene Generation

먼저 single image를 받게 되면, 보통 기존의 기법들은 해당 이미지가 정면을 바라보고 수평 상태에 있다고 가정한다. 그러나, 이 가정은 single image가 기울어져 있거나 elevation이 다를 경우 misalignment 문제를 야기할 수 있다. 그래서, SonoWorld는 먼저 single image를 받아 이를 calibration하는 것부터 시작한다.

$$
(\phi,f) = Calib(I) ,
$$

입력 이미지 $$I$$로 GeoClib라는 모델을 사용해서 gravity direction과 camera FOV를 추정하여 $$(\phi, f)$$인 camera elevation과 FoV를 추정한다. 다음으로 perspective image $$I$$를 equirectangular panorama에 reprojection하기위해 $$\mathcal{W}_G$$ Gaussian Pyramid를 기반으로 하는 warping operator를 사용한다. 해당 operator는 다중 스케일 안티앨리어싱 샘플링을 수행하고, 이후에 warped image는 WorldGen이라고 하는 outpainting model $$g_{outpaint}$$을 사용해서 360도 파노라마를 생성한다: 

$$
I_{pano} = g_{outpaint}(\mathcal{W}_G(I, \phi, f))
$$

그리고 기존의 panorama-to-3D reconstruction method를 사용하여 완성된 파노라마를 3D Scene으로 lift한다:

$$
V = \mathcal{G}_v(I_{pano})
$$

여기서 사용하는 모델은 Marble model 이나 HunyuanWrold1.0을 사용하여 사실적인 3D 환경을 구성한다.

#### 360° Audio-Visual Semantic Grounding

이번에는 3D Scene 안에서 어떤 물체가 소리를 낼 수 있고, 그 물체가 정확히 3D 공간의 어디에 있는지를 찾아내는 단계다.

먼저, VLM한테 입력 이미지 $$I$$를 주고, 4가지의 정보를 얻어낸다.

> - sound source 후보 category들의 집합 $$C$$
> - point/clustered/ambient sound type 분류
> - MMAudio가 waveform을 생성할 때 사용할 text prompt
> - 각 sound source가 얼마나 크게 들려야 하는지를 결정하는 Amplitude equalization parameter

그럼 이제 우리는 category 집합 $$C$$를 가지고, 해당 sound category들이 panorama의 어느 위치에 있는 지 알아야 한다. 이를 위해서 open-vocabulary segamentation model인 X-Decoder를 사용한다.

그런데, X-Decoder 는 일반 perspective image를 사용하도록 학습되어 있기 때문에 파노라마 이미지인 $$I_{pano}$$를 여러 개의 겹치는 perspective FoV image로 잘라서 사용한다. 이렇게 tile image들을 category를 조건으로 각 tile image에 해당되는 category segmentation mask를 뽑는다:

$$
M_{OVS, c}
$$

그러나, X-Decoder가 여러 tile로 잘라서 처리하기 때문에 tile 경계에서 mask가 끊기거나, sky/ground처럼 큰 영역에 불안할 수 있기 때문에 파노라마 이미지에서도 물체의 영역 자체를 깔끔하게 찾아네는 SAM2를 이용해서 mask를 한 번 더 뽑는다. 물론, X-Decoder는 segmentation mask에 대해 영역이 어떤 물체인지 semantic label을 얻을 수 있고, SAM2는 그럴 수 없다는 서로의 장단점을 보완해주기도 한다.

즉, SAM2 는 class-agnostic sementation mask 인

$$
M_{pano}
$$

를 뽑아낸다.

그리고 X-Decoder의 결과를 SAM2 region에 confidence-weighted vote를 진행한다.

overlapping과 semantic confidence가 충분히 크다면 해당 SAM2 mask를 특정 카테고리의 mask로 채택한다. 그러면 최종적으로 category $$c$$마다 $$M_c$$를 얻어 최종적으로:

$$
M=\cup_{c\in C}M_c
$$

가 된다.

그러나, 아직 2D 위치일 뿐이라서 이를 3D 공간에 올려놓아야 한다. 그래서 우리가 이전에 만들어 놓은 3D Scene $$V$$가 있어 해당 scene으로 부터 depth map $$D$$를 렌더링하고, 각 sound source mask $$M_i$$와 depth $$D$$를 이용해서 

$$
P_i = Lift(M_i, D)
$$

를 수행한다. 그러면 하나의 source가 

$$
P_i = \{p_1, p_2, ... , p_N\}
$$

같은 3D point 집합으로 표현된다.

논문은 모든 source의 $$P_i$$를 모은 결과를 $$P$$라고 하고, 이것이 scene 내 sounding object들의 3D 위치를 나타낸다고 설명한다.

> 그런데 여기에서 SonoWorld github 코드를 보면, X-Decoder와 SAM2를 사용하는게 아닌 SAM3를 사용하는 걸 볼 수 있다. 아마도 이전에 X-Decoder와 SAM2를 사용했을 때는 IOU 방식으로 처리해서 보완된 mask를 얻었겠지만, SAM3는 semantic label과 정교한 영역 mask를 모두 뽑을 수 있기 때문에 이제 두 모델을 돌리고 추가 알고리즘을 사용해서 mask를 걸러내는 작업을 할 필요가 없어진 것 같다. 즉, **SAM3 자체가 text concept으로 instance를 찾고 segmentation할 수 있는 모델이기 때문에 기존 방식인 SAM2 와 X-Decoder를 같이 사용할 필요가 없어진 것이다.**

#### Ambisonics Encoding

이번 stage에서는 장면에 들어갈 소리를 생성하고 최종적으로 들릴 3D 입체 오디오 신호를 수학적으로 조립하는 것이다.

먼저 전에 VLM이 이미지 분석을 통해 제안한 텍스트 프롬포트를 오디오 생성 AI인 MM Audio에 주입하여, 각 object에 어울리는 소리($$a_{i,raw}$$)와 전체 배경 소리($$a_{global}$$)를 생성한다. 생성된 waveform의 volumne을 그대로 사용하지 않고, VLM이 예측한 sound energy ($$v_i$$)를 밑의 수식을 통해 이 파라미터 수치만큼 데시벨을 실제 물리 신호 스케일로 환산하여 곱해준다:

$$
a_i(t) = 10^{v_i/20}a_{i,raw}(t)
$$

이렇게 곱해주면 각 소리의 균형감 있는 볼륨 조절을 마친다.

그리고 이제 앞에서 우리가 global인 전체 배경 소리와 각 object의 소리를 분리했었는데, 여기서 global은 그대로 두고 grounding 된 source들을 $$\mathcal{O}$$라고 하고, 이를 $$\mathcal{O}_{point}$$와 $$\mathcal{O}_{cluster}$$ 로 나눈다.

개념적으로, point source는 공간적으로 작은 source로 하나의 위치로 대표해도 괜찮은 경우고, clustered surce는 공간적으로 넓게 퍼져있어 한 점에서 소리가 난다고 하기보다, 전체의 여러 위치에서 소리가 나는 경우로 보면된다.

sound field를 제공하기 위해서는 listener pose 도 중요하다. 결국 **user tracking이 가능**해야 사용자에게 입체감 있는 사운드를 제공할 수 있다. 그러기 위해서 listener가 어디에 있고 어느 방향을 바라보는 지(pose)를 이렇게 표현한다:

$$
p = [R, t] \in SE(3)
$$

여기서 t는 listener의 3D position, R은 listener의 roatation이다. 이로써 청자의 고개를 돌렸을 때, 그리고 이동했을 때 사운드가 어떻게 변할 지 표현할 수 있다. 여기에 더해 source와 listener 사이의 거리($$d$$)를 기반으로 감쇠효과를 만들어야 한다. 그 수식은 이와 같이 표현할 수 있다:

$$
\sigma(d) = \frac{e^{-\alpha d}}{d}
$$

여기서 d를 분모에 둠으로써 source가 멀어질수록 소리가 작아지는 distance attenuation 효과를 내고, 분자에 있는 $$e^{-\alpha d}$$를 통해 거리가 멀어질수록 공기를 지나며 소리가 추가로 감쇠되는 air absorption을 모델링할 수 있다.

즉, 멀리 있을수록 소시를 줄이는 함수를 구현했다고 보면 된다.

그럼 이제 우리는 Ambisonics를 세 종류의 soruce 공간 음향으로 구성된다는 것을 알 수 있다:

$$
A = A_{point} + A_{cluster} + A_{global}
$$

각 source 들의 Ambisonics를 구하는 방법은 아래를 열어서 확인해보길 바란다.

<details markdown="block">
<summary>Point source</summary>

여기서 point source는 source $$i$$의 3D point cluster $$P_i$$를 하나의 centroid $$o_i$$로 표현한다.

그러면 상대 위치 벡터는

$$
d_i = t - o_i
$$

앞에서 우리는 Ambisonics로 만드는 수식을 다음과 같이 정의했다.

$$
a_L(t) = \sigma(d)a_{\mathrm{src}}(t)y_L(u)
$$

이 수식의 구조를 그대로 사용하여 point source를 Ambisonics로 만들면,

$$
A_{\mathrm{point}}
=
\sum_{i \in \mathcal{O}_{\mathrm{point}}}
a_{i,L}
\sigma(\lVert d_i \rVert)
y_L\left(
R^T \frac{d_i}{\lVert d_i \rVert}
\right)
$$

이렇게 된다.

그런데 여기서 $$u \rightarrow R^T \frac{d_i}{\lVert d_i \rVert}$$로 되어 있는데,
왜 $$R^T$$가 들어가는지 살펴보자.

그냥 $$\frac{d_i}{\lVert d_i \rVert}$$는 3D world coordinate에서의 source 방향이다.
하지만 우리는 listener가 현재 바라보고 있는 방향을 기준으로 source가 어디에 있는지를 원하기 때문에,
listener rotation $$R$$을 이용해 방향을 listener 기준으로 바꿔주어야 한다.

$$
y_L\left(
R^T \frac{d_i}{\lVert d_i \rVert}
\right)
$$

따라서 이 식은 현재 listener 기준으로 source $$i$$가 위치한 방향의 Spherical Harmonics 값이다.

자, 그러면 간략하게 정리해보자.

이 복잡한 표기를 직관적으로 적으면,

$$
\boxed{
\text{Point Ambisonics}
=
\sum_i
\text{source waveform}
\times
\text{distance attenuation}
\times
\text{source direction SH}
}
$$

라고 볼 수 있다.

즉,

$$
a_L(t) = \sigma(d)a_{\mathrm{src}}(t)y_L(u)
$$

해당 수식을 source마다 계산한 다음 모두 더한 것이 Point Source Ambisonics다.

</details>

<details markdown="block">
<summary>Clustered sources</summary>

강이나 폭포처럼 넓게 펼쳐진 source는 centroid 한 점으로 표현하기 어려워서 $$P_i$$ 안에 있는 모든 3D point를 사용한다:

$$
A_{\mathrm{cluster}}
=
\sum_{i \in \mathcal{O}_{\mathrm{cluster}}}
\frac{a_i}{\lvert P_i \rvert}
\sum_{o \in P_i}
\sigma\left(\lVert d \rVert\right)
y_L\left(
R^T \frac{d}{\lVert d \rVert}
\right)
$$

여기서 point souce와 다른 점은 그저 source 하나에 여러 3D point들이 있다는 것이다. 즉, 아래의 clustered source 내부에 있는 많은 각 point source에 대해 아래를 계산하고: 

$$
\sigma(\lVert d \rVert)y_L(\text{direction})
$$

전부 더한 다음, $$\frac{1}{\left \vert P_i \right \vert}$$로 나누어 평균을 낸다.

쉽게 말하면, $$\text{Clustered source} = \text{Average of point sources}$$이 된다.

그런데, 논문의 저자는 하나의 insight를 더 제공한다.

위의 수식처럼 clustered source를 여러 point source처럼 처리하게 되면, 각 point마다 방향이 다르기 때문에 다른 $$y_L$$ 값이 나오게 되는데, 그렇게 되면 Spherical Harmonics의 $$l>0$$ 성분은 방향에 따라 양수/음수 값을 가질 수 있기 때문에, 여러 방향의 값을 합치면 일부가 서로 없어질 수 있다. 

따라서, area sound가 listener를 둘러싸는 경우 $$l>0$$ 방향 성분들이 서로 상쇄되는 경향이 있어, head rotation에 덜 민감해질 수 있다.

</details>

<details markdown="block">
<summary>Global ambience</summary>

이제 마지막 바람, 아주 먼 traffic 등은 특정 3D object에 붙어 있는 소리로 처리하지 않는다. 

즉, global ambience에는 방향정보를 넣지 않겠다는 의미다.

$$
A_{global} = a_{global} 
\begin{bmatrix}
1 \\
0 \\
0 \\
\vdots \\
0
\end{bmatrix}
$$

라는 수식으로 표현된다.

첫 channel만 사용하는 이유는 첫 번째가 $$l=0$$인 omnidirectional component 이기 때문이다.

</details>

#### Free-Viewpoint Rendering

앞의과정들이 모두 끝났으면, 이제 우리는 만들어진 visual scene과 ambisonics를 기반으로 **visual과 binaural audio**를 렌더링해주어야 사용자가 보고 들을 수 있다.

binural audio라는 것은 left ear와 right ear 형태로 헤드폰용 오디오를 말한다.

앞의 과정들이 모두 끝나면 우리는 listenter pose $$p$$ 에서 

$$
a_L(t) = A(p,t)
$$

라는 Ambisonics가 만들어졌다.

FOA라면 L=1이므로 4개의 waveform이 만들어졌다고 생각하면 된다:

$$
a_1(t)= \begin{bmatrix}
a_{0,0}(t) \\
a_{1,-1}(t) \\
a_{1,0}(t) \\
a_{1,1}(t)
\end{bmatrix}
$$

이외에 visual scene 그냥 3DGS renderer로 pose $$p$$ 에 대해 $$V(p)$$로 그 위치에서의 카메라 이미지를 제공하면 된다.

Ambisonics를 어떻게 left/right ear audio로 제공할까? 그건 HRTF라는 기법이 답이 될 수 있다.

> HRTF는 특정 방향에서 온 소리가 사람의 머리와 귀를 거쳐 왼쪽 귀와 오른쪽 귀에 각각 어떻게 도달하는지를 나타내는 필터다.

그럼 이 HRTF 기반 binaural rendering을 통해서 논문에서는 time-domain filter를 

$$
h_{l,m}^{left}, h_{l,m}^{right}
$$

라고 표현한다. 정확히는 이를 HRIR이라고 부른다. 수식을 보면 알겠지만, Ambisonics의 각 channel에 대해 해당 filter를 convolution 해준다. 그럼 해당 Ambisonics channel이 left/right ear에 어떻게 실제로 들릴지 계산된다.

수식으로 예시를 보면,

$$
h_{l,m}^{left} * a_{l,m} ,\; h_{l,m}^{right} * a_{l,m}
$$

가 계산된다고 보면 된다.

그러면 최종식은 아래와 같다:

$$
\begin{bmatrix}
b_{\mathrm{left}} \\
b_{\mathrm{right}}
\end{bmatrix}
(t)
=
\sum_{\ell=0}^{L}
\sum_{m=-\ell}^{\ell}
\begin{bmatrix}
h_{\ell,m}^{\mathrm{left}} * a_{\ell,m} \\
h_{\ell,m}^{\mathrm{right}} * a_{\ell,m}
\end{bmatrix}
(t)
$$

즉, 왼쪽 귀에 대한 모든 ambisonics channel들과 filter의 convolution 값들을 더하고, 오른쪽 귀도 똑같이 해준 다음 사용자에게 제공하는 것이다.

## Experiments

## Contribution

## Limitations & Future work
