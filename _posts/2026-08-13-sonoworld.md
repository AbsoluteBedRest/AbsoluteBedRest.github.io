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

SonoWorld의 Key Tehcnique은 총 3개로 나뉜다.

- 첫 번째로, single image를 입력으로 받아 360도 파노라마를 생성하고 이를 3D Scene에 Lift하는 것
- 두 번째로, reconstructed된 3D 공간에서 소리가 있을 법한 실체를 식별하고 위치를 결정하는 것
- 세 번째로, spatial sound field를 생성하는 ambisonics encoder를 설계하는 것
- 네 번째로, 청취자 입장에서 어느 위치(pose)에서도 입체적인 binaural audio를 렌더링하는 것

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

#### 360 $$\degree$$ Aduio-Visual Semantic Grounding

이제, 3D 공간에 소리를 생성하고 위치시키는 방법이다.



## Experiments

## Contribution

## Limitations & Future work
