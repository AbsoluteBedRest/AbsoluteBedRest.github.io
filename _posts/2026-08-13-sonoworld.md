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

## Method & Technical Details




## Experiments

## Contribution

## Limitations & Future work
