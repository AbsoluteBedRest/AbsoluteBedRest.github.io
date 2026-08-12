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

이 논문은 이러한 한계점들을 파악하여 physic solver와 video diffusion를 함께 사용하여 **"Hybrid generative simulator"**를 만들어 해결하고자 한다.

### Contributions

논문에서 주장하는 주요 contribution을 정리한다.

- ...
- ...
- ...

## Method & Technical Details

### Overall Pipeline

전체 파이프라인을 먼저 설명한다.

```text
Input
  ↓
Module A
  ↓
Module B
  ↓
Optimization / Generation
  ↓
Output