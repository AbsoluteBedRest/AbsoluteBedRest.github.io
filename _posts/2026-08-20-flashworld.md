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

## Method & Technical Details



