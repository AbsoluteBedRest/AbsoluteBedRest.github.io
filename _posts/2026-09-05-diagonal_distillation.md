---
title: "[Paper Review, KR] Streaming Autoregressive Video Generation via Diagonal Distillation"
date: 2026-09-17
categories:
  - Preliminaries
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

