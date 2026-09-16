---
title: "[Paper Review, KR] MotionStream: Real-Time Video Generation with Interactive Motion Control"
date: 2026-09-07
categories:
  - Other Fields
tags:
  - distillation
  - real-time generation
  - autoregressive
---

> **Paper Information** \\
> **Title:** MotionStream: Real-Time Video Generation with Interactive Motion Control \\
> **Authors:** Joonghyuk Shin, Zhengqi Li, Richard Zhang, Jun-Yan Zhu, Jaesik Park, Eli Shechtman, Xun Huang \\
> **Venue:** ICLR 2026 Oral \\
> **Link:** [[Paper](https://joonghyuk.com/motionstream-web/paper.pdf)], [[Project](https://joonghyuk.com/motionstream-web/)], [[GitHub](https://github.com/alex4727/motionstream)]

## Teaser Image (Poster)

<p align="center">
  <img src="/assets/images/posts/2026-09-07-motionstream/1789522129230.png" width="100%">
</p>

## Introduction

Motion-controlled video generation은 사용자가 객체, 인물, 카메라의 움직임을 직접 지정할 수 있도록 발전해 왔다. 초기에는 구조(structure), 카메라, subject, audio 등 다양한 조건을 활용하는 controllable video generation 연구가 진행되었고, 이후 motion 자체가 영상의 동역학을 직접 표현하는 중요한 conditioning signal로 자리 잡으면서 optical flow, 2D/3D trajectory, bounding box, semantic segmentation 등 다양한 motion representation이 활용되기 시작했다. 최근 video diffusion model들은 이러한 조건을 이용해 높은 화질과 정교한 trajectory following을 달성했지만, 대부분 전체 영상과 전체 motion **condition을 한 번에 처리하는 bidirectional diffusion 구조에 기반**한다. 따라서 **사용자가 미래의 trajectory를 모두 미리 지정해야 하며, 생성 중간에 결과를 확인하거나 motion을 수정하기 어렵다**. 또한 **diffusion의 반복적인 denoising 과정 때문에 생성 속도가 매우 느린데**, 논문에서는 Motion Prompting의 경우 5초 영상을 생성하는 데 약 12분이 소요되는 사례를 들고 있다. 결국 **기존 motion-controlled video generation은 높은 품질과 제어 능력에도 불구하고 느린 생성 속도, non-causal processing, 짧은 생성 길이 때문에 실시간 상호작용에는 적합하지 않았다**.

한편 video generation의 시간적 생성 방식은 초기 GAN 기반 autoregressive/parallel video synthesis에서 시작하여, 이후 denoising objective를 사용하는 video diffusion model과 next-token prediction 기반 autoregressive video model로 발전해 왔으며, 최근에는 두 방식을 결합해 causal하면서도 고품질인 AR-diffusion 모델을 만드는 연구가 활발하다. 특히 느린 bidirectional diffusion teacher를 빠른 autoregressive student로 distillation하는 방식이 등장하면서 real-time generation 가능성이 커졌지만, 이러한 **모델들은 학습한 sequence 길이를 넘어 장시간 생성할 경우 color drift나 품질 저하가 누적되거나, 이를 방지하기 위해 복잡한 long-video fine-tuning이 필요하다는 한계**가 있다. 동시에 Genie 계열과 같은 interactive video world model 연구도 실시간 user interaction을 목표로 발전하고 있지만, 많은 방법이 **매우 큰 inference compute를 요구하거나 Minecraft·게임과 같은 closed-domain 또는 synthetic environment에 제한**되어 있다. 따라서 기존 연구들은 각각 controllability, causal generation, real-time interaction 중 일부는 달성했지만, **open-domain photorealistic video에서 motion control을 유지하면서 장시간 안정적으로 실시간 생성하는 문제는 여전히 해결되지 않은 과제**로 남아 있다.

## Method & Technical Details

#### Adding Motion Controls to Bi-Directional Teacher Models

