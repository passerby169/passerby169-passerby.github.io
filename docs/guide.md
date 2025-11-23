# Guide

 。通过引入调控网络先验，可以在 PBMC、肿瘤或再生医学数据集中探索潜在的驱动因子。  
本指南涵盖基础概念、安装和最小示例，帮助你在几分钟内构建第一个实验。

## 安装

```bash
npm install @sc-cafm/core pinia vue@3 
```

## 定义 store

```ts
import { defineStore } from 'pinia'
import { initScCafm } from '@sc-cafm/core'

export const useModelStore = defineStore('model', () => {
  const cafm = initScCafm({
    dataset: 'pbmc-10k',
    regulators: ['TF', 'Chromatin']
  })
  return { cafm }
})
```

## 下一步

- 配置 perturbation 场景，比较不同干预条件。
- 订阅 cafm 事件并推送到实验笔记或仪表盘。
