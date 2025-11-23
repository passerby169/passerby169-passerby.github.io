# Cookbook

付手续费

## 1. 迁移 Scanpy 项目

```ts
import { loadScanpy } from '@sc-cafm/adapters'

const cafm = await loadScanpy('adata.h5ad')
```

## 2. 内嵌实验日志

```ts
watch(() => store.result, (value) => {
  sendToNotebook({
    perturbation: store.lastRun,
    summary: value?.delta ?? 0
  })
})
```

## 3. 自定义可视化

将 CAFM 输出注入至任何 ECharts/Three.js 组件，利用 Pinia 保持状态一致。
