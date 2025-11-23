# API

 精简的 TypeScript API，以便与 Vue 组件、Pinia actions 或后台推理服务集成。

## 类型

```ts
export interface RunOptions {
  perturbation: string
  dosage?: number
  cells?: string[]
}

export interface CafmStore {
  cafm: ScCafmInstance
  regulators: string[]
  run(options: RunOptions): Promise<Result>
}
```

## 使用 Actions

```ts
export const usePerturbationStore = defineStore('perturb', {
  state: () => ({ status: 'idle', result: null }),
  actions: {
    async simulate(payload: RunOptions) {
      this.status = 'running'
      this.result = await this.cafm.run(payload)
      this.status = 'done'
    }
  }
})
```

## Devtools

启用 `debug: true` 以将 CAFM 运行轨迹推送到 Pinia Devtools，便于追踪每个干预结果。
