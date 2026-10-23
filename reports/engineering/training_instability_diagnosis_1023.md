# 训练不稳定诊断(主表 4 个方法,2026-10-23 12:5x CDT)

## 结论
**主要是超参数(lr 偏高 + 无 warmup)的问题,不是数据、不是某个方法本身,也没有发现训练器 bug。**
论文 recipe(Kang/AD:LoRA r64/α128,lr 2e-4,linear 衰减,warmup 0,AdamW β=(0.9, 0.999),eps 1e-8,clip 1.0)
被原样移植过来;Gemma-4-12B 在这个 lr 下处在稳定性边界附近,偶尔会触发 Adam 的失控更新。

## 证据
1. **与数据无关。** smc K300 的 e2/e4/e8/e16/e32 五档用的是同一套数据顺序(每一步的 batch 完全相同),
   但同一个 batch 上的梯度范数相差上千倍:update 121(batch [59, 220, 138, 258])在 e4 上是 3,710,在其他档是 0.5–6。
   其他档在同一位置都没有 spike。所以异常梯度来自当时的模型状态,不来自某条轨迹。
2. **与方法无关。** 4 个方法都出现过 spike:轨迹SFT、AD、SmartAD 三个 LoRA 方法共用一个训练器;SAD 是全参数训练,
   用另一个 objective,但 AdamW 和 clip 相同。
3. **失控过程**:3–5 步之内梯度范数从约 2 涨到上千,loss 只是小幅上升,之后 loss 才炸开
   (smc K300 e4,update 118–125:gnorm 2.3 → 27 → 167 → 3,710,loss 0.12 → 0.15 → 0.21 → 0.62 → 1.2 → 4.2;
   AD K500 e2,update 33–38:gnorm 5 → 296 → 1,018 → 2,738,loss 0.26 → 9.4)。
4. **clip 1.0 在每一步都生效**(LoRA 的梯度范数中位数是 2–30,SAD 的 loss 按 token 求和,中位数上万),
   所以 clip 区分不了失控的那一步。Adam 的步长约为 m/√v,对梯度整体缩放不敏感,clip 也限制不了步长。
5. **强烈依赖 lr**(spike 判定:单步 loss > max(5 × 前 20 步中位数, 中位数 + 0.5)):

| 设置 | 训练 run 数 | update 数 | spike 数(每千步) | 最终没恢复 |
|---|---|---|---|---|
| LoRA lr 2e-4(论文) | 10 | 1,914 | 4(2.1) | 2(AD K500 e2、轨迹SFT K100 e2) |
| LoRA lr 1e-4 | 33 | 28,256 | 26(0.9) | 2(轨迹SFT K500 e32、SmartAD K500 e32),另有 AD K500 e4(LONI)在第 1 个 epoch 发散 |
| LoRA lr 5e-5 | 10 | 25,598 | 3(0.1) | 0 |
| SAD 全参数,lr 2e-5 cosine | 10 | 7,767 | 6(0.8) | 1(SAD K100 ×1) |

   同样的 lr 下发生与否带有随机性:AD K500 lr 1e-4 时 e2 正常,e4 第 1 个 epoch 就发散了。
   2 卡和 4 卡的数值误差让各档在头几步就分叉,之后是否触发 spike 取决于这些细小差异。
6. **训练 loss 恢复不等于模型恢复**:AD K300 e32(lr 1e-4)在 epoch 4 发生 spike,持续 360 步,
   最终训练 loss 回到 0.004,注册的 loss 检查判它有效,但 dev140 只有 4/140(39% 无效动作)。
   它恢复的是对训练集的记忆,通用能力没有回来。

## 检查过、排除掉的训练器问题
- **梯度裁剪**:各 trainer 都有(clip 1.0,error_if_nonfinite)。
- **LoRA 配方**:AdamW、weight decay 0、fp32 LoRA master 权重、bf16 冻结 base,都和官方一致(baselines/kang/DEVIATIONS.md)。
- **loss 归一化**:LoRA 方法按每条轨迹的 token 均值,再按实际 batch 内的轨迹数平均。SAD 按 token 求和,这是按论文注册的实现,
  会让梯度范数变大,但 Adam 对整体缩放不敏感,不是诱因。
- **数据**:同一 batch 在别的档上正常(见证据 1)。
