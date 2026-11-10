# Chemical Skull Modification Stimulation Model

SD 大鼠颅骨化学改性的三层理论模型，含体外阻抗趋势对照、平台绑定等效标定与留鼠内部验证。

## 快速开始

### 在线交互网页（推荐）
直接打开 GitHub Pages 上的参数化仿真页面：

https://yanqing-huang.github.io/Chemical-Skull-Modification-Stimulation-Model/

所有参数（频率、试剂类型、浓度、浸泡时间、几何、孔隙率、动力学、电学）实时可调节，点击"运行仿真"即可看到脱钙与阻抗响应。

### 本地 Python 仿真
```bash
pip install -r requirements.txt
python run_simulation.py          # 生成理论先验结果（CSV + 图）
python run_prior_mechanism.py     # Track A：200 条文献先验 LHS 轨迹
python run_equivalent_calibration.py --data /本地路径/体外EDTA数据记录汇总.xlsx
python run_trend_comparison.py    # Track A × 实测低频阻抗趋势对照
python -m unittest discover -s tests -v   # 31 项单元测试
```

## 模型结构

### 三层颅骨（一维厚度方向）
- 外板（密质骨）~0.43 mm
- 板障（松质骨）~0.20 mm
- 内板（密质骨）~0.17 mm

### 三步时序化学改性
1. **50% 乙酸预处理**（300 s）——酸溶解初步脱钙
2. **EDTA 螯合**（30 min，0.1 mol/L）——主脱钙阶段
3. **饱和尿素**（10 min）——溶胀，不消耗固相钙

### 控制方程（多孔介质）
$$\frac{\partial(\varepsilon c_j)}{\partial t} = \frac{\partial}{\partial x}\left(D_{e,j}\frac{\partial c_j}{\partial x}\right) - \nu_j k_{r,j} c_j q_j$$

- $D_e = D_{\text{bulk}} \varepsilon / \tau$（有效扩散系数，已含孔隙率）
- 固相钙守恒：$\partial q / \partial t = -k_r c q$
- 孔隙率演化：$\varepsilon = \varepsilon_0 + \alpha (q_0 - q)$
- 50% 乙酸仅接触外板（外侧 Robin BC、内侧零通量）；EDTA 和尿素完全浸没，内外表面均为 Robin BC
- Archie 型电阻率：$\sigma = \sigma_m + \sigma_f \varepsilon^m$

### 诊断量
- Thiele 模数 $\phi = h \sqrt{k_r q_0 / D_e}$
- 有效因子 $\eta = \tanh(\phi) / \phi$
- Biot 数 $\text{Bi} = k_f h / D_e$

## 项目结构

```
├── parameters/theoretical_prior.json    # SD 大鼠推测先验参数（含不确定度范围）
├── cranial_model/
│   ├── core.py                           # 三层有限体积求解器（浸没/单侧边界、SCM、分层 k_r、EDTA 位点容量）
│   ├── diagnostics.py                    # SCM 解析验证 + Thiele/Biot
│   ├── ensemble.py                       # LHS 采样与集成预测
│   ├── experimental.py                   # 实测阻抗谱读取
│   ├── equivalent_calibration.py         # B v2：双分支观测算子、等效标定、留鼠验证、残差诊断
│   └── plotting.py                       # SCI 风格作图
├── run_simulation.py                     # Python 仿真入口
├── run_prior_mechanism.py                # A 主线：文献先验 LHS 机理集成（不用实测数据调参）
├── run_equivalent_calibration.py         # B v2：基质+界面双分支标定、留鼠验证与界面份额分解
├── run_trend_comparison.py               # A×实测 Z_ratio 趋势对照（非校准）
├── calibrate.py                          # 后续体外数据校准入口
├── examples/experimental_data_template.csv
├── tests/                                # 31 项单元测试
├── results/                              # 理论结果、去标识化实测目标与验证输出
│   ├── prior_mechanism/                  # A 主线集成输出
│   ├── equivalent_calibration_v2/        # B v2：双分支等效标定输出
│   └── trend_comparison/                 # 趋势对照输出
└── index.html                           # 在线交互仿真（轻量近似，定量以 Python 为准）
```

## 当前理论预测（EDTA 浓度扫描，30 min）

| EDTA 浓度 | 全颅骨钙剩余 | 基质面积电阻比（模型） |
|-----------|-------------|----------------------|
| 0.05 mol/L| 94.85%      | 0.772                |
| 0.10 mol/L| 93.97%      | 0.741                |
| 0.15 mol/L| 93.27%      | 0.719                |
| 0.20 mol/L| 92.66%      | 0.704                |

**以上均为待校准理论预测，不是实验结果。**

## 真实数据对照与验证范围

### 数据与质控

- 原始工作簿：24 只大鼠、432 个频谱、1728 行频率测量，频率为 100/120/1000/10000 Hz；源文件 SHA-256 为 `486ec7e995944a814d23f6ecec995c20b716dbfe143519cc9e5fbcb39d18b056`。
- 当前验证目标是 100/120 Hz 的 `Z_post/Z_baseline`：先逐部位、逐频率相对自身基线，再在大鼠内平均 3 个部位和 2 个频率。
- 按实验备注剔除冻存、乙醇+尿素处理等混杂测量，排除 22 个大鼠-条件目标；最终为 **22 只大鼠、98 个目标**，每个目标均含 3 个部位、6 个低频比值。
- 0.15 和 0.20 mol/L 的 90 min 条件各只有 1 只大鼠，相关统计和留鼠结果需谨慎解释。留鼠验证属于**内部验证**，不是独立外部验证。
- 原始 Excel 不纳入仓库；公开内容仅为代码、去标识化大鼠级目标/预测结果和源数据哈希。

### Track A：文献先验机理与实测趋势

四个 EDTA 浓度的实测条件均值和仿真中位数均单调下降，各浓度的 Spearman ρ 均为 1.0；但实测总降幅为 0.485–0.617，纯基质模型仅为 0.034–0.089。因此只能认定**趋势方向一致**，不能认定幅值吻合或本征动力学参数得到验证。钙剩余量、脱钙深度以及 PLV/SNR 不在本次实测验证范围内。

### Track B：等效标定与留鼠内部验证（未通过）

预设兼容阈值为条件间大鼠 SD 的 RMS，即 0.10328。同一批 98 个目标上的结果为：

| 候选空间 | 候选数 | 兼容数 | 最佳 RMSE |
|---|---:|---:|---:|
| Track A 文献先验纯基质 | 200 | 0 | 0.25399 |
| 阶段边界修正后的等效纯基质 | 200 | 0 | 0.11826 |
| 基质+外表面界面双分支 | 3200 | 0 | 0.10987 |

界面分支较相同 200 条纯基质轨迹的最佳 RMSE 进一步降低 0.00839（7.1%），但仍高于兼容阈值 0.00659。**22/22 个留鼠折均为空兼容集**，所以覆盖率和区间宽度不可定义，而不是“覆盖率为 0%”。最佳 5% 候选的全数据探索性点覆盖为 62.2%，不是独立验证证据；残差随 EDTA 时间显著漂移（Spearman ρ=-0.635，p=7.09×10⁻¹⁰），触发结构失配标记。

因此当前模型**未通过实测幅值与留鼠内部验证，不能作为已验证预测工具**；兼容集为空，也不报告界面份额。完整数值见 [B v2 指标](results/equivalent_calibration_v2/equivalent_calibration_metrics.json)、[留鼠汇总](results/equivalent_calibration_v2/loo_rat_summary.csv) 和 [残差诊断](results/equivalent_calibration_v2/FigB2_residual_drift.png)。

## 其他观测的后续校准

若获得钙释放曲线、脱钙深度、micro-CT 孔隙率或独立阻抗数据集，可运行：

```bash
python calibrate.py --data 实验数据.csv --fit stages.edta.k_r_m3_per_mol_s
```

输出"理论先验 vs 校准后"逐点对照表。

## 方法学说明

### 参数来源边界
- 人体颅骨论文提供**结构方法和分层测量**，**不能**直接迁移数值给 SD 大鼠；
- 固定化反应工程 PPT 提供 **$D_e$ / Thiele / Biot / 有效因子** 的方法论；
- 所有 SD 大鼠的分层厚度、孔隙率、钙含量、反应速率必须标为"推测先验"，保留不确定度范围。

### 频率的使用
- 脱钙过程本身与频率无关；
- 频率仅作用于电学读出层；
- 4–30 Hz SSVEP 频段（吴小明论文定性）呈阻性，采用准静态电导率近似；
- 不能仅凭体积传导模型预测 PLV 或直接证明 SNR 提升。

### 边界条件
- 50% 乙酸仅接触外板：`stages.acetic_acid.boundary_mode=outer_only`，内侧零通量；
- EDTA 和尿素阶段为离心管完全浸没：`stages.edta.boundary_mode=immersed`、`stages.urea.boundary_mode=immersed`，外板和内板均接触同一药液；
- 体内外板涂抹情景：将相应阶段设为 `outer_only`，内侧零通量；
- 若研究脑脊液/脑组织暴露效应，必须新增 CSF 和脑组织域，不能继续用零通量代替组织域。

### Track B v2 解读边界
- 每条机理轨迹使用 16 组条件 LHS 界面参数（默认 200×16=3200 个复合候选），不增加正向仿真次数；纯基质基线仍使用原始 200 条轨迹。
- 复合动力学参数及界面参数 $\beta,m$ 仅适用于当前测量平台，不是羟基磷灰石本征动力学，不可迁移到其他电极平台。
- 界面份额是全数据兼容集上的描述性分解，不是后验概率或独立验证的因果份额；预测性能须另看留鼠覆盖率和区间宽度。
- Track A 不用实测阻抗调参，与实测低频阻抗仅作趋势级对照。

## 验证

- 液相质量守恒最大相对误差 < 3×10⁻¹⁰；钙守恒最大绝对残差 < 8×10⁻¹⁴ mol/m²
- 基准网格相对细网格：EDTA 末钙总量差 0.0046%，面积电阻差 0.0143%
- 时间步收敛（20 s 相对 5 s）：钙总量与面积电阻相对差均 < 10⁻⁸%
- 31 项单元测试全过（含无反应极限、SCM 解析验证、浸没边界守恒、分层 k_r、EDTA 位点容量、B v2 双分支合成数据测试）
- 网页用于轻量交互演示；定量结果以 Python 参考输出为准

## 参考文献

1. 林婧璇. 现生人群颅骨骨壁厚度多维度研究与古人类演化启示.
2. 刘澳. 考虑颅骨力学性能增龄改变的人体头部建模与应用.
3. 吴小明. 人颅骨电阻抗频率特性及其与解剖结构关系研究.
4. 生物反应工程与设备（第三章固定化生物催化剂）.
