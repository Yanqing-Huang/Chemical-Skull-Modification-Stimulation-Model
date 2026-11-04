# Chemical Skull Modification Stimulation Model

SD 大鼠颅骨化学改性与 SSVEP 刺激的理论先验仿真模型（待体外实验校准）。

## 快速开始

### 在线交互网页（推荐）
直接打开 GitHub Pages 上的参数化仿真页面：

https://yanqing-huang.github.io/Chemical-Skull-Modification-Stimulation-Model/

所有参数（频率、试剂类型、浓度、浸泡时间、几何、孔隙率、动力学、电学）实时可调节，点击"运行仿真"即可看到脱钙与阻抗响应。

### 本地 Python 仿真
```bash
pip install -r requirements.txt
python run_simulation.py          # 生成理论先验结果（CSV + 图）
python -m unittest discover -s tests -v   # 7 项单元测试
python calibrate.py --data 您的体外数据.csv  # 后续校准
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
- 外边界液膜通量（Robin BC），内边界零通量
- Archie 型电阻率：$\sigma = \sigma_m + \sigma_f \varepsilon^m$

### 诊断量
- Thiele 模数 $\phi = h \sqrt{k_r q_0 / D_e}$
- 有效因子 $\eta = \tanh(\phi) / \phi$
- Biot 数 $\text{Bi} = k_f h / D_e$

## 项目结构

```
├── parameters/theoretical_prior.json    # SD 大鼠推测先验参数（含不确定度范围）
├── cranial_model/
│   ├── core.py                           # 三层有限体积求解器
│   ├── diagnostics.py                    # SCM 解析验证 + Thiele/Biot
│   └── plotting.py                       # SCI 风格作图
├── run_simulation.py                     # Python 仿真入口
├── calibrate.py                          # 后续体外数据校准入口
├── examples/experimental_data_template.csv
├── tests/                                # 7 项单元测试
├── results/                              # 理论先验结果（CSV + 图）
└── website/parameter-simulation.html     # 在线交互仿真（全内联）
```

## 当前理论预测（EDTA 浓度扫描，30 min）

| EDTA 浓度 | 全颅骨钙剩余 | 阻抗比（处理后/原始） |
|-----------|-------------|----------------------|
| 0 mol/L   | 96.54%      | 0.793                |
| 0.05 mol/L| 95.45%      | 0.747                |
| 0.10 mol/L| 94.37%      | 0.707                |
| 0.20 mol/L| 92.33%      | 0.645                |

**以上均为待校准理论预测，不是实验结果。**

## 后续校准

等体外实验数据（钙释放曲线、脱钙深度、micro-CT 孔隙率、阻抗频谱）到齐后，运行：

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
- 颅骨改性：内侧零通量；
- 若研究脑脊液/脑组织暴露效应，必须新增 CSF 和脑组织域，不能继续用零通量。

## 验证

- 质量守恒误差 < 10⁻¹⁰
- 网格收敛（40+24+20 → 1.28x 全网格）：0.096%
- 时间步收敛：0.004%
- 7 项单元测试全过（含无反应极限、SCM 解析验证）
- 浏览器仿真引擎与 Python 参考偏差 < 1%

## 参考文献

1. 林婧璇. 现生人群颅骨骨壁厚度多维度研究与古人类演化启示.
2. 刘澳. 考虑颅骨力学性能增龄改变的人体头部建模与应用.
3. 吴小明. 人颅骨电阻抗频率特性及其与解剖结构关系研究.
4. 生物反应工程与设备（第三章固定化生物催化剂）.
