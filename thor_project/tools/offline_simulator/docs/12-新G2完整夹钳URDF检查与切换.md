# 新 G2 完整夹钳 URDF 检查与切换

## 结论

Nexus Simulator 的 DH 变体直接使用平台已经整理并冻结的“旧机身/双臂 + DH 末端”
`G2_dahuan-v1` 权威模型，不再在工具目录维护同名 URDF 副本。`omnipicker-legacy` 继续
作为默认回滚版本。

本次切换只影响三维视觉与 FK。生产安全参考仍使用旧版固定哈希碰撞几何，任何真机能力仍被禁用。

## 资产检查结果

| 项目 | 结果 |
| --- | --- |
| URDF | XML 结构有效，机器人名为 `G2_dahuan` |
| 结构 | 40 个 link、39 个 joint |
| 自由度 | 32 个可控制关节；outer 夹爪关节由 mimic 驱动，不单独计数 |
| 三维资产 | 37 个 visual mesh，保留旧机身 DAE 材质并替换 DH 末端 |
| 路径完整性 | 47 次 mesh 引用、30 个唯一文件，未发现缺失文件 |
| 材质/纹理 | URDF 有白/灰 RGBA 材质；实际交付无 `textures/` 且无 `<texture>` 引用 |
| 夹钳 | 左右各两个 prismatic joint，每侧 outer 通过 mimic 跟随 inner，范围均为 `0~0.025 m` |
| 受控 URDF SHA-256 | `be60ec05409e87e2426e0f0156cfa3a8dc11995a4a13e319a7ac112dc0246316` |
| 原始导入 URDF SHA-256 | `0277a3d983332c8645973a9db8cb9bf4c341b18b47d120ac3b326fdd18242037` |
| 资产文件数 | 109（包含保留的旧版对照资产） |

受控副本位置：

```text
../../src/nexus_deploy/viewer/assets/g2/G2_dahuan/
├── README.md
├── manifest.json
├── G2_dahuan.urdf
└── meshes/
```

该目录及其 `manifest.json` 是唯一可编辑权威源；工具启动时同时校验冻结版本和 URDF
SHA-256。供应商原始整机文件以 `G2_dahuan_vendor.urdf` 留作追溯，不参与工具运行。
权威模型使用已验证的 `base_link` 根节点，因此工作面 `height=0.75 m` 表示相对
`base_link` 的 0.75 m。

## Episode 夹钳映射

现有数据集使用旧夹爪角度语义，内指关节范围约为 `-0.785398~0 rad`；新 URDF 使用
`0~0.025 m` 的直线位移。显示层采用线性映射：

```text
-0.785398 rad（闭合） -> 0.025 m
 0.000000 rad（张开） -> 0.000 m
```

新 URDF 的视觉零位对应张开、正向位移对应闭合，所以这里使用反向映射。同侧 inner/outer
两指由同一旧值对称驱动。映射仅作用于新模型的显示字典，原始 Episode
向量以及安全层输入保持不变；超出旧协议范围的值会被拒绝，而不是截断后静默显示。

## 切换与回滚

旧模型仍是默认启动项：

```bash
./scripts/nexus-sim serve --dataset-root /path/to/lerobot-dataset \
  --host 0.0.0.0 --port 8766 --meshcat-port 8767
```

新模型建议用独立端口预览：

```bash
./scripts/nexus-sim serve --dataset-root /path/to/lerobot-dataset \
  --host 0.0.0.0 --port 8770 --meshcat-port 8771 \
  --robot-model G2_dahuan
```

模型网格在服务启动时建立，现阶段不支持在同一 MeshCat 场景内热切换。回滚只需停止新预览，
继续访问 8766；不需要修改或恢复旧资产。

## 必须保留的安全边界

1. 新 URDF 的视觉/碰撞几何尚未和 Nexus 生产安全点位重新标定。
2. 页面会显示“视觉预览 · 安全几何仍使用旧版”，接口也返回
   `visual_collision_geometry_aligned=false`。
3. 导入目录名称写有“摄像头”，但 URDF 中没有独立 camera link、camera joint、sensor、
   optical frame、RGB/depth 参数或明确外参。它可能在头部或夹钳网格外观中包含摄像头结构，
   但不能仅凭网格建立实时图像坐标系。
4. 在获得相机外参前，现有 Episode 头部视频只能继续作为独立回放画面，不能宣称已和新 URDF
   的相机光轴完成几何对齐。
5. 当前浅色机器人外观来自 URDF 内以白/灰为主的材质定义；外层 README 虽列出
   `textures/`，实际文件包中没有纹理目录。这不是浏览器漏载纹理，若需要品牌色或精细表面，需由
   资产提供方补充材质/纹理，或另建明确的可视化材质覆盖配置。

## 后续接入顺序

1. 人工确认新模型的零位、左右臂方向、夹钳开合方向和安装位置。
2. 向 URDF 增加明确的相机坐标系及外参，或从机器人标定文件读取并作为独立配置加载。
3. 为新夹钳建立安全采样点/保守包络，完成桌面高度边界和轨迹拦截回归。
4. 新旧模型同轨迹对比通过后，才考虑把平台默认模型切换到新变体。

## 新 DH 数据集回放

已验证数据集：

```text
/path/to/logistics-g2a-dh-dataset
```

该数据集是 LeRobot v3.0 / G2A / 24D / 30 FPS，共 428 个 Episode。夹钳使用原生
`0~0.025 m` 编码，但两类向量的极性不同：

- observation：`0=闭合`，`0.025=张开`；
- action：`0=张开命令`，`0.025=闭合命令`。

映射到新 URDF 时再按其实际运动方向处理：URDF `0=张开`，正向位移为闭合。因而
observation 使用 `0.025-value`，action 可以直接使用其命令值。

该方向不只依据字段说明判断：在零位与 `0.025 m` 位分别计算了左右指网格中心距离，正向
位移后两指视觉中心距离减小，因此确认正向位移在这份网格装配中代表闭合。

加载器从 `conversion_manifest.json` 的逐 Episode `effector_encoding=native` 自动识别该协议。
显示层把 observation/action 分别转换为相同的实际夹钳位移；安全层则始终接收其要求的 DH
命令极性。Episode 0 的 5,123 帧 observation 与 action 已完成关节限位扫描，未发现越界。

当前预览命令：

```bash
./scripts/nexus-sim serve \
  --host 0.0.0.0 --port 8772 --meshcat-port 8773 \
  --robot-model G2_dahuan \
  --dataset-root /path/to/logistics-g2a-dh-dataset \
  --episode 0
```
