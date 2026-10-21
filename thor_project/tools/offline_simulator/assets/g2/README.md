# G2 模型资源

运行时仍可选择两个模型版本，但 DH 模型不再在工具目录保存副本：

```text
assets/g2/
├── G2_omnipicker/
│   ├── G2_omnipicker_visual.urdf
│   ├── G2_omnipicker_fixed_dual.urdf
│   ├── G2_omnipicker_safety_geometry.urdf
│   ├── meshes/
│   ├── collision/
│   ├── source_manifest.json
│   └── manifest.json

../../../../src/nexus_deploy/viewer/assets/g2/G2_dahuan/
└── G2_dahuan.urdf       # Nexus 冻结权威版本
```

| 运行标识 | 用途 | URDF SHA-256 |
| --- | --- | --- |
| `omnipicker-legacy` | 已验证旧版、默认回滚版本 | `e66ad093999c4d9675902def559e5a3a7a42c4ed0cfe5ab2f3efd3b087582814` |
| `G2_dahuan` | 原厂 CRS-B 机身/双臂 + 25 mm DH 末端的 Nexus 权威版本 | `be60ec05409e87e2426e0f0156cfa3a8dc11995a4a13e319a7ac112dc0246316` |

`nexus-sim sync-assets` 只更新 `G2_omnipicker/`：它从该目录的
`source_manifest.json` 指定的机器人客户端构建目录复制资产，并生成 `manifest.json`。该操作不会
修改平台的 `G2_dahuan` 权威资产。二进制网格和本机 URDF 不提交 Git，清单与说明保留。
