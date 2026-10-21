# G2 Omnipicker URDF Viewer

本目录包含两个不依赖 ROS package path 的可视版本：

- `G2_omnipicker_visual.urdf`：官方完整外观网格；
- `G2_omnipicker_safety_geometry.urdf`：机身保持完整外观，左右夹爪显示为
  Safety Layer 使用的官方凸碰撞几何（橙色）。

## VS Code 打开方法

1. 用 VS Code 打开整个 `g2a_deploy_project` 文件夹，不要只打开单个 URDF；
2. 安装推荐扩展 `morningfrog.urdf-visualizer`；
3. 打开上述任一 `.urdf`；
4. 点击编辑器右上角预览按钮，或执行命令
   `URDF Visualizer: Preview URDF/Xacro`。

模型只使用相对 mesh 路径，不需要 ROS、Xacro 或额外 package 映射。插件内可以拖动
关节；夹爪的被动指节已根据官方 mimic 系数跟随主动关节。

## 加入现场桌面

例如桌面顶面在 `base_link` 中为 `z=0.742 m`：

```bash
python tools/build_urdf_viewer.py \
  --table-height 0.742 \
  --table-size 1.6 1.2 \
  --table-center 0.0 0.6
```

命令会重建两个 URDF，并加入半透明蓝色桌面。`--table-height`表示桌面顶面高度，单位
为米；它应与 Safety 配置中的 `surfaces[].height`使用相同的 `base_link`坐标系。
生成脚本更新机器人客户端源码资产；随后执行下面命令同步到 `dist`：

```bash
.venv/bin/python -m nexus_deploy.deployment --config deploy.toml client bundle --robot g2a
```

若要移除桌面，直接运行：

```bash
python tools/build_urdf_viewer.py
```

资产来源和固定版本见 `SOURCE.json`，许可证见 `LICENSE`。
