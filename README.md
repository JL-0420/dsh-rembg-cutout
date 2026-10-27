# dsh-rembg-cutout · 本地抠图插件

把图片主体抠成**透明背景 PNG**，顺手裁到主体外框。纯本地跑（onnxruntime + u2net/u2netp 权重），**不花钱、不上传、断网可用**。

给 DeepSeek Harness 提供两个模型工具：

| 工具 | 作用 |
|---|---|
| `remove_background` | 抠主体 → 透明 PNG；`crop: true` 时只留主体那一块（含留白参数） |
| `image_trim` | 只裁边不抠图：自动去掉四周纯色留白（白边/黑边/任意纯色底），或按透明通道裁到内容边界 |

## 安装

**方式一（推荐）：插件安装器填本地目录**
设置 → 插件 →「添加插件」→ 包名或地址填：

```
C:\Users\17833\Desktop\dsh插件研发\自我配置\dsh-rembg-cutout
```

安装源随便（本地目录不走 registry）。装完**完全重启** DSH 桌面端。

**方式二：命令行**
```powershell
dsh plugin --profile desktop add "C:\Users\17833\Desktop\dsh插件研发\自我配置\dsh-rembg-cutout"
```

## 首次使用会自动初始化

第一次调用 `remove_background` 时，插件会在 `<DSH_HOME>/rembg-cutout/` 下：

1. 用 **DSH 自带的 Python**（已带 Pillow + numpy）安装 `onnxruntime` 到 `pylibs/`（约 15MB，走清华镜像，约 10~30 秒）；
2. 下载模型 `u2netp.onnx`（约 4.7MB，来自 rembg 官方发布）；
3. 之后每次抠图就是纯本机推理，1~3 秒一张。

不想自动装、想自己装也行（报错信息里也会给出这条命令）：
```powershell
& "C:\Users\17833\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe" -m pip install --target "$env:USERPROFILE\.dsh\rembg-cutout\pylibs" -i https://pypi.tuna.tsinghua.edu.cn/simple onnxruntime
```

## 用法示例

> 这张 `C:\Users\17833\Desktop\pics\doubao-01.png` 把主体抠出来，裁掉白边，存成透明 PNG

模型会调用：
```json
{ "input": "C:\\Users\\17833\\Desktop\\pics\\doubao-01.png", "crop": true, "padding": 8 }
```

参数速查（`remove_background`）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `input` | 必填 | 绝对路径，或相对会话工作目录的路径 |
| `output` | 同目录 `*_cutout.png` | 输出路径；透明背景必须 .png |
| `model` | `u2netp` | `u2netp` 快（4.7MB）；`u2net` 细节好（176MB，首次会再下一次） |
| `crop` | `true` | 裁到主体外框 |
| `padding` | `0` | 裁剪时四周留白像素 |
| `feather` | `0` | 边缘羽化半径；边缘太硬给 0.5~2 |
| `threshold` | `128` | alpha 阈值；抠多了调低，抠漏了调高 |
| `background` | `transparent` | 也可以填 `#ffffff` 之类的纯色底 |

`image_trim`：`input`、`output`、`tolerance`（默认 8，背景有噪点就调大）、`mode`（auto/alpha/color）、`padding`。

## 配置（可选）

在 profile 的 `cordis.patch.yml` 里给 `rembg-cutout` 这一行加 `config`：

| 字段 | 默认 | 说明 |
|---|---|---|
| `pythonPath` | 自动探测 | 指定解释器（需自带 Pillow + numpy） |
| `dataDir` | `<DSH_HOME>/rembg-cutout` | 放 pylibs 和模型的位置 |
| `pipIndexUrl` | `https://pypi.tuna.tsinghua.edu.cn/simple` | 首次装 onnxruntime 用的源 |
| `defaultModel` | `u2netp` | 默认模型 |
| `timeoutMs` | `180000` | 单次执行超时 |

## 数据与清理

全部文件都在 `%USERPROFILE%\.dsh\rembg-cutout\`：

- `pylibs/`：onnxruntime（约 20MB 解压后）
- `models/u2netp.onnx`、`models/u2net.onnx`（如果用过）

不想要了：卸载插件后直接删掉这个目录即可。工作区、系统 Python、其他 profile 都不受影响。

## 已知限制

- u2netp 对**实拍/插画**主体效果好；主体与背景颜色极其接近、或纯色底图（白底豆包图）建议直接用 `image_trim` 裁边，抠图反而多此一举。
- 没做 alpha matting（发丝级细节），需要更细边缘时用 `--model u2net` + `feather 0.5`。
- 只处理静态位图（png/jpg/webp/bmp），GIF 动画只取第一帧；视频不做。
- 超大图（> 4000px）会先缩到 320 推理再放大 mask，速度仍快，但边缘会略糊。
