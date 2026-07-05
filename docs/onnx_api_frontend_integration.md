# Python 3.6 可用的 `onnxruntime==1.10.0` 官方只支持到 ONNX opset 15，所以给 Python 3.6 部署使用的模型应导出为 `--opset 15`。

## 2. 导出 ONNX

在有 PyTorch 的训练环境中执行：

```powershell
& 'C:\Nvidia_sim\miniconda3\envs\env_isaaclab\python.exe' scripts\export_pt_to_onnx.py `
  --checkpoint runs\crt_dataset_all_multiclass_neg_smoke\best.pt `
  --output models\crt_dataset_all_multiclass_neg_smoke_opset15.onnx `
  --device cpu `
  --opset 15 `
  --static-batch `
  --no-constant-folding
```

输出会同时生成同名 metadata：

```text
models/crt_dataset_all_multiclass_neg_smoke_opset15.metadata.json
```

metadata 中包含输入尺寸、归一化参数、类别名、阈值和最小面积过滤参数。前端和后端解析时应以该 metadata 为准。

## 3. Python 3.6 ONNX 环境

本机已创建并测试环境：

```powershell
conda create -y -n corrosion_py36_onnx python=3.6 pip
conda install -y -n corrosion_py36_onnx numpy=1.19 pillow=8.3 flask=1.1 werkzeug=1.0
conda install -y -n corrosion_py36_onnx flatbuffers protobuf packaging
```

ONNXRuntime 1.10.0 的 Windows/Python 3.6 wheel 已下载到：

```text
resources/wheels/onnxruntime-1.10.0-cp36-cp36m-win_amd64.whl
resources/wheels/flatbuffers-2.0.7-py2.py3-none-any.whl
```

离线安装：

```powershell
& 'C:\Nvidia_sim\miniconda3\envs\corrosion_py36_onnx\python.exe' -m pip install `
  --no-index --find-links resources\wheels `
  flatbuffers==2.0.7 onnxruntime==1.10.0
```

Windows 下建议用 `conda run` 启动，否则可能因为 DLL 搜索路径导致 `numpy` 导入失败。

## 5. 启动 HTTP API

```powershell
& 'C:\Nvidia_sim\miniconda3\Scripts\conda.exe' run -n corrosion_py36_onnx python scripts\onnx_inference_api.py `
  --model models\crt_dataset_all_multiclass_neg_smoke_opset15.onnx `
  --metadata models\crt_dataset_all_multiclass_neg_smoke_opset15.metadata.json `
  --serve `
  --host 0.0.0.0 `
  --port 9100
```

接口：

- `GET /api/health`
- `POST /api/onnx/predict`

上传字段：

- `image`：图片文件，multipart/form-data。
- `include_overlay`：可选，`1`/`0`，默认 `1`。为 `1` 时返回 base64 PNG overlay。

前端请求示例：

```js
async function predictCorrosion(file) {
  const form = new FormData();
  form.append("image", file);
  form.append("include_overlay", "1");

  const res = await fetch("http://127.0.0.1:9100/api/onnx/predict", {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}
```

## 6. 返回字段解析

核心字段：

- `corrosion_level`：图像级等级，`0-5`。
- `corrosion_level_name`：等级名称。
- `corrosion_area_percent`：阈值和连通域过滤后的腐蚀面积占比。
- `raw_corrosion_area_percent`：未做最终面积过滤前的 argmax 前景面积占比，用于调试。
- `severity_pixels`：各腐蚀等级像素数，键为 `"1"` 到 `"5"`。
- `severity_area_percent`：各腐蚀等级面积占比。
- `corrosion_blocks`：腐蚀连通区域列表，可用于前端框选/hover。
- `overlay_image`：可直接赋给 `<img src>` 的 `data:image/png;base64,...`。
- `detection_confidence`：如果有腐蚀区域，为前景平均置信度；如果无腐蚀，为背景平均置信度。

等级含义：

| 值 | 含义 |
| --- | --- |
| 0 | 背景 / 无腐蚀 |
| 1 | 轻微腐蚀 |
| 2 | 轻中度腐蚀 |
| 3 | 中度腐蚀 |
| 4 | 重度腐蚀 |
| 5 | 极重度腐蚀 |

前端展示建议：

```js
function renderResult(data) {
  levelText.textContent = `${data.corrosion_level} - ${data.corrosion_level_name}`;
  areaText.textContent = `${data.corrosion_area_percent.toFixed(2)}%`;
  confidenceText.textContent = data.detection_confidence.toFixed(4);

  if (data.overlay_image) {
    overlayImg.src = data.overlay_image;
  }

  blockList.innerHTML = "";
  for (const block of data.corrosion_blocks || []) {
    const item = document.createElement("li");
    item.textContent = `#${block.id} ${block.corrosion_level_name}, area=${block.area_pixels}px`;
    blockList.appendChild(item);
  }
}
```

如果 `corrosion_level === 0` 或 `corrosion_area_percent === 0`，前端应显示“未检测到腐蚀”，不要渲染腐蚀详情列表。

## 7. 返回 JSON 示例

```json
{
  "model_output_type": "onnx_multiclass",
  "num_classes": 6,
  "image_width": 512,
  "image_height": 512,
  "threshold": 0.65,
  "corrosion_pixels": 0,
  "corrosion_area_percent": 0.0,
  "corrosion_level": 0,
  "corrosion_level_name": "Background",
  "severity_pixels": {
    "1": 0,
    "2": 0,
    "3": 0,
    "4": 0,
    "5": 0
  },
  "corrosion_blocks": [],
  "detection_confidence": 0.9997
}
```

## 8. 调参位置

推理过滤参数在 metadata 的 `postprocess` 中：

```json
{
  "threshold": 0.65,
  "min_area_percent": 0.05,
  "min_component_area_ratio": 0.0002
}
```

如果前端反馈误检仍多：

- 提高 `threshold`，例如 `0.70`。
- 提高 `min_area_percent`，例如 `0.10`。
- 提高 `min_component_area_ratio`，过滤更小的碎片。

如果漏检明显：

- 降低 `threshold`，例如 `0.60`。
- 降低 `min_area_percent`。

修改 metadata 后无需重新导出 ONNX，重启 API 即可生效。
