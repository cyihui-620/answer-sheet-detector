# 答题卡识别可视化前端

本仓库在 `src` 下新增了一个基于 Streamlit 的前端：`app.py`。用户可以通过网页界面上传答题卡图片与标准答案（可选），展示识别结果并下载识别 JSON。

快速使用：

1. 创建并激活虚拟环境（例如 Windows PowerShell）：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. 运行 Streamlit 应用（在项目根目录运行）：

```powershell
streamlit run src/app.py
```

3. 在侧边栏上传答题卡图片（必须），可选上传标准答案 JSON（否则使用默认 `src/standard_answers.json`）。点击“开始识别”后等待处理结果。

注意：识别依赖 OpenCV 和项目中已有的模块（`find_region.py`、`id.py`、`recognize_answer.py`）。识别耗时与图片分辨率和机器性能有关。
