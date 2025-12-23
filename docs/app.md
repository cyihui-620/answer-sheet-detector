# 答题卡识别可视化前端

本仓库在 `src` 下提供基于 Streamlit 的前端：`app.py`。功能概览：

- 上传答题卡图片并自动定位答题区与学号区。
- 使用仓库内成熟模块识别学号与选择题答案（调用 `recognize_id.py` 和 `recognize_answer.py`）。
- 在界面显示识别结果、得分与答题明细（固定显示全部 85 题），并可下载 JSON 结果。
- 支持“显示识别中间图”侧边栏开关，中间过程图默认折叠展示以节省空间。
- 图片采用响应式展示，会根据浏览器/列宽自动缩放。

快速使用：

1. 创建并激活 Conda 虚拟环境（示例 Windows PowerShell）：

```powershell
conda create -n answer python=3.10 -y
conda activate answer
pip install -r requirements.txt
```

2. 运行 Streamlit 应用（在项目根目录运行）：

```powershell
streamlit run src/app.py
```

3. 使用说明：

- 在侧边栏上传答题卡图片（必填）。可选上传标准答案 JSON（格式为题号到选项的映射），否则使用默认 `src/standard_answers.json`。
- 可通过侧边栏的“显示识别中间图”复选框控制是否展示中间可视化图。
- 点击“开始识别”后，页面会显示进度条；完成后左侧为答题区与学号区的响应式预览，右侧显示得分卡，底部显示全部 85 题的识别明细。

输出与下载：

- 页面底部提供“下载识别结果（JSON）”按钮，保存的 JSON 包含 `student_id`、`score`、`total` 与每题 `details`。

实现说明与注意事项：

- 前端直接调用仓库内识别函数，运行前请确保按 `requirements.txt` 安装依赖（`opencv-python`、`numpy`、`pandas`、`matplotlib`、`scikit-learn`、`streamlit` 等）。
- 中间可视化图为 matplotlib Figure 对象，默认以折叠面板呈现；若启用会增加处理与渲染时间。
- 若需要更高识别准确率，请上传更清晰的答题卡图片并保持拍摄时版面平整。

