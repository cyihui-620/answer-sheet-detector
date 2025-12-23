'''
Funtion: 启动项目前端，可视化答题卡识别结果
Note: 详见docs/app.md
Author: 吴峻松
Time: 2025-12-23
'''
import streamlit as st
import json
import tempfile
import os
import cv2
import numpy as np
import base64

from find_region import process_answer_sheet
from recognize_id import recognize_student_id
from recognize_answer import recognize_answers


st.set_page_config(page_title='答题卡识别可视化', layout='wide')

st.markdown("""
<style>
/* 中央标题和卡片样式 */
.stApp header {display:none}
.big-title {text-align:center; font-size:28px; font-weight:700; margin-bottom:10px}
.score-card {background:#f7fbff; padding:12px; border-radius:8px; border:1px solid #e6f0ff}
.small-note {color:#666; font-size:13px}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">答题卡识别可视化展示</div>', unsafe_allow_html=True)

st.sidebar.header('输入')
uploaded_file = st.sidebar.file_uploader('上传答题卡图片', type=['jpg', 'jpeg', 'png'])
answers_file = st.sidebar.file_uploader('上传标准答案（可选 JSON）', type=['json'])
run_btn = st.sidebar.button('开始识别')

st.sidebar.markdown('---')
st.sidebar.markdown('运行提示：上传清晰答题卡图片，等待处理完成后查看识别结果与可视化图。')
show_figs = st.sidebar.checkbox('显示识别中间图', value=True)



def save_uploaded_file(uploaded) -> str:
    suffix = os.path.splitext(uploaded.name)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.getvalue())
    tmp.flush()
    tmp.close()
    return tmp.name


def load_standard_answers(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), 'standard_answers.json')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _to_base64_img(img: np.ndarray) -> str:
    _, buf = cv2.imencode('.png', img)
    b64 = base64.b64encode(buf).decode('utf-8')
    return f'data:image/png;base64,{b64}'


def responsive_image(img: np.ndarray, caption: str = ''):
    data = _to_base64_img(img)
    if caption:
        st.markdown(f"<figure><img src=\"{data}\" style=\"max-width:100%;height:auto;\"><figcaption style=\"text-align:center; font-size:14px; color:#444;\">{caption}</figcaption></figure>", unsafe_allow_html=True)
    else:
        st.markdown(f"<img src=\"{data}\" style=\"max-width:100%;height:auto;\">", unsafe_allow_html=True)


if run_btn:
    if uploaded_file is None:
        st.sidebar.error('请上传答题卡图片后再运行')
    else:
        with st.spinner('正在处理图片——保存上传文件...'):
            img_path = save_uploaded_file(uploaded_file)

        try:
            progress = st.progress(0)
            with st.spinner('定位答题区与学号区...'):
                progress.progress(10)
                sheet_warped, id_region_warped = process_answer_sheet(img_path, show=False)
                progress.progress(30)

            # 结果与预览布局将于下方展示（避免重复预览）

            # 识别学号（使用 recognize_id.py 中的成熟流程），并展示中间图
            with st.spinner('识别学号...'):
                progress.progress(40)
                try:
                    student_id, id_figs = recognize_student_id(id_region_warped, show=show_figs)
                except Exception:
                    student_id = '识别失败'
                    id_figs = []
                progress.progress(55)

            # 结果与预览布局：左侧预览图，右侧分数卡
            left_col, right_col = st.columns([2, 1])
            with left_col:
                responsive_image(cv2.cvtColor(sheet_warped, cv2.COLOR_BGR2RGB), caption='答题区（透视变换后）')
                responsive_image(cv2.cvtColor(id_region_warped, cv2.COLOR_BGR2RGB), caption='学号区（透视变换后）')
            with right_col:
                st.markdown('<div class="score-card">', unsafe_allow_html=True)
                st.markdown('**识别到的学号**')
                st.markdown(f'### {student_id}')
                st.markdown('</div>', unsafe_allow_html=True)

            if id_figs and show_figs:
                st.subheader('学号识别中间图')
                for i, fig in enumerate(id_figs, 1):
                    if fig is not None:
                        with st.expander(f'学号处理图 {i}', expanded=False):
                            st.pyplot(fig)

            # 识别选择题答案，并展示中间可视化图（可能较慢）
            with st.spinner('识别选择题答案...（可能较慢）'):
                progress.progress(60)
                all_answers, figs, region_plots = recognize_answers(sheet_warped, show=show_figs)
                progress.progress(90)

            # 显示答题区总体可视化图
            if show_figs:
                st.subheader('答题区识别中间图')
                for i, fig in enumerate(figs, 1):
                    if fig is not None:
                        with st.expander(f'总体图 {i}', expanded=False):
                            st.pyplot(fig)

            # 不展示各子区域的单独处理图（按要求保留总体图与学号图）

            # 标准答案加载
            if answers_file is not None:
                try:
                    std = json.loads(answers_file.getvalue().decode('utf-8'))
                except Exception:
                    st.sidebar.error('上传的答案文件不是有效 JSON，将使用默认答案')
                    std = load_standard_answers()
            else:
                std = load_standard_answers()

            # 计算成绩
            total = 0
            correct = 0
            result_list = []
            for q in range(1, 86):
                qstr = str(q)
                detected = all_answers.get(q, '/')
                answer_std = std.get(qstr, '/')
                is_correct = (detected == answer_std)
                total += 1
                if is_correct:
                    correct += 1
                result_list.append({'q': q, 'detected': detected, 'standard': answer_std, 'correct': is_correct})

            score = correct
            # 更新进度到完成
            progress.progress(100)

            # 显示得分卡片
            score_col1, score_col2 = st.columns([1, 2])
            with score_col1:
                st.metric(label='得分', value=f'{score}/{total}')
            with score_col2:
                st.markdown('<div class="small-note">识别结果以默认标准答案对比，单项可能存在误判。若需更高准确率，请提供更清晰图片或批量调参。</div>', unsafe_allow_html=True)

            # 展示答题明细表（固定显示全部85题）
            import pandas as pd
            df = pd.DataFrame(result_list)
            st.subheader(f'答题明细（全部 {len(df)} 题）')
            st.dataframe(df)

            # 允许下载结果
            out = {'student_id': student_id, 'score': score, 'total': total, 'details': result_list}
            st.download_button('下载识别结果（JSON）', data=json.dumps(out, ensure_ascii=False, indent=2), file_name='result.json', mime='application/json')

        except Exception as e:
            st.error(f'识别过程中发生错误：{e}')
