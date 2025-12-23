import streamlit as st
import json
import tempfile
import os
import cv2
import numpy as np

from find_region import process_answer_sheet
from recognize_id import recognize_student_id
from recognize_answer import recognize_answers


st.set_page_config(page_title='答题卡识别可视化', layout='wide')

st.title('答题卡识别可视化展示')

st.sidebar.header('输入')
uploaded_file = st.sidebar.file_uploader('上传答题卡图片', type=['jpg', 'jpeg', 'png'])
answers_file = st.sidebar.file_uploader('上传标准答案（可选 JSON）', type=['json'])
run_btn = st.sidebar.button('开始识别')


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


if run_btn:
    if uploaded_file is None:
        st.sidebar.error('请上传答题卡图片后再运行')
    else:
        with st.spinner('正在处理图片——保存上传文件...'):
            img_path = save_uploaded_file(uploaded_file)

        try:
            with st.spinner('定位答题区与学号区...'):
                sheet_warped, id_region_warped = process_answer_sheet(img_path, show=False)

            st.subheader('定位结果预览')
            col1, col2 = st.columns(2)
            with col1:
                st.image(cv2.cvtColor(sheet_warped, cv2.COLOR_BGR2RGB), caption='答题区（透视变换后）')
            with col2:
                st.image(cv2.cvtColor(id_region_warped, cv2.COLOR_BGR2RGB), caption='学号区（透视变换后）')

            # 识别学号（使用 recognize_id.py 中的成熟流程）
            with st.spinner('识别学号...'):
                try:
                    student_id, _ = recognize_student_id(id_region_warped, show=False)
                except Exception:
                    student_id = '识别失败'

            st.markdown('**识别到的学号：** ' + str(student_id))

            # 识别选择题答案
            with st.spinner('识别选择题答案...（可能较慢）'):
                all_answers, figs, region_plots = recognize_answers(sheet_warped, show=False)

            # 显示部分可视化图
            st.subheader('识别过程示意图')
            for i, fig in enumerate(figs):
                if fig is not None:
                    st.pyplot(fig)

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
            st.subheader('得分')
            st.write(f'正确：{correct} / {total} ，得分：{score}')

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
