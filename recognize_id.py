'''
Step2: 学号区域识别与填涂检测 (可视化精排版)
Author: 蔡奕辉
Time: 2025-12-20
'''

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.lines import Line2D

# ==================== 0. 参数管理中心 ====================
ID_CONFIG = {
    'TARGET_SIZE': (480, 320),
    'MIN_AREA': 10,
    'MAX_AREA': 450,
    'MAX_W': 50,                # 宽度过滤
    'MAX_H': 15,                # 高度过滤
    'X_GROUP_THRES': 10,
    'Y_GROUP_THRES': 10,
    'CORRECT_THRES': 6.0,
    'ROI_SIZE': (24, 16),
    'PIXEL_THRESH': 100,
    'JUDGE_MODE': 'relative',   # 'relative' 或 'absolute'
    'ABS_FILL_THRESH': 0.6,
    'REL_SCORE_THRESH': 0.3,
}

def _preprocess(img_raw, config):
    """阶段1：双路径图像预处理"""
    img_std = cv2.resize(img_raw, config['TARGET_SIZE'])
    h, w = img_std.shape[:2]
    roi = img_std[int(0.25*h):, :int(0.85*w)]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # 检测路径
    kernel_open = np.ones((6, 2), np.uint8)
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_open)
    _, bin_detect = cv2.threshold(opened, 95, 255, cv2.THRESH_BINARY_INV)
    
    # 计算路径
    kernel_erode = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(gray, kernel_erode, iterations=1)
    
    return roi, gray, opened, bin_detect, eroded

def _extract_valid_contours(bin_img, config):
    """统一轮廓筛选逻辑"""
    cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_cnts = []
    pts_list = []
    for c in cnts:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        if (config['MIN_AREA'] < area < config['MAX_AREA'] and 
            0.1 < (w/h) < 10 and 
            h < config['MAX_H'] and w < config['MAX_W']):
            valid_cnts.append(c)
            pts_list.append({'cx': x + w//2, 'cy': y + h//2, 'area': area})
    return valid_cnts, pd.DataFrame(pts_list)

def _reconstruct_grid(df_pts, config):
    """阶段2：行列聚类与网格对齐"""
    df = df_pts.copy()
    for ax in ['cx', 'cy']:
        df = df.sort_values(ax).reset_index(drop=True)
        col = f'g{ax[-1]}'
        df[col] = (df[ax].diff() > config[f'{ax[-1].upper()}_GROUP_THRES']).cumsum()
        valid = df[col].value_counts()[df[col].value_counts() >= 5].index
        df = df[df[col].isin(valid)].copy()
        df[col] = df[col].map({old: new for new, old in enumerate(sorted(valid))})

    df = df.sort_values('area', ascending=False).drop_duplicates(subset=['gx', 'gy'])
    df_raw_clustered = df.copy()

    map_x = df.groupby('gx')['cx'].median().to_dict()
    map_y = df.groupby('gy')['cy'].median().to_dict()
    
    grid_data = []
    for ix in range(max(map_x.keys()) + 1):
        for iy in range(max(map_y.keys()) + 1):
            if ix not in map_x or iy not in map_y: continue
            ideal_cx, ideal_cy = map_x[ix], map_y[iy]
            match = df[(df['gx'] == ix) & (df['gy'] == iy)]
            if not match.empty:
                curr = match.iloc[0]
                final_x = ideal_cx if abs(curr['cx'] - ideal_cx) > config['CORRECT_THRES'] else curr['cx']
                final_y = ideal_cy if abs(curr['cy'] - ideal_cy) > config['CORRECT_THRES'] else curr['cy']
                status = 'Corrected' if (final_x != curr['cx'] or final_y != curr['cy']) else 'Original'
                grid_data.append({'cx': final_x, 'cy': final_y, 'gx': ix, 'gy': iy, 'status': status})
            else:
                grid_data.append({'cx': ideal_cx, 'cy': ideal_cy, 'gx': ix, 'gy': iy, 'status': 'Filled'})
    return pd.DataFrame(grid_data), df_raw_clustered

def _judge_logic(df_grid, img_eroded, config):
    """阶段3：OMR判定逻辑"""
    rw, rh = config['ROI_SIZE']
    for idx, row in df_grid.iterrows():
        x1, y1, x2, y2 = int(row['cx']-rw//2), int(row['cy']-rh//2), int(row['cx']+rw//2), int(row['cy']+rh//2)
        roi = img_eroded[max(0,y1):y2, max(0,x1):x2]
        df_grid.at[idx, 'abs_ratio'] = np.count_nonzero(roi < config['PIXEL_THRESH'])/roi.size if roi.size > 0 else 0

    df_grid['baseline'] = 0.0
    for gy in sorted(df_grid['gy'].unique()):
        row_data = df_grid[df_grid['gy'] == gy]
        ratios = row_data['abs_ratio'].values.reshape(-1, 1)
        if len(np.unique(ratios)) < 2: baseline = np.median(ratios)
        else:
            km = KMeans(n_clusters=2, random_state=0, n_init=10).fit(ratios)
            baseline = row_data.iloc[km.labels_ == np.argmin(km.cluster_centers_.flatten())]['abs_ratio'].median()
            if (np.max(km.cluster_centers_) - np.min(km.cluster_centers_)) < 0.2:
                baseline = 0.0 if np.min(km.cluster_centers_) > 0.6 else np.median(ratios)
        df_grid.loc[df_grid['gy'] == gy, 'baseline'] = baseline

    df_grid['rel_score'] = df_grid['abs_ratio'] - df_grid['baseline']
    id_str = ""
    df_grid['is_filled'] = False
    for gx in sorted(df_grid['gx'].unique()):
        col = df_grid[df_grid['gx'] == gx]
        best = col.loc[col['rel_score'].idxmax()]
        valid = best['rel_score'] > config['REL_SCORE_THRESH'] if config['JUDGE_MODE'] == 'relative' else best['abs_ratio'] > config['ABS_FILL_THRESH']
        if valid:
            df_grid.at[best.name, 'is_filled'] = True
            id_str += str(int(best['gy']))
        else: id_str += "?"
    return id_str, df_grid

def _visualize_debug(roi, gray, opened, bin_detect, eroded, valid_cnts, df_raw, df_final, id_str, config):
    """分多页精美可视化展示"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # --- Page 1: 预处理与检测 (2x2) ---
    fig1, ax1 = plt.subplots(2, 2, figsize=(12, 10))
    ax1[0,0].imshow(gray, cmap='gray'); ax1[0,0].set_title('1. 原始灰度图')
    ax1[0,1].imshow(opened, cmap='gray'); ax1[0,1].set_title('2. 灰度开运算 (Detection Path)')
    ax1[1,0].imshow(bin_detect, cmap='gray'); ax1[1,0].set_title('3. 检测用二值图')
    img_cnt = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    for c in valid_cnts:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(img_cnt, (x,y), (x+w,y+h), (255,0,0), 1)
    ax1[1,1].imshow(img_cnt); ax1[1,1].set_title('4. 尺度过滤后轮廓')
    plt.tight_layout()

    # --- Page 2: 网格与增强 (2x2) ---
    fig2, ax2 = plt.subplots(2, 2, figsize=(12, 10))
    # 5. 原始点坐标
    ax2[0,0].scatter(df_raw['cx'], df_raw['cy'], c='g', s=20)
    for _, r in df_raw.iterrows(): ax2[0,0].text(r['cx']+3, r['cy'], f"({int(r['gx'])},{int(r['gy'])})", fontsize=7)
    ax2[0,0].set_title('5. 原始聚类点与逻辑坐标'); ax2[0,0].invert_yaxis()
    # 6. 网格状态 (优化图例位置)
    c_map = {'Original':'g', 'Corrected':'r', 'Filled':'y'}
    for st, g in df_final.groupby('status'): ax2[0,1].scatter(g['cx'], g['cy'], c=c_map[st], label=st, s=20)
    ax2[0,1].set_title('6. 网格对正状态分布'); ax2[0,1].invert_yaxis()
    ax2[0,1].legend(loc='upper left', bbox_to_anchor=(1, 1)) # 移到坐标轴外
    # 7-8. 计算路径
    ax2[1,0].imshow(eroded, cmap='gray'); ax2[1,0].set_title('7. 灰度腐蚀增强 (Calculation Path)')
    _, bin_calc = cv2.threshold(eroded, config['PIXEL_THRESH'], 255, cv2.THRESH_BINARY_INV)
    ax2[1,1].imshow(bin_calc, cmap='gray'); ax2[1,1].set_title('8. 计算用二值化图')
    plt.tight_layout()

    # --- Page 3: 填涂判定分析 (2x2) ---
    fig3, ax3 = plt.subplots(2, 2, figsize=(12, 11))
    img_bin_roi = cv2.cvtColor(bin_calc, cv2.COLOR_GRAY2RGB)
    rw, rh = config['ROI_SIZE']
    for _, r in df_final.iterrows(): cv2.rectangle(img_bin_roi, (int(r['cx']-rw//2), int(r['cy']-rh//2)), (int(r['cx']+rw//2), int(r['cy']+rh//2)), (0,255,255), 1)
    ax3[0,0].imshow(img_bin_roi); ax3[0,0].set_title('9. 判定ROI框分布')
    
    p_abs = df_final.pivot(index='gy', columns='gx', values='abs_ratio')
    im10 = ax3[0,1].imshow(p_abs, cmap='RdYlBu_r', vmin=0, vmax=1)
    for y in range(p_abs.shape[0]):
        for x in range(p_abs.shape[1]): ax3[0,1].text(x, y, f"{p_abs.iloc[y,x]:.2f}", ha='center', va='center', fontsize=8)
    ax3[0,1].set_title('10. 绝对占比数值热力图'); plt.colorbar(im10, ax=ax3[0,1])

    if config['JUDGE_MODE'] == 'absolute':
        img_res = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        for _, r in df_final.iterrows():
            if r['is_filled']: cv2.rectangle(img_res, (int(r['cx']-rw//2), int(r['cy']-rh//2)), (int(r['cx']+rw//2), int(r['cy']+rh//2)), (0,255,0), 2)
        ax3[1,0].imshow(img_res); ax3[1,0].set_title(f'11. 识别结果 (ABS): {id_str}')
        fig3.delaxes(ax3[1,1])
    else:
        ax3[1,0].scatter(df_final['abs_ratio'], df_final['gy'], c='blue', alpha=0.5)
        for gy in sorted(df_final['gy'].unique()): ax3[1,0].vlines(df_final[df_final['gy']==gy]['baseline'].iloc[0], gy-0.4, gy+0.4, color='orange', lw=2)
        ax3[1,0].set_title('11. 行绝对占比与基准线'); ax3[1,0].invert_yaxis()
        
        p_rel = df_final.pivot(index='gy', columns='gx', values='rel_score')
        im12 = ax3[1,1].imshow(p_rel, cmap='RdYlBu_r', vmin=-0.1, vmax=0.8)
        for y in range(p_rel.shape[0]):
            for x in range(p_rel.shape[1]): ax3[1,1].text(x, y, f"{p_rel.iloc[y,x]:.2f}", ha='center', va='center', fontsize=8)
        ax3[1,1].set_title('12. 相对得分数值热力图'); plt.colorbar(im12, ax=ax3[1,1])
        
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        img_res = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        for _, r in df_final.iterrows():
            if r['is_filled']: cv2.rectangle(img_res, (int(r['cx']-rw//2), int(r['cy']-rh//2)), (int(r['cx']+rw//2), int(r['cy']+rh//2)), (0,255,0), 2)
        ax4.imshow(img_res); ax4.set_title(f'13. 最终识别学号: {id_str}'); ax4.axis('off')

    plt.tight_layout(); plt.show()

def recognize_id_process(img_raw, custom_config=None, debug=False):
    config = ID_CONFIG.copy()
    if custom_config: config.update(custom_config)
    
    roi, gray, opened, bin_detect, eroded = _preprocess(img_raw, config)
    valid_cnts, df_pts = _extract_valid_contours(bin_detect, config)
    if df_pts.empty: return "Error"
    
    df_grid, df_raw = _reconstruct_grid(df_pts, config)
    id_str, df_final = _judge_logic(df_grid, eroded, config)
    
    if debug:
        _visualize_debug(roi, gray, opened, bin_detect, eroded, valid_cnts, df_raw, df_final, id_str, config)
    return id_str

if __name__ == "__main__":
    from find_region import process_answer_sheet
    _, id_raw = process_answer_sheet('images/test4.jpg', show=False)
    student_id = recognize_id_process(id_raw, debug=True)
    print(f"\n[SUCCESS] 最终识别学号: {student_id}")

