"""
Step2: 学号填涂识别
Author: 蔡奕辉
Date: 2025-12-20
Description: 完整的学号填涂识别流程，包括预处理、聚类、插值和填涂检测
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from find_region import process_answer_sheet
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib
from typing import Tuple, Optional, Dict, Any

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 默认参数配置 ====================
DEFAULT_PARAMS = {
    'target_size': (480, 320),  # 缩放尺寸
    'clahe_clip': 3.0,          # CLAHE
    'clahe_grid': (8, 8),       
    'adaptive_block': 31,       # 自适应二值化
    'adaptive_c': 15,           
    'morph_type': 'close',      # 形态学操作
    'morph_kernel': (3, 3),     
    'calc_erode_iter': 1,       # 腐蚀
    
    # 聚类参数
    'min_area': 35,
    'max_area': 600,
    'min_width': 5,
    'max_width': 50,
    'min_height': 5,
    'max_height': 30,
    'expected_rows': 10,  # 学号0-9
    'expected_cols': 11,  # 学号11位数
    
    # ROI和填涂检测参数
    'roi_size': (20, 12),       # ROI大小 (宽, 高)
    'fill_threshold': 0.4,      # 填涂阈值 (绝对占比)
}


# ==================== 辅助函数 ====================
def independent_row_col_clustering(df_points: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """独立的行列聚类，确保行列号从上到下、从左到右排序"""
    df = df_points.copy()
    
    print("=" * 50)
    print("步骤1: 独立行列聚类")
    print(f"输入点数: {len(df)}")
    
    # 1. 独立行聚类（对所有y坐标聚类）
    y_coords = df['cy'].values.reshape(-1, 1)
    n_rows = min(params['expected_rows'], len(df))
    
    kmeans_y = KMeans(n_clusters=n_rows, random_state=42, n_init=10)
    df['row_label'] = kmeans_y.fit_predict(y_coords)
    
    # 计算每个聚类标签的y坐标中位数
    row_centers = {}
    for label in df['row_label'].unique():
        row_centers[label] = df[df['row_label'] == label]['cy'].median()
    
    # 按y坐标从小到大排序（从上到下）
    sorted_labels = sorted(row_centers.items(), key=lambda x: x[1])
    
    # 重新分配行号：y坐标最小的为行0，最大的为行9
    row_mapping = {old_label: i for i, (old_label, _) in enumerate(sorted_labels)}
    df['row_id'] = df['row_label'].map(row_mapping)
    
    # 2. 独立列聚类（对所有x坐标聚类）
    x_coords = df['cx'].values.reshape(-1, 1)
    n_cols = min(params['expected_cols'], len(df))
    
    kmeans_x = KMeans(n_clusters=n_cols, random_state=42, n_init=10)
    df['col_label'] = kmeans_x.fit_predict(x_coords)
    
    # 计算每个聚类标签的x坐标中位数
    col_centers = {}
    for label in df['col_label'].unique():
        col_centers[label] = df[df['col_label'] == label]['cx'].median()
    
    # 按x坐标从小到大排序（从左到右）
    sorted_col_labels = sorted(col_centers.items(), key=lambda x: x[1])
    
    # 重新分配列号：x坐标最小的为列0，最大的为列10
    col_mapping = {old_label: i for i, (old_label, _) in enumerate(sorted_col_labels)}
    df['col_id'] = df['col_label'].map(col_mapping)
    
    print(f"聚类结果: {df['row_id'].nunique()}行 × {df['col_id'].nunique()}列")
    print("=" * 50)
    
    return df


def simple_deduplicate(df_clustered: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """简单去重：同一行列位置取均值"""
    df = df_clustered.copy()
    
    if 'row_id' not in df.columns or 'col_id' not in df.columns:
        return df
    
    print("步骤2: 去重处理")
    
    deduplicated = []
    duplicates_found = 0
    
    for (row_id, col_id), group in df.groupby(['row_id', 'col_id']):
        if len(group) > 1:
            # 取均值
            mean_point = {
                'row_id': int(row_id),
                'col_id': int(col_id),
                'cx': group['cx'].mean(),
                'cy': group['cy'].mean(),
                'area': group['area'].mean(),
                'x': int(group['x'].mean()),
                'y': int(group['y'].mean()),
                'w': int(group['w'].mean()),
                'h': int(group['h'].mean()),
                'status': 'merged',
                'count': len(group)
            }
            deduplicated.append(mean_point)
            duplicates_found += 1
        else:
            point = group.iloc[0].to_dict()
            point['row_id'] = int(point['row_id'])
            point['col_id'] = int(point['col_id'])
            point['status'] = 'original'
            point['count'] = 1
            deduplicated.append(point)
    
    result = pd.DataFrame(deduplicated)
    print(f"去重结果: 合并了 {duplicates_found} 组重复点，剩余 {len(result)} 个点")
    
    return result


def improved_interpolate(df_dedup: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """改进的插值算法：考虑网格间距的线性插值"""
    df = df_dedup.copy()
    
    if 'row_id' not in df.columns or 'col_id' not in df.columns:
        return df
    
    print("步骤3: 网格插值")
    
    # 获取行和列的范围
    rows = sorted(df['row_id'].unique())
    cols = sorted(df['col_id'].unique())
    
    # 创建点字典，方便快速查找
    point_dict = {}
    for _, point in df.iterrows():
        point_dict[(int(point['row_id']), int(point['col_id']))] = point.to_dict()
    
    all_points = []
    interpolated_count = 0
    
    # 计算平均行间距和列间距
    row_spacings = []
    col_spacings = []
    
    # 计算行间距（垂直间距）
    for col in cols:
        col_points = df[df['col_id'] == col]
        if len(col_points) > 1:
            col_points = col_points.sort_values('row_id')
            for i in range(1, len(col_points)):
                row_diff = col_points.iloc[i]['row_id'] - col_points.iloc[i-1]['row_id']
                y_diff = col_points.iloc[i]['cy'] - col_points.iloc[i-1]['cy']
                if row_diff > 0:
                    row_spacings.append(y_diff / row_diff)
    
    # 计算列间距（水平间距）
    for row in rows:
        row_points = df[df['row_id'] == row]
        if len(row_points) > 1:
            row_points = row_points.sort_values('col_id')
            for i in range(1, len(row_points)):
                col_diff = row_points.iloc[i]['col_id'] - row_points.iloc[i-1]['col_id']
                x_diff = row_points.iloc[i]['cx'] - row_points.iloc[i-1]['cx']
                if col_diff > 0:
                    col_spacings.append(x_diff / col_diff)
    
    avg_row_spacing = np.mean(row_spacings) if row_spacings else 20
    avg_col_spacing = np.mean(col_spacings) if col_spacings else 20
    
    for row in rows:
        for col in cols:
            # 检查这个位置是否有点
            if (row, col) in point_dict:
                all_points.append(point_dict[(row, col)])
            else:
                # 需要插值
                interpolated_point = {
                    'row_id': int(row),
                    'col_id': int(col),
                    'status': 'interpolated',
                    'count': 0
                }
                
                # 1. 寻找最近的上下邻居
                up_neighbor = None
                down_neighbor = None
                
                # 向上找最近邻居
                for r in range(row-1, -1, -1):
                    if (r, col) in point_dict:
                        up_neighbor = (r, point_dict[(r, col)]['cy'])
                        break
                
                # 向下找最近邻居
                for r in range(row+1, max(rows)+1):
                    if (r, col) in point_dict:
                        down_neighbor = (r, point_dict[(r, col)]['cy'])
                        break
                
                # 2. 计算y坐标
                if up_neighbor and down_neighbor:
                    # 上下都有，线性插值
                    up_row, up_y = up_neighbor
                    down_row, down_y = down_neighbor
                    ratio = (row - up_row) / (down_row - up_row)
                    interpolated_point['cy'] = up_y + (down_y - up_y) * ratio
                elif up_neighbor:
                    # 只有上邻居，向下外推
                    up_row, up_y = up_neighbor
                    interpolated_point['cy'] = up_y + (row - up_row) * avg_row_spacing
                elif down_neighbor:
                    # 只有下邻居，向上外推
                    down_row, down_y = down_neighbor
                    interpolated_point['cy'] = down_y - (down_row - row) * avg_row_spacing
                else:
                    # 没有邻居，用该列的均值
                    col_points = df[df['col_id'] == col]
                    if len(col_points) > 0:
                        interpolated_point['cy'] = col_points['cy'].mean()
                    else:
                        interpolated_point['cy'] = df['cy'].mean()
                
                # 3. 寻找最近的左右邻居
                left_neighbor = None
                right_neighbor = None
                
                # 向左找最近邻居
                for c in range(col-1, -1, -1):
                    if (row, c) in point_dict:
                        left_neighbor = (c, point_dict[(row, c)]['cx'])
                        break
                
                # 向右找最近邻居
                for c in range(col+1, max(cols)+1):
                    if (row, c) in point_dict:
                        right_neighbor = (c, point_dict[(row, c)]['cx'])
                        break
                
                # 4. 计算x坐标
                if left_neighbor and right_neighbor:
                    # 左右都有，线性插值
                    left_col, left_x = left_neighbor
                    right_col, right_x = right_neighbor
                    ratio = (col - left_col) / (right_col - left_col)
                    interpolated_point['cx'] = left_x + (right_x - left_x) * ratio
                elif left_neighbor:
                    # 只有左邻居，向右外推
                    left_col, left_x = left_neighbor
                    interpolated_point['cx'] = left_x + (col - left_col) * avg_col_spacing
                elif right_neighbor:
                    # 只有右邻居，向左外推
                    right_col, right_x = right_neighbor
                    interpolated_point['cx'] = right_x - (right_col - col) * avg_col_spacing
                else:
                    # 没有邻居，用该行的均值
                    row_points = df[df['row_id'] == row]
                    if len(row_points) > 0:
                        interpolated_point['cx'] = row_points['cx'].mean()
                    else:
                        interpolated_point['cx'] = df['cx'].mean()
                
                # 5. 填充其他字段
                interpolated_point['area'] = df['area'].mean() if len(df) > 0 else 100
                interpolated_point['x'] = int(interpolated_point['cx'] - 10)
                interpolated_point['y'] = int(interpolated_point['cy'] - 5)
                interpolated_point['w'] = 20
                interpolated_point['h'] = 10
                
                all_points.append(interpolated_point)
                interpolated_count += 1
    
    result = pd.DataFrame(all_points)
    result = result.sort_values(['row_id', 'col_id']).reset_index(drop=True)
    
    print(f"插值结果: 插值了 {interpolated_count} 个点，总点数: {len(result)}")
    
    return result


def detect_filling(df_grid: pd.DataFrame, bin_calc: np.ndarray, params: Dict[str, Any]) -> pd.DataFrame:
    """检测每个网格点是否填涂"""
    df = df_grid.copy()
    
    if df.empty:
        return df
    
    print("步骤4: 填涂检测")
    
    # 初始化填涂相关列
    df['fill_ratio'] = 0.0
    df['is_filled'] = False
    df['fill_type'] = 'unknown'
    
    # 获取ROI大小
    roi_w, roi_h = params['roi_size']
    threshold = params['fill_threshold']
    
    for idx, row in df.iterrows():
        # 计算ROI边界
        x1 = int(row['cx'] - roi_w // 2)
        y1 = int(row['cy'] - roi_h // 2)
        x2 = int(row['cx'] + roi_w // 2)
        y2 = int(row['cy'] + roi_h // 2)
        
        # 确保ROI在图像范围内
        h, w = bin_calc.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 > x1 and y2 > y1:
            roi = bin_calc[y1:y2, x1:x2]
            if roi.size > 0:
                # 计算白色像素占比
                total_pixels = roi.size
                white_pixels = cv2.countNonZero(roi)
                fill_ratio = white_pixels / total_pixels
                
                df.at[idx, 'fill_ratio'] = fill_ratio
                
                # 判断是否填涂
                is_filled = fill_ratio > threshold
                df.at[idx, 'is_filled'] = is_filled
                
                # 标记填涂类型
                if is_filled:
                    if row['status'] == 'interpolated':
                        df.at[idx, 'fill_type'] = 'interpolated_filled'
                    else:
                        df.at[idx, 'fill_type'] = 'filled'
                else:
                    if row['status'] == 'interpolated':
                        df.at[idx, 'fill_type'] = 'interpolated_empty'
                    else:
                        df.at[idx, 'fill_type'] = 'empty'
    
    # 识别学号
    student_id = _recognize_student_id(df)
    
    df['student_id'] = student_id
    
    print(f"填涂检测结果:")
    print(f"  阈值: {threshold}")
    print(f"  学号: {student_id}")
    print(f"  填涂点: {sum(df['is_filled'])}个")
    print("=" * 50)
    
    return df


def _recognize_student_id(df: pd.DataFrame) -> str:
    """从填涂结果识别学号"""
    if 'row_id' not in df.columns or 'col_id' not in df.columns:
        return ""
    
    # 获取列数范围
    cols = sorted(df['col_id'].unique())
    
    student_id = ""
    
    for col in cols:
        # 获取该列的所有点
        col_points = df[df['col_id'] == col]
        
        # 查找填涂的点
        filled_points = col_points[col_points['is_filled']]
        
        if len(filled_points) == 1:
            # 只有一个填涂点，取其行号
            row_id = int(filled_points.iloc[0]['row_id'])
            student_id += str(row_id)
        elif len(filled_points) > 1:
            # 多个填涂点，取填涂比例最高的
            best_idx = filled_points['fill_ratio'].idxmax()
            row_id = int(filled_points.loc[best_idx, 'row_id'])
            student_id += str(row_id)
        else:
            # 没有填涂点
            student_id += "?"
    
    return student_id


# ==================== 可视化函数 ====================
def visualize_complete_results(roi: np.ndarray, df_raw: pd.DataFrame, df_independent: pd.DataFrame, 
                              df_dedup: pd.DataFrame, df_interp: pd.DataFrame, df_final: pd.DataFrame, 
                              bin_calc: np.ndarray, gray_enhanced: np.ndarray, 
                              bin_detect: np.ndarray, morph_detect: np.ndarray, params: Dict[str, Any]) -> None:
    """可视化完整结果"""
    
    # 创建一个大型图形，使用网格布局
    fig = plt.figure(figsize=(18, 12))
    
    # 设置全局布局参数
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.92, 
                       wspace=0.2, hspace=0.4)  # 增加行间距
    
    # 1. 预处理步骤 (4个)
    ax1 = plt.subplot(4, 4, 1)
    ax1.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB) if len(roi.shape) == 3 else roi, 
              cmap='gray' if len(roi.shape) == 2 else None)
    ax1.set_title('1. 原始学号区域', fontsize=10, pad=5)
    ax1.axis('off')
    
    ax2 = plt.subplot(4, 4, 2)
    ax2.imshow(gray_enhanced, cmap='gray')
    ax2.set_title('2. CLAHE增强', fontsize=10, pad=5)
    ax2.axis('off')
    
    ax3 = plt.subplot(4, 4, 3)
    ax3.imshow(bin_detect, cmap='gray')
    ax3.set_title('3. 自适应二值化', fontsize=10, pad=5)
    ax3.axis('off')
    
    ax4 = plt.subplot(4, 4, 4)
    ax4.imshow(morph_detect, cmap='gray')
    ax4.set_title('4. 形态学闭运算', fontsize=10, pad=5)
    ax4.axis('off')
    
    # 2. 轮廓检测
    ax5 = plt.subplot(4, 4, 5)
    contour_img = cv2.cvtColor(roi.copy(), cv2.COLOR_BGR2RGB) if len(roi.shape) == 3 else cv2.cvtColor(roi.copy(), cv2.COLOR_GRAY2RGB)
    for _, point in df_raw.iterrows():
        cv2.rectangle(contour_img, (int(point['x']), int(point['y'])), 
                     (int(point['x']+point['w']), int(point['y']+point['h'])), 
                     (0, 255, 0), 1)
    ax5.imshow(contour_img)
    ax5.set_title(f'5. 有效轮廓检测 ({len(df_raw)}个)', fontsize=10, pad=5)
    ax5.axis('off')
    
    # 3. 计算用二值图
    ax6 = plt.subplot(4, 4, 6)
    ax6.imshow(bin_calc, cmap='gray')
    ax6.set_title('6. 计算用二值图(腐蚀)', fontsize=10, pad=5)
    ax6.axis('off')
    
    # 4. 独立行列聚类
    ax7 = plt.subplot(4, 4, 7)
    if 'row_id' in df_independent.columns and 'col_id' in df_independent.columns:
        unique_rows = sorted(df_independent['row_id'].unique())
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_rows)))
        
        for row_id, color in zip(unique_rows, colors):
            row_data = df_independent[df_independent['row_id'] == row_id]
            ax7.scatter(row_data['cx'], row_data['cy'], 
                       color=color, s=30, label=f'行{row_id}', alpha=0.7)
    
    ax7.legend(loc='upper right', fontsize='x-small')
    ax7.invert_yaxis()
    ax7.set_title('7. 独立行列聚类', fontsize=10, pad=5)
    ax7.grid(True, alpha=0.3)
    ax7.set_xlabel('X坐标', fontsize=8)
    ax7.set_ylabel('Y坐标', fontsize=8)
    ax7.tick_params(labelsize=8)
    
    # 5. 插值/合并结果
    ax8 = plt.subplot(4, 4, 8)
    if 'status' in df_interp.columns:
        colors = {'original': 'blue', 'merged': 'orange', 'interpolated': 'green'}
        markers = {'original': 'o', 'merged': 's', 'interpolated': '^'}
        
        for status, color in colors.items():
            status_data = df_interp[df_interp['status'] == status]
            if not status_data.empty:
                marker = markers[status]
                ax8.scatter(status_data['cx'], status_data['cy'], 
                          color=color, s=40, marker=marker, 
                          label=status, alpha=0.7)
    
    ax8.legend(loc='upper right', fontsize='x-small')
    ax8.invert_yaxis()
    ax8.set_title('8. 插值/合并结果', fontsize=10, pad=5)
    ax8.grid(True, alpha=0.3)
    ax8.set_xlabel('X坐标', fontsize=8)
    ax8.set_ylabel('Y坐标', fontsize=8)
    ax8.tick_params(labelsize=8)
    
    # 6. ROI框显示
    ax9 = plt.subplot(4, 4, 9)
    if not df_final.empty:
        img_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB) if len(roi.shape) == 3 else cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
        
        roi_w, roi_h = params['roi_size']
        
        for _, point in df_final.iterrows():
            x1 = int(point['cx'] - roi_w // 2)
            y1 = int(point['cy'] - roi_h // 2)
            x2 = int(point['cx'] + roi_w // 2)
            y2 = int(point['cy'] + roi_h // 2)
            
            # 根据填涂状态选择颜色
            if point['is_filled']:
                color = (0, 255, 0)  # 绿色：填涂
            else:
                color = (255, 0, 0)  # 红色：未填涂
            
            cv2.rectangle(img_roi, (x1, y1), (x2, y2), color, 1)
        
        ax9.imshow(img_roi)
        ax9.set_title('9. ROI框显示(红:未填,绿:已填)', fontsize=10, pad=5)
    ax9.axis('off')
    
    # 7. 填涂结果
    ax10 = plt.subplot(4, 4, 10)
    if 'fill_type' in df_final.columns:
        colors = {
            'filled': 'red',
            'empty': 'blue',
            'interpolated_filled': 'orange',
            'interpolated_empty': 'green'
        }
        markers = {
            'filled': 'o',
            'empty': 'o',
            'interpolated_filled': '^',
            'interpolated_empty': '^'
        }
        
        for fill_type, color in colors.items():
            type_data = df_final[df_final['fill_type'] == fill_type]
            if not type_data.empty:
                marker = markers[fill_type]
                ax10.scatter(type_data['cx'], type_data['cy'], 
                          color=color, s=40, marker=marker, 
                          label=fill_type, alpha=0.7)
    
    ax10.legend(loc='upper right', fontsize='x-small')
    ax10.invert_yaxis()
    student_id = df_final.iloc[0]['student_id'] if 'student_id' in df_final.columns else "N/A"
    ax10.set_title(f'10. 填涂结果\n学号: {student_id}', fontsize=10, pad=5)
    ax10.grid(True, alpha=0.3)
    ax10.set_xlabel('X坐标', fontsize=8)
    ax10.set_ylabel('Y坐标', fontsize=8)
    ax10.tick_params(labelsize=8)
    
    # 8. 填涂比例热力图
    ax11 = plt.subplot(4, 4, 11)
    if 'fill_ratio' in df_final.columns and 'row_id' in df_final.columns and 'col_id' in df_final.columns:
        # 创建填涂比例矩阵
        rows = sorted(df_final['row_id'].unique())
        cols = sorted(df_final['col_id'].unique())
        
        fill_matrix = np.zeros((len(rows), len(cols)))
        
        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                point = df_final[(df_final['row_id'] == row) & (df_final['col_id'] == col)]
                if not point.empty:
                    fill_matrix[i, j] = point.iloc[0]['fill_ratio']
        
        im = ax11.imshow(fill_matrix, cmap='RdYlBu_r', vmin=0, vmax=1)
        
        # 添加数值标签
        for i in range(fill_matrix.shape[0]):
            for j in range(fill_matrix.shape[1]):
                ax11.text(j, i, f"{fill_matrix[i, j]:.2f}", 
                        ha='center', va='center', fontsize=6,
                        color='black' if fill_matrix[i, j] < 0.5 else 'white')
        
        plt.colorbar(im, ax=ax11, fraction=0.046, pad=0.04)
        ax11.set_title('11. 填涂比例热力图', fontsize=10, pad=5)
        ax11.set_xlabel('列号', fontsize=8)
        ax11.set_ylabel('行号', fontsize=8)
        ax11.set_xticks(range(len(cols)))
        ax11.set_yticks(range(len(rows)))
        ax11.set_xticklabels([str(c) for c in cols], fontsize=6)
        ax11.set_yticklabels([str(r) for r in rows], fontsize=6)
    
    # 9. 统计信息
    ax12 = plt.subplot(4, 4, 12)
    ax12.axis('off')
    
    # 创建统计文本
    student_id = df_final.iloc[0]['student_id'] if 'student_id' in df_final.columns else "N/A"
    stats_text = f"学号: {student_id}\n\n"
    stats_text += f"原始点数: {len(df_raw)}\n"
    stats_text += f"网格大小: {df_final['row_id'].nunique()}行 × {df_final['col_id'].nunique()}列\n"
    stats_text += f"总点数: {len(df_final)}\n"
    
    if 'status' in df_final.columns:
        status_counts = df_final['status'].value_counts()
        stats_text += "\n点状态统计:\n"
        for status, count in status_counts.items():
            if status == 'original':
                stats_text += f"  原始点: {count}个\n"
            elif status == 'merged':
                stats_text += f"  合并点: {count}个\n"
            elif status == 'interpolated':
                stats_text += f"  插值点: {count}个\n"
            else:
                stats_text += f"  {status}: {count}个\n"
    
    if 'is_filled' in df_final.columns:
        filled_count = df_final['is_filled'].sum()
        stats_text += f"\n填涂点: {filled_count}个\n"
        stats_text += f"未填涂点: {len(df_final) - filled_count}个"
        
        # 添加填涂检测阈值
        stats_text += f"\n\n阈值: {params['fill_threshold']}"
    
    ax12.text(0.1, 0.5, stats_text, fontsize=9, 
             verticalalignment='center', transform=ax12.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax12.set_title('12. 统计信息', fontsize=10, pad=5)
    
    # 添加主标题
    plt.suptitle('学号识别 - 完整流程分析', fontsize=16, y=0.98)
    plt.show()
    
    # 单独显示填涂矩阵
    _print_detailed_results(df_final)


def _print_detailed_results(df_final: pd.DataFrame) -> None:
    """打印详细的识别结果"""
    print("\n" + "="*60)
    print("识别结果详情")
    print("="*60)
    
    if not df_final.empty:
        student_id = df_final.iloc[0]['student_id'] if 'student_id' in df_final.columns else "N/A"
        # 显示填涂矩阵
        print(f"网格大小: {df_final['row_id'].nunique()}行 × {df_final['col_id'].nunique()}列")
        print(f"识别学号: {student_id}")
        
        if 'fill_type' in df_final.columns:
            print("\n填涂状态统计:")
            fill_counts = df_final['fill_type'].value_counts()
            for fill_type, count in fill_counts.items():
                if fill_type == 'filled':
                    print(f"  填涂点: {count}个")
                elif fill_type == 'empty':
                    print(f"  未填涂点: {count}个")
                elif fill_type == 'interpolated_filled':
                    print(f"  插值填涂点: {count}个")
                elif fill_type == 'interpolated_empty':
                    print(f"  插值未填涂点: {count}个")
                else:
                    print(f"  {fill_type}: {count}个")
        
        # 显示填涂矩阵
        print("\n填涂矩阵 (行0-9, 列0-10):")
        rows = sorted(df_final['row_id'].unique())
        cols = sorted(df_final['col_id'].unique())
        
        # 创建网格表示
        grid = []
        for row in rows:
            row_chars = []
            for col in cols:
                point = df_final[(df_final['row_id'] == row) & (df_final['col_id'] == col)]
                if not point.empty:
                    if point.iloc[0]['is_filled']:
                        if point.iloc[0]['status'] == 'interpolated':
                            row_chars.append('○')
                        else:
                            row_chars.append('●')
                    else:
                        if point.iloc[0]['status'] == 'interpolated':
                            row_chars.append('□')
                        else:
                            row_chars.append('▪')
                else:
                    row_chars.append(' ')
            grid.append(row_chars)
        
        # 打印网格
        print("   " + " ".join([f"{c:>2}" for c in cols]))
        for i, row_chars in enumerate(grid):
            print(f"{i:2} " + " ".join(row_chars))
        
        print("\n图例: ●=填涂 ○=插值填涂 ▪=未填涂 □=插值未填涂")


# ==================== 主函数 ====================
def recognize_student_id(image_path: str, 
                        params: Optional[Dict[str, Any]] = None,
                        show: bool = True) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    完整的学号识别流程
    
    Args:
        image_path: 输入图片路径
        params: 参数字典，不传则使用默认参数
        show: 是否显示可视化结果
    
    Returns:
        Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
            df_final: 最终结果DataFrame
            roi: 学号区域图像
            morph_detect: 形态学处理结果
            bin_calc: 计算用二值图
    """
    # 使用默认参数或传入参数
    if params is None:
        params = DEFAULT_PARAMS.copy()
    
    print("=" * 60)
    print("开始学号识别流程")
    print(f"图片路径: {image_path}")
    print("=" * 60)
    
    # 1. 获取学号区域
    try:
        res = process_answer_sheet(image_path, show=False)
        if len(res) == 3: 
            _, roi_raw, _ = res
        else: 
            _, roi_raw = res
    except Exception as e:
        print(f"区域提取失败: {e}，使用原图测试")
        roi_raw = cv2.imread(image_path)
        if roi_raw is None:
            raise ValueError(f"无法读取图片: {image_path}")

    # 2. 预处理
    resized = cv2.resize(roi_raw, params['target_size'])
    w, h = params['target_size']
    roi = resized[int(0.25*h):, :int(0.85*w)]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=params['clahe_clip'], tileGridSize=params['clahe_grid'])
    gray_enhanced = clahe.apply(gray)
    
    bin_detect = cv2.adaptiveThreshold(gray_enhanced, 255, 
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 
                                       params['adaptive_block'], 
                                       params['adaptive_c'])
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, params['morph_kernel'])
    if params['morph_type'] == 'close':
        morph_detect = cv2.morphologyEx(bin_detect, cv2.MORPH_CLOSE, kernel)
    else:
        morph_detect = bin_detect
        
    # 3. 轮廓检测
    cnts, _ = cv2.findContours(morph_detect, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    points = []
    for c in cnts:
        area = cv2.contourArea(c)
        x, y, w_rect, h_rect = cv2.boundingRect(c)
        
        if (params['min_area'] < area < params['max_area'] and 
            0.2 < w_rect/h_rect < 5 and 
            params['min_height'] < h_rect < params['max_height'] and 
            w_rect < params['max_width']):
            
            cx = x + w_rect // 2
            cy = y + h_rect // 2
            
            points.append({
                'x': x, 'y': y, 'w': w_rect, 'h': h_rect,
                'cx': cx, 'cy': cy, 
                'area': area,
            })
    
    df_points = pd.DataFrame(points) if points else pd.DataFrame()
    
    # 4. 计算路径增强（用于填涂检测）
    kernel_calc = np.ones((3, 3), np.uint8)
    bin_calc = cv2.erode(morph_detect, kernel_calc, iterations=params['calc_erode_iter'])
    
    # 5. 聚类和填涂检测
    if not df_points.empty:
        # 独立行列聚类
        df_independent = independent_row_col_clustering(df_points.copy(), params)
        
        # 去重
        df_dedup = simple_deduplicate(df_independent.copy(), params)
        
        # 插值
        df_interp = improved_interpolate(df_dedup.copy(), params)
        
        # 填涂检测
        df_final = detect_filling(df_interp.copy(), bin_calc, params)
        
        # 可视化
        if show:
            visualize_complete_results(roi, df_points, df_independent, df_dedup, df_interp, 
                                      df_final, bin_calc, gray_enhanced, bin_detect, morph_detect, params)
        
    else:
        df_final = pd.DataFrame()
        print("没有检测到有效点")
    
    print("学号识别流程完成")
    print("=" * 60)
    
    return df_final, roi, morph_detect, bin_calc


def get_student_id(df_final: pd.DataFrame) -> str:
    """
    从结果DataFrame中提取学号
    
    Args:
        df_final: recognize_student_id函数返回的DataFrame
    
    Returns:
        学号字符串
    """
    if df_final.empty or 'student_id' not in df_final.columns:
        return ""
    
    return df_final.iloc[0]['student_id']


# ==================== 示例主程序 ====================
if __name__ == "__main__":
    """
    示例主程序，展示如何使用本模块
    """
    import sys
    
    def main():
        # 命令行参数处理
        if len(sys.argv) > 1:
            image_path = sys.argv[1]
        else:
            # 默认图片路径
            image_path = 'images/test14.jpg'
            print(f"使用默认图片: {image_path}")
        
        # 方法1: 使用默认参数
        print("方法1: 使用默认参数")
        df_final, roi, morph_detect, bin_calc = recognize_student_id(
            image_path=image_path,
            show=True
        )
        
        # 获取学号
        student_id = get_student_id(df_final)
        print(f"识别到的学号: {student_id}")
        
        # 方法2: 自定义参数
        print("\n" + "="*60)
        print("方法2: 自定义参数")
        custom_params = DEFAULT_PARAMS.copy()
        custom_params['fill_threshold'] = 0.35  # 调整填涂阈值
        
        df_final2, _, _, _ = recognize_student_id(
            image_path=image_path,
            params=custom_params,
            show=False  # 不显示可视化
        )
        
        student_id2 = get_student_id(df_final2)
        print(f"自定义参数识别到的学号: {student_id2}")
        
        # 方法3: 批量处理（示例）
        print("\n" + "="*60)
        print("方法3: 批量处理示例")
        
        # 模拟多个图片路径
        image_paths = [
            'images/test13.jpg',
            # 'images/test14.jpg',
            # 'images/test15.jpg',
        ]
        
        for img_path in image_paths:
            try:
                df, _, _, _ = recognize_student_id(img_path, show=False)
                sid = get_student_id(df)
                print(f"图片: {img_path}, 学号: {sid}")
            except Exception as e:
                print(f"处理 {img_path} 时出错: {e}")
        
        print("\n学号识别示例程序运行完成！")
    
    # 运行主程序
    main()