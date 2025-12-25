"""
Step3: 选择题填涂识别
Author: 蔡奕辉
Date: 2025-12-23
"""

from find_region import process_answer_sheet
from recognize_id import independent_row_col_clustering, remove_duplicated_and_outliers, grid_linear_interpoltate
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from typing import Tuple, Optional, Dict, Any, List

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#--------------------------------------------------------------------------------------------------------------------
# Part1: 图像预处理
def preprocess_image(image: np.ndarray, show: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, plt.Figure]:
    """
    图像预处理
    
    参数:
    ◦ image: 原始BGR图像

    ◦ show: 是否显示处理过程图

    
    返回:
    ◦ tuple: (标准化图像, 二值化图像, 闭运算图像, 图表)

    """
    # Step1：尺寸标准化
    target_size = (820, 680)
    resized = cv2.resize(image, target_size)
    
    # Step2: 遮盖左下角文字区域
    masked = resized.copy()
    mask_bottom = int(target_size[0] * 0.35)
    mask_right = int(target_size[1] * 0.26)
    masked[mask_bottom:, :mask_right] = 255
    
    # Step3: 转灰度图
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    
    # Step4：clahe->增强图像对比度
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_enhanced = clahe.apply(gray)
    
    # Step5：自适应二值化
    binary = cv2.adaptiveThreshold(gray_enhanced, 255, 
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY_INV, 
                                    blockSize=31, 
                                    C=15)
    
    # Step6: 闭运算(将各点位轮廓闭合成方块)
    right_strip_width = 15
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel) 
    closed[:, -right_strip_width:] = 0  # 最右侧归零
    
    if not show:
        return resized, binary, closed, None
    
    # 绘图
    fig = plt.figure(figsize=(12, 8))
    plt.suptitle('Part1 图像预处理')
    
    steps = [
        (resized, 'cv2.COLOR_BGR2RGB', f'尺寸标准化 {target_size[0]}X{target_size[1]}'),
        (masked, 'cv2.COLOR_BGR2RGB', '遮挡左下角文字区域'),
        (gray, None, '灰度图'),
        (gray_enhanced, None, 'clahe增强'),
        (binary, None, '自适应阈值二值化'),
        (closed, None, f'闭运算(6x6) + 削除右侧{right_strip_width}px')
    ]
    
    for i, (img, conversion, title) in enumerate(steps, 1):
        ax = plt.subplot(2, 3, i)
        
        if conversion:
            img = cv2.cvtColor(img, eval(conversion))
        
        ax.imshow(img, cmap='gray' if conversion is None else None)
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
    
    plt.tight_layout()
    plt.show()
    
    return resized, binary, closed, fig

#--------------------------------------------------------------------------------------------------------------------
def kmeans_clustering_regions(regions: List[Dict], image: np.ndarray, n_rows: int = 5, n_cols: int = 4) -> Tuple[List[float], List[float]]:
    """KMeans聚类分割区域"""
    h, w = image.shape[:2]
    
    if len(regions) < max(n_rows, n_cols):
        print(f"连通域数量({len(regions)})不足，无法进行聚类")
        return [0, h], [0, w]
    
    print(f"进行KMeans聚类: {n_rows}行 × {n_cols}列")
    
    points = np.array([[r['centroid'][0], r['centroid'][1]] for r in regions])
    x_coords = points[:, 0].reshape(-1, 1)
    y_coords = points[:, 1].reshape(-1, 1)
    
    # 对Y坐标聚类
    kmeans_y = KMeans(n_clusters=min(n_rows, len(y_coords)), 
                     random_state=42, n_init=10)
    row_labels_raw = kmeans_y.fit_predict(y_coords)
    
    # 计算每个Y聚类的中心并排序
    row_centers = {}
    for label in np.unique(row_labels_raw):
        row_centers[label] = y_coords[row_labels_raw == label].mean()
    
    sorted_row_labels = sorted(row_centers.items(), key=lambda x: x[1])
    row_mapping = {old_label: i for i, (old_label, _) in enumerate(sorted_row_labels)}
    
    # 对X坐标聚类
    kmeans_x = KMeans(n_clusters=min(n_cols, len(x_coords)), 
                     random_state=42, n_init=10)
    col_labels_raw = kmeans_x.fit_predict(x_coords)
    
    # 计算每个X聚类的中心并排序
    col_centers = {}
    for label in np.unique(col_labels_raw):
        col_centers[label] = x_coords[col_labels_raw == label].mean()
    
    sorted_col_labels = sorted(col_centers.items(), key=lambda x: x[1])
    col_mapping = {old_label: i for i, (old_label, _) in enumerate(sorted_col_labels)}
    
    # 为每个区域分配行列标签
    for i, region in enumerate(regions):
        region['row_raw'] = int(row_labels_raw[i])
        region['col_raw'] = int(col_labels_raw[i])
        region['row'] = row_mapping[region['row_raw']]
        region['col'] = col_mapping[region['col_raw']]
    
    # 计算网格边界
    row_bounds = {}
    col_bounds = {}
    
    for region in regions:
        row = region['row']
        col = region['col']
        cx, cy = region['centroid']
        
        if row not in row_bounds:
            row_bounds[row] = {'min_y': cy, 'max_y': cy}
        else:
            row_bounds[row]['min_y'] = min(row_bounds[row]['min_y'], cy)
            row_bounds[row]['max_y'] = max(row_bounds[row]['max_y'], cy)
            
        if col not in col_bounds:
            col_bounds[col] = {'min_x': cx, 'max_x': cx}
        else:
            col_bounds[col]['min_x'] = min(col_bounds[col]['min_x'], cx)
            col_bounds[col]['max_x'] = max(col_bounds[col]['max_x'], cx)
    
    # 计算行分界线
    horizontal_lines = []
    for r in range(n_rows - 1):
        if r in row_bounds and r+1 in row_bounds:
            y_bottom = row_bounds[r]['max_y']
            y_top = row_bounds[r+1]['min_y']
            boundary = (y_bottom + y_top) / 2
            horizontal_lines.append(boundary)
    
    # 计算列分界线
    vertical_lines = []
    for c in range(n_cols - 1):
        if c in col_bounds and c+1 in col_bounds:
            x_right = col_bounds[c]['max_x']
            x_left = col_bounds[c+1]['min_x']
            boundary = (x_right + x_left) / 2
            vertical_lines.append(boundary)
    
    # 添加图像边界
    all_horizontal = [0] + sorted(horizontal_lines) + [h]
    all_vertical = [0] + sorted(vertical_lines) + [w]
    
    return all_horizontal, all_vertical

def calculate_boundary_distances(region_points: List[Dict], x1: float, y1: float, x2: float, y2: float) -> Dict:
    """计算点到边界的距离"""
    if not region_points:
        return {
            'min_to_left': float('inf'),
            'min_to_right': float('inf'),
            'min_to_top': float('inf'),
            'min_to_bottom': float('inf'),
            'max_to_left': 0,
            'max_to_right': 0,
            'max_to_top': 0,
            'max_to_bottom': 0
        }
    
    distances = {
        'to_left': [],
        'to_right': [],
        'to_top': [],
        'to_bottom': []
    }
    
    for point in region_points:
        cx, cy = point['cx'], point['cy']
        distances['to_left'].append(cx - x1)
        distances['to_right'].append(x2 - cx)
        distances['to_top'].append(cy - y1)
        distances['to_bottom'].append(y2 - cy)
    
    return {
        'min_to_left': min(distances['to_left']) if distances['to_left'] else float('inf'),
        'min_to_right': min(distances['to_right']) if distances['to_right'] else float('inf'),
        'min_to_top': min(distances['to_top']) if distances['to_top'] else float('inf'),
        'min_to_bottom': min(distances['to_bottom']) if distances['to_bottom'] else float('inf'),
        'max_to_left': max(distances['to_left']) if distances['to_left'] else 0,
        'max_to_right': max(distances['to_right']) if distances['to_right'] else 0,
        'max_to_top': max(distances['to_top']) if distances['to_top'] else 0,
        'max_to_bottom': max(distances['to_bottom']) if distances['to_bottom'] else 0
    }

def adjust_region_boundary(x1: float, y1: float, x2: float, y2: float, 
                          dist_info: Dict, margin_threshold: float = 20.0,
                          min_expand: float = 5.0, max_expand: float = 30.0) -> Tuple[float, float, float, float]:
    """根据点到边界距离调整区域边界"""
    new_x1, new_y1, new_x2, new_y2 = x1, y1, x2, y2
    
    # 计算需要的扩展距离
    left_expand = 0
    right_expand = 0
    top_expand = 0
    bottom_expand = 0
    
    # 如果点到左边界的距离小于阈值，则向左扩展
    if dist_info['min_to_left'] < margin_threshold:
        left_expand = min(max_expand, max(min_expand, margin_threshold - dist_info['min_to_left']))
    
    # 如果点到右边界的距离小于阈值，则向右扩展
    if dist_info['min_to_right'] < margin_threshold:
        right_expand = min(max_expand, max(min_expand, margin_threshold - dist_info['min_to_right']))
    
    # 如果点到上边界的距离小于阈值，则向上扩展
    if dist_info['min_to_top'] < margin_threshold:
        top_expand = min(max_expand, max(min_expand, margin_threshold - dist_info['min_to_top']))
    
    # 如果点到下边界的距离小于阈值，则向下扩展
    if dist_info['min_to_bottom'] < margin_threshold:
        bottom_expand = min(max_expand, max(min_expand, margin_threshold - dist_info['min_to_bottom']))
    
    # 应用边界扩展
    new_x1 = new_x1 - left_expand
    new_x2 = new_x2 + right_expand
    new_y1 = new_y1 - top_expand
    new_y2 = new_y2 + bottom_expand
    
    # 记录调整信息
    adjustment_info = {
        'left_expand': left_expand,
        'right_expand': right_expand,
        'top_expand': top_expand,
        'bottom_expand': bottom_expand,
        'original_bounds': (x1, y1, x2, y2),
        'adjusted_bounds': (new_x1, new_y1, new_x2, new_y2)
    }
    
    return new_x1, new_y1, new_x2, new_y2, adjustment_info

def crop_regions(image: np.ndarray, regions: List[Dict], 
                 all_horizontal: List[float], all_vertical: List[float],
                 n_rows: int = 5, n_cols: int = 4, 
                 min_points_per_region: int = 5,
                 margin_threshold: float = 20.0,
                 target_aspect_ratio: float = 1.5, aspect_ratio_threshold: float = 1.0) -> List[Dict]:
    """区域裁剪 - 只处理有效区域（点数>=5），无效区域被过滤"""
    h, w = image.shape[:2]
    
    cropped_regions = []
    region_counter = 1
    
    for r in range(n_rows):
        for c in range(n_cols):
            y1, y2 = all_horizontal[r], all_horizontal[r + 1]
            x1, x2 = all_vertical[c], all_vertical[c + 1]
            
            # Step 1: 收集区域内的点
            region_points = []
            for region in regions:
                cx = region.get('centroid', region.get('center', (0, 0)))[0]
                cy = region.get('centroid', region.get('center', (0, 0)))[1]
                
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    region_points.append({'cx': cx, 'cy': cy})
            
            # 如果区域内点数小于阈值，跳过这个区域
            if len(region_points) < min_points_per_region:
                print(f"  区域({r},{c}): 无效区域，点数={len(region_points)} (<{min_points_per_region})，已跳过")
                continue
            
            print(f"  区域({r},{c}): 有效区域，点数={len(region_points)}")
            
            # Step 2: 计算点到边界的距离
            dist_info = calculate_boundary_distances(region_points, x1, y1, x2, y2)
            
            # Step 3: 根据距离调整边界
            original_x1, original_y1, original_x2, original_y2 = x1, y1, x2, y2
            adjusted = False
            adjustment_info = None
            
            new_x1, new_y1, new_x2, new_y2, adjustment_info = adjust_region_boundary(
                x1, y1, x2, y2, dist_info, margin_threshold
            )
            
            # 检查是否进行了调整
            if (new_x1 != x1 or new_x2 != x2 or new_y1 != y1 or new_y2 != y2):
                adjusted = True
                x1, y1, x2, y2 = new_x1, new_y1, new_x2, new_y2
                print(f"    边界调整 - 左:{adjustment_info['left_expand']:.1f}px, "
                      f"右:{adjustment_info['right_expand']:.1f}px, "
                      f"上:{adjustment_info['top_expand']:.1f}px, "
                      f"下:{adjustment_info['bottom_expand']:.1f}px")
            
            # Step 4: 宽高比调整（可选）
            width = int(x2 - x1)
            height = int(y2 - y1)
            aspect_ratio = width / max(height, 1)
            
            if aspect_ratio < aspect_ratio_threshold and height > 0:
                target_height = int(width / target_aspect_ratio)
                
                if 0 < target_height < height:
                    crop_top = sum(1 for p in region_points if y1 <= p['cy'] <= y1 + target_height)
                    crop_bottom = sum(1 for p in region_points if y2 - target_height <= p['cy'] <= y2)
                    
                    if crop_top >= crop_bottom:
                        y2 = y1 + target_height
                    else:
                        y1 = y2 - target_height
                    
                    height = int(y2 - y1)
                    aspect_ratio = width / height
            
            # Step 5: 确保边界在图像范围内
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            
            # 检查裁剪区域是否有效
            if x2 <= x1 or y2 <= y1:
                print(f"    警告: 裁剪区域无效 ({x1},{y1})-({x2},{y2})，跳过")
                continue
            
            # Step 6: 裁剪区域
            cropped = image[y1:y2, x1:x2]
            
            cropped_regions.append({
                'region_id': region_counter,
                'row': r,
                'col': c,
                'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2),
                'original_x1': int(original_x1), 'original_y1': int(original_y1),
                'original_x2': int(original_x2), 'original_y2': int(original_y2),
                'width': int(x2 - x1),
                'height': int(y2 - y1),
                'aspect_ratio': aspect_ratio,
                'point_count': len(region_points),
                'is_valid': True,
                'adjusted': adjusted,
                'adjustment_info': adjustment_info,
                'cropped': cropped
            })
            region_counter += 1
    
    print(f"\n区域裁剪完成: 总共{n_rows*n_cols}个区域, 保留{len(cropped_regions)}个有效区域")
    return cropped_regions

def get_grid_points(image: np.ndarray, binary_image: np.ndarray, show: bool = True) -> Tuple[List[Dict], List[Dict], plt.Figure]:
    """获取网格点位置"""
    # 检测连通域 + 面积筛选
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    
    regions = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if 80 <= area <= 400:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w_rect = stats[i, cv2.CC_STAT_WIDTH]
            h_rect = stats[i, cv2.CC_STAT_HEIGHT]
            centroid_x, centroid_y = centroids[i]
            
            regions.append({
                'label': i,
                'area': area,
                'centroid': (centroid_x, centroid_y),
                'x': x,
                'y': y,
                'width': w_rect,
                'height': h_rect,
            })
    
    # KMeans聚类(5行4列) -> 确定区域分界线
    h_lines, v_lines = kmeans_clustering_regions(regions, image, n_rows=5, n_cols=4)
    
    # 原图区域分割（只保留有效区域）
    cropped_regions = crop_regions(
        image, regions, h_lines, v_lines, 
        n_rows=5, n_cols=4, 
        min_points_per_region=5,  # 最少5个点才认为是有效区域
        margin_threshold=20.0,  # 边界距离阈值
        target_aspect_ratio=1.5, 
        aspect_ratio_threshold=1.0
    )
    
    if not show:
        return regions, cropped_regions, None
    
    # 绘图
    fig = plt.figure(figsize=(12, 8))
    plt.suptitle('Part2 填涂点位检测')
    
    # 连通域检测
    ax1 = plt.subplot(2, 2, 1)
    label_colors = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255)
    ]
    
    for idx, region in enumerate(regions):
        color_idx = idx % len(colors)
        color = colors[color_idx]
        label_colors[labels == region['label']] = color
    
    ax1.imshow(label_colors)
    ax1.set_title('连通域检测 (面积筛选后)')
    ax1.set_axis_off()
    
    # 连通域质心分布
    ax2 = plt.subplot(2, 2, 2)
    centroids_x = [r['centroid'][0] for r in regions]
    centroids_y = [r['centroid'][1] for r in regions]
    ax2.scatter(centroids_x, centroids_y, s=20, c='blue', alpha=0.6)
    ax2.set_title('连通域质心分布')
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()
    
    # KMeans聚类结果
    ax3 = plt.subplot(2, 2, 3)
    if regions and 'row' in regions[0] and 'col' in regions[0]:
        unique_rows = sorted(set(r['row'] for r in regions))
        unique_cols = sorted(set(r['col'] for r in regions))
        
        # 绘制行聚类结果
        for row_idx, row in enumerate(unique_rows):
            row_points = [r for r in regions if r['row'] == row]
            x_vals = [r['centroid'][0] for r in row_points]
            y_vals = [r['centroid'][1] for r in row_points]
            ax3.scatter(x_vals, y_vals, s=40, label=f'行{row}', alpha=0.7)
        
        # 绘制列聚类结果
        for col_idx, col in enumerate(unique_cols):
            col_points = [r for r in regions if r['col'] == col]
            x_vals = [r['centroid'][0] for r in col_points]
            y_vals = [r['centroid'][1] for r in col_points]
            ax3.scatter(x_vals, y_vals, s=20, marker='s', label=f'列{col}', alpha=0.7)
        
        # 绘制分割线
        for y in h_lines[1:-1]:
            ax3.axhline(y=y, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        for x in v_lines[1:-1]:
            ax3.axvline(x=x, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax3.set_title('KMeans聚类->确定区域分界线')
    ax3.grid(True, alpha=0.3)
    ax3.invert_yaxis()
    
    # 原图区域划分
    ax4 = plt.subplot(2, 2, 4)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax4.imshow(img_rgb)

    # 只绘制有效区域（调整后）
    for region in cropped_regions:
        x1, y1, x2, y2 = region['x1'], region['y1'], region['x2'], region['y2']

        # 绘制调整后边界（实线）
        rect = plt.Rectangle(
            (x1, y1), 
            x2 - x1, 
            y2 - y1,
            linewidth=2, 
            edgecolor='red', 
            linestyle='--',
            facecolor='none', 
            alpha=0.8
        )
        ax4.add_patch(rect)

        # 显示区域ID
        ax4.text(
            x1 + (x2 - x1) / 2, 
            y1 + (y2 - y1) / 2, 
            f"R{region['region_id']}", 
            ha='center', 
            va='center', 
            fontsize=12, 
            color='blue', 
            weight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

    ax4.set_title('区域划分最终结果(动态扩展边界)')
    ax4.set_axis_off()
    
    plt.tight_layout()
    plt.show()
    
    return regions, cropped_regions, fig
#--------------------------------------------------------------------------------------------------------------------
# Part3: 检测点位填涂情况
def process_single_region(region_img: np.ndarray, region_id: int, is_plot: bool = True, show: bool = False) -> Tuple[Dict, plt.Figure]:
    """处理单个区域"""
    # Step1: 转换为灰度
    gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)
    
    # Step2: CLAHE增强
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_result = clahe.apply(gray)
    
    # Step3: 自适应二值化
    binary = cv2.adaptiveThreshold(clahe_result, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 10)
    
    # Step4: 闭运算
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 2))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Step5: 边缘置黑
    border_width = 10
    h, w = closed.shape
    if border_width > 0:
        closed[0:border_width, :] = 0
        closed[h-border_width:h, :] = 0
        closed[:, 0:border_width] = 0
        closed[:, w-border_width:w] = 0
    
    # Step6: 轮廓检测+尺寸过滤
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    points = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w_rect, h_rect = cv2.boundingRect(cnt)
        aspect_ratio = w_rect / h_rect if h_rect > 0 else 0
        
        if (20 <= area <= 300 and
            0.5 <= aspect_ratio <= 6.0):
               
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx = x + w_rect // 2
                cy = y + h_rect // 2
            
            points.append({
                'x': x, 'y': y, 'w': w_rect, 'h': h_rect,
                'cx': cx, 'cy': cy, 
                'area': area,
            })    
    df_points = pd.DataFrame(points)

    # Step7: KMeans行列聚类
    df_clustered = independent_row_col_clustering(df_points.copy(), rows=5, cols=5, check=True)

    # Step8: 过滤离群点和重复点  
    df_cleaned, df_peripheral_removed, df_duplicates = remove_duplicated_and_outliers(df_clustered, interval_multiplier=1.5)
    
    # Step9: 插值填补缺失点
    df_grid, df_interpolated = grid_linear_interpoltate(df_cleaned, n_rows=5, n_cols=5)
     
    # Step10: 腐蚀二值化图
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  
    eroded = cv2.erode(binary, kernel_erode)

    # Step11: 检测各点位填涂情况
    roi_w, roi_h = 22, 14
    fill_threshold = 0.3
    answer, df_result = detect_answers(region_id, df_grid, eroded, roi_size=(roi_w, roi_h), fill_threshold=fill_threshold)
    
    base_question_num = (region_id - 1) * 5 + 1

    fig = None
    if is_plot:
        fig = plt.figure(figsize=(15, 10))
        plt.suptitle(f"区域{region_id} 处理流程")
        
        # 子图1: 原图
        ax1 = plt.subplot(3, 3, 1)
        plt.imshow(cv2.cvtColor(region_img, cv2.COLOR_BGR2RGB))
        ax1.set_title(f'原图')
        ax1.set_axis_off()
        
        # 子图2: 二值化
        ax2 = plt.subplot(3, 3, 2)
        plt.imshow(binary, cmap='gray')
        ax2.set_title(f'灰度->Clahe图像增强->自适应二值化')
        ax2.set_axis_off()
        
        # 子图3: 闭运算
        ax3 = plt.subplot(3, 3, 3)
        plt.imshow(closed, cmap='gray')
        ax3.set_title(f'闭运算+边缘置黑')
        ax3.set_axis_off()   
        
        # 子图4: 轮廓检测
        ax4 = plt.subplot(3, 3, 4)
        contour_img = cv2.cvtColor(region_img.copy(), cv2.COLOR_BGR2RGB) 
        for _, point in df_points.iterrows():
            cv2.rectangle(contour_img, (int(point['x']), int(point['y'])), 
                          (int(point['x']+point['w']), int(point['y']+point['h'])), 
                          (255, 0, 0), 2)
        ax4.imshow(contour_img)
        ax4.set_title(f'轮廓检测(尺寸筛选后)')
        ax4.set_axis_off()
        
        # 子图5: 去偏去重
        ax5 = plt.subplot(3, 3, 5)
        ax5.set_xlim(0, region_img.shape[1])
        ax5.set_ylim(region_img.shape[0], 0)
        ax5.set_aspect('equal')
        for idx, row in df_duplicates.iterrows():
            ax5.plot(row['cx'], row['cy'], 'o', 
                    color='orange', markersize=3, alpha=0.6, markerfacecolor='orange')
        for idx, row in df_peripheral_removed.iterrows():
            ax5.plot(row['cx'], row['cy'], 'x', 
                    color='gray', markersize=5, alpha=0.8, mew=1.5)
        for idx, row in df_cleaned.iterrows():
            ax5.plot(row['cx'], row['cy'], 'ro', markersize=3, alpha=0.8)
        ax5.set_title('KMeans聚类 + 去偏去重\n(红=保留, 橙=重复, 灰X=滤除)')
        ax5.grid(True, alpha=0.3)
        
        # 子图6: 插值
        ax6 = plt.subplot(3, 3, 6)
        ax6.set_xlim(0, region_img.shape[1])
        ax6.set_ylim(region_img.shape[0], 0)
        ax6.set_aspect('equal')
        for idx, row in df_interpolated.iterrows():
            ax6.plot(row['cx'], row['cy'], '^', 
                    color='green', markersize=3, alpha=0.6, markerfacecolor='green')
        for idx, row in df_cleaned.iterrows():
            ax6.plot(row['cx'], row['cy'], 'ro', markersize=3, alpha=0.8)
        ax6.set_title('插值\n(红=保留, 绿=插值)')
        ax6.grid(True, alpha=0.3)
        
        # 子图7: 腐蚀图+ROI检测框
        ax7 = plt.subplot(3, 3, 7)
        if len(eroded.shape) == 2:
            eroded_color = cv2.cvtColor(eroded, cv2.COLOR_GRAY2BGR)
        else:
            eroded_color = eroded.copy()
        
        for idx, row in df_result.iterrows():
            x1 = int(row['cx'] - roi_w // 2)
            y1 = int(row['cy'] - roi_h // 2)
            x2 = int(row['cx'] + roi_w // 2)
            y2 = int(row['cy'] + roi_h // 2)
            
            h_img, w_img = eroded_color.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)
            
            if x2 > x1 and y2 > y1:
                cv2.rectangle(eroded_color, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
        ax7.imshow(cv2.cvtColor(eroded_color, cv2.COLOR_BGR2RGB))
        ax7.set_title(f'腐蚀图+ROI检测框\n(ROI大小: {roi_w}×{roi_h})')
        ax7.set_axis_off()
        
        # 子图8: 原图标注填涂结果
        ax8 = plt.subplot(3, 3, 8)
        region_rgb = cv2.cvtColor(region_img, cv2.COLOR_BGR2RGB)
        ax8.imshow(region_rgb)
        
        answer_display = ' '.join([ans if ans != '/' else '/' for ans in answer])
        ax8.set_title(f'检测答案: {answer_display}')
        
        for idx, row in df_result.iterrows():
            row_id = int(row['row_id'])
            col_id = int(row['col_id'])
            
            x1 = int(row['cx'] - roi_w // 2)
            y1 = int(row['cy'] - roi_h // 2)
            x2 = int(row['cx'] + roi_w // 2)
            y2 = int(row['cy'] + roi_h // 2)
            
            h_img, w_img = region_img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)
            
            if row_id == 0:
                rect = plt.Rectangle((x1, y1), roi_w, roi_h, 
                                    fill=False, edgecolor='orange', 
                                    linewidth=1.5, alpha=0.8)
                
                if 'question_num' in row and not pd.isna(row['question_num']):
                    question_num = int(row['question_num'])
                    ax8.text(row['cx'], y1 - 5, f'Q{question_num}', 
                            color='orange', fontsize=11, ha='center', va='bottom', 
                            fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", 
                                                        facecolor='white', alpha=0.7))
            elif 1 <= row_id <= 4:
                if row['is_filled']:
                    color = 'green'
                    fill_face = False
                else:
                    color = 'red'
                    fill_face = False
                
                rect = plt.Rectangle((x1, y1), roi_w, roi_h, 
                                    fill=fill_face, edgecolor=color, 
                                    facecolor=color if fill_face else 'none',
                                    linewidth=2, alpha=0.8)
            
            ax8.add_patch(rect)
        
        ax8.set_axis_off()
        
        # 子图9: 5×5填涂比例热力图
        ax9 = plt.subplot(3, 3, 9)
        fill_matrix = np.zeros((5, 5))
        for idx, row in df_result.iterrows():
            row_id = int(row['row_id'])
            col_id = int(row['col_id'])
            if 0 <= row_id < 5 and 0 <= col_id < 5:
                fill_matrix[row_id, col_id] = row['fill_ratio']
        
        im = ax9.imshow(fill_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
        cbar = plt.colorbar(im, ax=ax9, fraction=0.046, pad=0.04)
        cbar.set_label('填涂比例', fontsize=9)
        
        ax9.set_xticks(range(5))
        ax9.set_yticks(range(5))
        
        col_labels = []
        for col in range(5):
            question_num = base_question_num + col
            col_labels.append(str(question_num))
        
        row_labels = ['题号', 'A', 'B', 'C', 'D']
        ax9.set_xticklabels(col_labels, fontsize=9)
        ax9.set_yticklabels(row_labels, fontsize=9)
        ax9.set_xlabel('题号', fontsize=10)
        ax9.set_ylabel('选项', fontsize=10)
        
        for i in range(5):
            for j in range(5):
                value = fill_matrix[i, j]
                text_color = 'black'
                text = f'{value:.2f}'
                fontweight = 'bold' if value > fill_threshold else 'normal'
                
                ax9.text(j, i, text, 
                        ha='center', va='center', 
                        color=text_color, fontsize=8, fontweight=fontweight)
                
                if value > fill_threshold:
                    rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, 
                                        fill=False, edgecolor='black', 
                                        linewidth=2, alpha=0.8)
                    ax9.add_patch(rect)
        
        ax9.set_xticks(np.arange(-0.5, 4.5, 1), minor=True)
        ax9.set_yticks(np.arange(-0.5, 4.5, 1), minor=True)
        ax9.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)
        ax9.tick_params(which="minor", size=0)
        
        ax9.set_title(f'填涂比例热力图(阈值={fill_threshold})', fontsize=11)
    
        if show:
            plt.show()
        else:
            plt.close(fig)       
    
    return {
        'region_id': region_id,
        'answer': answer,
        'df_result': df_result,
        'base_question_num': base_question_num
    }, fig

def detect_answers(region_id: int, df_grid: pd.DataFrame, image: np.ndarray, 
                   roi_size: Tuple[int, int], fill_threshold: float) -> Tuple[List[str], pd.DataFrame]:
    """检测选择题填涂情况"""
    df = df_grid.copy()
    if df.empty or len(df) != 25:
        print(f"警告: 区域{region_id}网格点数量异常 ({len(df)}个点)")
        return [], df
    
    base_question_num = (region_id - 1) * 5 + 1
    print(f"区域{region_id}: 对应题号 {base_question_num}~{base_question_num+4}")
    
    # 初始化列
    df['fill_ratio'] = 0.0
    df['is_filled'] = False
    df['is_question_row'] = False
    df['question_num'] = 0
    df['option_char'] = ''
    df['option_name'] = ''
    df['is_valid_option'] = False
    
    roi_w, roi_h = roi_size
    threshold = fill_threshold
    
    # 检测所有点的填涂情况
    for idx, row in df.iterrows():
        x1 = int(row['cx'] - roi_w // 2)
        y1 = int(row['cy'] - roi_h // 2)
        x2 = int(row['cx'] + roi_w // 2)
        y2 = int(row['cy'] + roi_h // 2)
        
        h, w = image.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 > x1 and y2 > y1:
            roi = image[y1:y2, x1:x2]
            if roi.size > 0:
                total_pixels = roi.size
                white_pixels = cv2.countNonZero(roi)
                fill_ratio = white_pixels / total_pixels
                
                df.at[idx, 'fill_ratio'] = fill_ratio
                df.at[idx, 'is_filled'] = fill_ratio > threshold
    
    # 识别答案
    answers = _recognize_answers(df, region_id, base_question_num)
    
    return answers, df

def _recognize_answers(df: pd.DataFrame, region_id: int, base_question_num: int) -> List[str]:
    """从填涂结果识别选择题答案"""
    if 'row_id' not in df.columns or 'col_id' not in df.columns:
        return []
    
    row_to_option = {0: '', 1: 'A', 2: 'B', 3: 'C', 4: 'D'}
    
    answers = []
    cols = sorted(df['col_id'].unique())
    if len(cols) != 5:
        print(f"警告: 区域{region_id}列数异常 ({len(cols)}列)")
        cols = list(range(5))
    
    for col in cols:
        if col < 0 or col >= 5:
            continue
            
        question_num = base_question_num + col
        col_points = df[df['col_id'] == col]
        
        # 标记题号行
        for idx, row in col_points.iterrows():
            if row['row_id'] == 0:
                df.at[idx, 'is_question_row'] = True
                df.at[idx, 'question_num'] = question_num
                df.at[idx, 'option_name'] = f'题号{question_num}'
        
        # 获取选项行
        option_points = col_points[col_points['row_id'].between(1, 4)]
        filled_options = []
        
        for idx, row in option_points.iterrows():
            row_id = int(row['row_id'])
            option_char = row_to_option.get(row_id, '')
            
            df.at[idx, 'is_valid_option'] = True
            df.at[idx, 'question_num'] = question_num
            df.at[idx, 'option_char'] = option_char
            df.at[idx, 'option_name'] = f'{question_num}{option_char}'
            
            if row['is_filled'] and option_char:
                filled_options.append(option_char)
        
        filled_options.sort()
        answer_str = ''.join(filled_options) if filled_options else '/'
        answers.append(answer_str)
        
        if filled_options:
            fill_ratios = []
            for i in range(1, 5):
                points = option_points[option_points['row_id']==i]
                if len(points) > 0:
                    fill_ratios.append(points['fill_ratio'].values[0])
                else:
                    fill_ratios.append(0)
            print(f"  题{question_num}: 答案 {answer_str} (填涂比例: {fill_ratios})")
    
    return answers

def process_all_regions(image: np.ndarray, cropped_regions: List[Dict], 
                       region_plot: Any = None, region_plot_show: bool = False, 
                       show: bool = True) -> Tuple[Dict, Dict, plt.Figure, plt.Figure]:
    """处理所有区域 - 只处理有效区域"""
    region_results = []
    region_plots = {}
    
    # 处理region_plot参数
    if region_plot is None:
        plot_mode = 'none'
    elif region_plot == 'All':
        plot_mode = 'all'
    else:
        plot_mode = 'optional'
        region_plot_list = region_plot if isinstance(region_plot, list) else []
    
    for i, region in enumerate(cropped_regions, 1):
        print(f"\n正在处理区域 {i} (原始位置: 行{region['row']}, 列{region['col']})...")
        
        region_img = region['cropped'].copy()
        
        if plot_mode == 'none':
            is_plot = False
            result, fig = process_single_region(region_img, i, is_plot=is_plot, show=False)
        elif plot_mode == 'optional':
            is_plot = (i in region_plot_list)
            result, fig = process_single_region(region_img, i, is_plot=is_plot, show=(region_plot_show and is_plot))
        else:
            is_plot = True
            result, fig = process_single_region(region_img, i, is_plot=is_plot, show=region_plot_show)
        
        if is_plot and fig is not None:
            region_plots[i] = fig
        region_results.append(result)
    
    # 统计信息
    all_answers = {}
    
    for region_idx, (region, result) in enumerate(zip(cropped_regions, region_results)):
        region_id = region['region_id']
        
        if 'answer' in result:
            answers = result['answer']
            base_num = result.get('base_question_num', (region_id-1)*5 + 1)
            
            for i, ans in enumerate(answers):
                question_num = base_num + i
                all_answers[question_num] = ans
    
    # 第一张图：全局答题情况
    fig1 = None
    if show:
        fig1 = _plot_global_result(image, cropped_regions, region_results)
    
    # 第二张图：答题情况表格
    fig2 = None
    if show:
        fig2 = _plot_answer_table(all_answers)
        
    return all_answers, region_plots, fig1, fig2

def _plot_global_result(image: np.ndarray, cropped_regions: List[Dict], region_results: List[Dict]) -> plt.Figure:
    """绘制全局答题情况图"""
    fig = plt.figure(figsize=(18, 12))
    ax = plt.subplot(111)
    
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    question_texts = []
    
    for region_idx, (region, result) in enumerate(zip(cropped_regions, region_results)):
        region_id = region['region_id']
        
        if 'df_result' in result and not result['df_result'].empty:
            df_result = result['df_result']
            
            for idx, row in df_result.iterrows():
                if 'row_id' not in row or 'col_id' not in row:
                    continue
                    
                row_id = int(row['row_id'])
                col_id = int(row['col_id'])
                
                point_cx = int(region['x1'] + row['cx'])
                point_cy = int(region['y1'] + row['cy'])
                
                roi_w, roi_h = 24, 16
                x1_rect = point_cx - roi_w // 2
                y1_rect = point_cy - roi_h // 2
                x2_rect = point_cx + roi_w // 2
                y2_rect = point_cy + roi_h // 2
                
                h_img, w_img = image.shape[:2]
                x1_rect = max(0, x1_rect)
                y1_rect = max(0, y1_rect)
                x2_rect = min(w_img, x2_rect)
                y2_rect = min(h_img, y2_rect)
                
                if row_id == 0:
                    if 'question_num' in row and not pd.isna(row['question_num']):
                        question_num = int(row['question_num'])
                        ax.text(point_cx, y1_rect + roi_h - 4, f'{question_num}', 
                                color='orange', fontsize=14, ha='center', va='bottom', 
                                fontweight='bold', 
                                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9))
                        question_texts.append((point_cx, y1_rect + roi_h + 8, question_num))
                
                elif 1 <= row_id <= 4:
                    if 'is_filled' in row and row['is_filled']:
                        color = '#00FF00'
                        linewidth = 2
                        alpha = 0.8
                    else:
                        color = '#FF0000'
                        linewidth = 1.5
                        alpha = 0.6
                    
                    if x2_rect > x1_rect and y2_rect > y1_rect:
                        rect = plt.Rectangle((x1_rect, y1_rect), 
                                            x2_rect-x1_rect, y2_rect-y1_rect, 
                                            fill=False, edgecolor=color, 
                                            linewidth=linewidth, alpha=alpha)
                        ax.add_patch(rect)
    
    ax.set_axis_off()
    ax.set_title('答题卡全局填涂检测结果 (只显示有效区域)', fontsize=16, pad=20)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='white', edgecolor='#00FF00', linewidth=2, label='已填涂'),
        Patch(facecolor='white', edgecolor='#FF0000', linewidth=1.5, label='未填涂'),
        Patch(facecolor='white', edgecolor='orange', linewidth=2, label='题号位置')
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=10, 
               bbox_to_anchor=(1.0, 1.0), framealpha=0.9)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def _plot_answer_table(all_answers: Dict[int, str]) -> plt.Figure:
    """绘制答题情况表格"""
    fig = plt.figure(figsize=(16, 6))
    ax = plt.subplot(111)
    ax.axis('off')
    ax.set_title('答题情况汇总表 (只显示有效区域)', fontsize=16, pad=50)
    
    all_answers_list = ['/'] * 85
    for q_num, ans in all_answers.items():
        if 1 <= q_num <= 85:
            all_answers_list[q_num-1] = ans
    
    table_rows = 5
    table_cols = 17
    question_numbers = np.arange(1, 86).reshape(table_rows, table_cols)
    answer_matrix = np.array(all_answers_list).reshape(table_rows, table_cols)
    
    table_data = []
    for row in range(table_rows):
        question_row = [f"{question_numbers[row, col]:3d}" for col in range(table_cols)]
        table_data.append(question_row)
        answer_row = [answer_matrix[row, col] for col in range(table_cols)]
        table_data.append(answer_row)
    
    table = ax.table(cellText=table_data, 
                     loc='center', 
                     cellLoc='center',
                     colWidths=[0.06] * table_cols)
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)
    
    for i in range(len(table_data)):
        for j in range(table_cols):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_text_props(weight='bold', color='navy')
                cell.set_facecolor('#F0F8FF')
                cell.set_height(0.12)
            else:
                answer = table_data[i][j]
                if answer != '/':
                    cell.set_text_props(weight='bold', color='darkgreen')
                    cell.set_facecolor('#F0FFF0')
                else:
                    cell.set_text_props(color='darkred')
                    cell.set_facecolor('#FFF0F0')
                cell.set_height(0.12)
    
    plt.tight_layout()
    plt.show()
    
    return fig

#--------------------------------------------------------------------------------------------------------------------
# 主程序: 完整答案识别程序
def recognize_answers(answer_region, show=True):
    """识别答题卡答案"""
    
    # Part1 图像预处理
    resized, binary, closed, fig1 = preprocess_image(answer_region, show)
    
    # Part2 识别填涂点位
    regions, cropped_regions, fig2 = get_grid_points(resized, closed, show)
    
    # Part3 确认填涂情况，识别答案
    all_answers, region_plots, fig3, fig4 = process_all_regions(
        resized, cropped_regions, 
        region_plot=[3], # 'All'：绘制所有区域处理过程图  (e.x.)[1, 3]: 只绘制指定区域处理过程图  None：不绘制
        region_plot_show=True,  # 区域处理图是否显示
        show=show
    )
    
    return all_answers, [fig1, fig2, fig3, fig4], region_plots

if __name__ == '__main__':
    image_path = '../images/test15.jpg'
    ans_raw, _ = process_answer_sheet(image_path, show=False)
    all_answers, all_figures, region_plots = recognize_answers(ans_raw, show=True) 
    
    print("\n" + "="*50)
    print("答题卡识别结果 (只包含有效区域)")
    print("="*50)
    
    for question_num, answer in sorted(all_answers.items()):
        print(f"  题{question_num:2d}: {answer}")