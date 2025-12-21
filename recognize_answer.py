"""
Step3: 答题区填涂识别
Author: 蔡奕辉
Date: 2025-12-21
Description: 完整的答题区识别流程，包括预处理、区域分割、点检测、填涂识别和结果可视化
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from find_region import process_answer_sheet
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import matplotlib
from typing import Dict, Any, List, Tuple, Optional

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 参数配置 ====================
MAIN_PARAMS = {
    'target_size': (820, 680),  # 答题卡标准尺寸
    'clahe_clip': 3.0,          # CLAHE
    'clahe_grid': (8, 8),       
    'adaptive_block': 31,       # 自适应二值化
    'adaptive_c': 15,           
    'morph_kernel': (6, 6),     # 闭运算核大小
    'mask_bottom_ratio': 0.38,  # 底部掩码比例
    'mask_right_ratio': 0.24,   # 右侧掩码比例
    'right_strip_width': 20,    # 最右侧归零宽度
    
    # 连通域参数
    'min_area': 80,     # 最小连通域面积
    'max_area': 400,    # 最大连通域面积
    
    # KMeans聚类参数
    'n_rows': 5,        # 行数（自上而下）
    'n_cols': 4,        # 列数（自左向右）
    
    # 区域裁剪参数
    'min_points_per_region': 5,  # 区域最小点数阈值
    'border_expand': 10,         # 边界扩展像素数
    'target_aspect_ratio': 1.5,  # 目标宽高比
    'aspect_ratio_threshold': 1.0,  # 宽高比阈值
}

REGION_PARAMS = {
    'clahe_clip': 2.0,
    'clahe_grid': (8, 8),
    'adaptive_block': 31,
    'adaptive_c': 10,
    'close_kernel': (4, 2),  # 闭运算核
    'border_width': 10,  # 边缘置黑宽度
    'min_area': 20,
    'max_area': 300,
    'min_aspect_ratio': 0.5,  # 最小宽高比
    'max_aspect_ratio': 6.0,  # 最大宽高比
    
    # KMeans聚类参数
    'n_rows': 5,  # 行数
    'n_cols': 5,  # 列数
    
    # 可疑区域阈值
    'suspicious_point_threshold': 20,  # 点数小于20标记为可疑
    
    # 填涂检测参数
    'erode_kernel': (2, 2),  # 腐蚀核大小
    'roi_size': (24, 16),  # ROI大小 (宽, 高)
    'fill_threshold': 0.4,  # 填涂阈值
}

# ==================== 主处理函数 ====================
def preprocess_image(img_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """读取和预处理图片"""
    print(f"图片路径: {img_path}")
    img = cv2.imread(img_path)
    print(f"原始图片: {img.shape}")
    
    # 提取答题区域
    try:
        res = process_answer_sheet(img_path, show=False)
        sheet_warped = res[0]
        print("✓ 提取答题区域")
    except:
        sheet_warped = img
        print("⚠ 使用原始图片")
    
    # 缩放
    resized = cv2.resize(sheet_warped, MAIN_PARAMS['target_size'])
    
    # 应用左下角掩码
    h, w = resized.shape[:2]
    masked = resized.copy()
    mask_bottom = int(h * MAIN_PARAMS['mask_bottom_ratio'])
    mask_right = int(w * MAIN_PARAMS['mask_right_ratio'])
    masked[mask_bottom:, :mask_right] = 255
    
    # 灰度转换
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    
    # CLAHE增强
    clahe = cv2.createCLAHE(clipLimit=MAIN_PARAMS['clahe_clip'], 
                           tileGridSize=MAIN_PARAMS['clahe_grid'])
    gray_enhanced = clahe.apply(gray)
    
    # 二值化
    binary = cv2.adaptiveThreshold(gray_enhanced, 255, 
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 
                                  MAIN_PARAMS['adaptive_block'], 
                                  MAIN_PARAMS['adaptive_c'])
    
    # 闭运算
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MAIN_PARAMS['morph_kernel'])
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 最右侧归零
    closed[:, -MAIN_PARAMS['right_strip_width']:] = 0
    print(f"右侧{MAIN_PARAMS['right_strip_width']}px归零")
    
    return resized, masked, gray, gray_enhanced, binary, closed

def analyze_connected_components(closed: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
    """分析连通域"""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
    
    regions = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if MAIN_PARAMS['min_area'] <= area <= MAIN_PARAMS['max_area']:
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
    
    print(f"连通域: {len(regions)}个")
    return labels, regions

def kmeans_clustering_regions(regions: List[Dict], resized: np.ndarray) -> Tuple[List[float], List[float]]:
    """KMeans聚类分割区域"""
    h, w = resized.shape[:2]
    
    if len(regions) >= max(MAIN_PARAMS['n_rows'], MAIN_PARAMS['n_cols']):
        print(f"进行KMeans聚类: {MAIN_PARAMS['n_rows']}行 × {MAIN_PARAMS['n_cols']}列")
        
        points = np.array([[r['centroid'][0], r['centroid'][1]] for r in regions])
        x_coords = points[:, 0].reshape(-1, 1)
        y_coords = points[:, 1].reshape(-1, 1)
        
        # 对Y坐标聚类
        kmeans_y = KMeans(n_clusters=min(MAIN_PARAMS['n_rows'], len(y_coords)), 
                         random_state=42, n_init=10)
        row_labels_raw = kmeans_y.fit_predict(y_coords)
        
        # 计算每个Y聚类的中心并排序
        row_centers = {}
        for label in np.unique(row_labels_raw):
            row_centers[label] = y_coords[row_labels_raw == label].mean()
        
        sorted_row_labels = sorted(row_centers.items(), key=lambda x: x[1])
        row_mapping = {old_label: i for i, (old_label, _) in enumerate(sorted_row_labels)}
        
        # 对X坐标聚类
        kmeans_x = KMeans(n_clusters=min(MAIN_PARAMS['n_cols'], len(x_coords)), 
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
        for r in range(MAIN_PARAMS['n_rows'] - 1):
            if r in row_bounds and r+1 in row_bounds:
                y_bottom = row_bounds[r]['max_y']
                y_top = row_bounds[r+1]['min_y']
                boundary = (y_bottom + y_top) / 2
                horizontal_lines.append(boundary)
        
        # 计算列分界线
        vertical_lines = []
        for c in range(MAIN_PARAMS['n_cols'] - 1):
            if c in col_bounds and c+1 in col_bounds:
                x_right = col_bounds[c]['max_x']
                x_left = col_bounds[c+1]['min_x']
                boundary = (x_right + x_left) / 2
                vertical_lines.append(boundary)
        
        # 添加图像边界
        all_horizontal = [0] + sorted(horizontal_lines) + [h]
        all_vertical = [0] + sorted(vertical_lines) + [w]
        
    else:
        print(f"连通域数量({len(regions)})不足，无法进行聚类")
        all_horizontal = [0, h]
        all_vertical = [0, w]
    
    return all_horizontal, all_vertical

def crop_regions(resized: np.ndarray, all_horizontal: List[float], all_vertical: List[float], 
                regions: List[Dict]) -> List[Dict]:
    """根据分界线裁剪区域"""
    h, w = resized.shape[:2]
    expand = MAIN_PARAMS['border_expand']
    
    # 计算扩展后的分界线
    expanded_horizontal = []
    expanded_vertical = []
    
    for i, y in enumerate(all_horizontal):
        if i == 0:
            expanded_y = max(0, y - expand)
        elif i == len(all_horizontal) - 1:
            expanded_y = min(h, y + expand)
        else:
            expanded_y = y
        expanded_horizontal.append(expanded_y)
    
    for i, x in enumerate(all_vertical):
        if i == 0:
            expanded_x = max(0, x - expand)
        elif i == len(all_vertical) - 1:
            expanded_x = min(w, x + expand)
        else:
            expanded_x = x
        expanded_vertical.append(expanded_x)
    
    # 裁剪区域
    cropped_regions = []
    region_counter = 1
    
    for r in range(MAIN_PARAMS['n_rows']):
        for c in range(MAIN_PARAMS['n_cols']):
            y1 = expanded_horizontal[r]
            y2 = expanded_horizontal[r + 1]
            x1 = expanded_vertical[c]
            x2 = expanded_vertical[c + 1]
            
            # 计算该区域的点数
            point_count = 0
            for region in regions:
                cx, cy = region['centroid']
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    point_count += 1
            
            # 只保留点数足够的区域
            if point_count >= MAIN_PARAMS['min_points_per_region']:
                cropped = resized[int(y1):int(y2), int(x1):int(x2)]
                cropped_regions.append({
                    'region_id': region_counter,
                    'row': r,
                    'col': c,
                    'x1': int(x1),
                    'y1': int(y1),
                    'x2': int(x2),
                    'y2': int(y2),
                    'width': int(x2 - x1),
                    'height': int(y2 - y1),
                    'point_count': point_count,
                    'cropped': cropped
                })
                region_counter += 1
            else:
                print(f"忽略区域({r},{c}): 点数不足 ({point_count} < {MAIN_PARAMS['min_points_per_region']})")
    
    print(f"有效区域数: {len(cropped_regions)}")
    return cropped_regions

def cluster_row_col(centers: List[Tuple[float, float]], n_rows: int = 5, n_cols: int = 5) -> List[Dict]:
    """对点进行行列聚类"""
    if len(centers) < max(n_rows, n_cols):
        return []
    
    points = np.array(centers)
    x_coords = points[:, 0].reshape(-1, 1)
    y_coords = points[:, 1].reshape(-1, 1)
    
    # 对y坐标聚类得到行标签
    kmeans_y = KMeans(n_clusters=min(n_rows, len(y_coords)), random_state=42, n_init=10)
    y_labels = kmeans_y.fit_predict(y_coords)
    
    # 计算每个y聚类的中心并排序
    y_centers = {}
    for label in np.unique(y_labels):
        y_centers[label] = y_coords[y_labels == label].mean()
    
    sorted_y_labels = sorted(y_centers.items(), key=lambda x: x[1])
    row_mapping = {old_label: i for i, (old_label, _) in enumerate(sorted_y_labels)}
    
    # 对x坐标聚类得到列标签
    kmeans_x = KMeans(n_clusters=min(n_cols, len(x_coords)), random_state=42, n_init=10)
    x_labels = kmeans_x.fit_predict(x_coords)
    
    # 计算每个x聚类的中心并排序
    x_centers = {}
    for label in np.unique(x_labels):
        x_centers[label] = x_coords[x_labels == label].mean()
    
    sorted_x_labels = sorted(x_centers.items(), key=lambda x: x[1])
    col_mapping = {old_label: i for i, (old_label, _) in enumerate(sorted_x_labels)}
    
    # 为每个点分配行列标签
    clustered_points = []
    for i, (x, y) in enumerate(points):
        clustered_points.append({
            'row': row_mapping[y_labels[i]],
            'col': col_mapping[x_labels[i]],
            'x': x,
            'y': y,
            'original_index': i
        })
    
    return clustered_points

def detect_and_remove_outliers(points: List[Dict], n_rows: int = 5, n_cols: int = 5, 
                              row_thresh: float = 8.0, col_thresh: float = 8.0) -> Tuple[List[Dict], List[Dict]]:
    """检测和剔除离群点"""
    if len(points) == 0:
        return points, []
    
    # 按行列分组
    grid_dict = {}
    for point in points:
        key = (point['row'], point['col'])
        if key not in grid_dict:
            grid_dict[key] = []
        grid_dict[key].append(point)
    
    # 找出无争议点
    uncontested_points = []
    for key, point_list in grid_dict.items():
        if len(point_list) == 1:
            uncontested_points.append(point_list[0])
    
    if len(uncontested_points) < 3:
        return points, []
    
    # 计算每行的y坐标中位数
    rows_dict = {}
    for point in uncontested_points:
        row = point['row']
        if row not in rows_dict:
            rows_dict[row] = []
        rows_dict[row].append(point)
    
    row_medians = {}
    for row, row_points in rows_dict.items():
        if len(row_points) >= 2:
            y_values = [p['y'] for p in row_points]
            row_medians[row] = np.median(y_values)
    
    # 计算每列的x坐标中位数
    cols_dict = {}
    for point in uncontested_points:
        col = point['col']
        if col not in cols_dict:
            cols_dict[col] = []
        cols_dict[col].append(point)
    
    col_medians = {}
    for col, col_points in cols_dict.items():
        if len(col_points) >= 2:
            x_values = [p['x'] for p in col_points]
            col_medians[col] = np.median(x_values)
    
    # 检测并剔除离群点
    kept_points = []
    outlier_points = []
    
    for point in points:
        row = point['row']
        col = point['col']
        x, y = point['x'], point['y']
        is_outlier = False
        
        # 只检查无争议点
        if len(grid_dict[(row, col)]) == 1:
            if row in row_medians:
                y_diff = abs(y - row_medians[row])
                if y_diff > row_thresh:
                    is_outlier = True
            
            if col in col_medians:
                x_diff = abs(x - col_medians[col])
                if x_diff > col_thresh:
                    is_outlier = True
        
        if is_outlier:
            outlier_points.append(point)
        else:
            kept_points.append(point)
    
    return kept_points, outlier_points

def unify_grid_points(points: List[Dict], n_rows: int = 5, n_cols: int = 5) -> Tuple[List[Dict], List[Dict]]:
    """统一的去重和插值"""
    if len(points) < 6:
        return points, []
    
    # 准备数据
    rows = np.array([p['row'] for p in points]).reshape(-1, 1)
    cols = np.array([p['col'] for p in points]).reshape(-1, 1)
    xs = np.array([p['x'] for p in points])
    ys = np.array([p['y'] for p in points])
    
    # 拟合x和y坐标模型
    X_features = np.hstack([rows, cols])
    model_x = LinearRegression()
    model_y = LinearRegression()
    model_x.fit(X_features, xs)
    model_y.fit(X_features, ys)
    
    # 按行列分组
    grid_dict = {}
    for point in points:
        key = (point['row'], point['col'])
        if key not in grid_dict:
            grid_dict[key] = []
        grid_dict[key].append(point)
    
    # 处理每个网格位置
    unified_points = []
    interpolated_points = []
    
    for row in range(n_rows):
        for col in range(n_cols):
            key = (row, col)
            
            # 预测理论值
            X_test = np.array([[row, col]])
            x_theory = model_x.predict(X_test)[0]
            y_theory = model_y.predict(X_test)[0]
            
            if key in grid_dict:
                candidates = grid_dict[key]
                
                if len(candidates) == 1:
                    point = candidates[0]
                    point['theory_distance'] = np.sqrt((point['x'] - x_theory)**2 + (point['y'] - y_theory)**2)
                    unified_points.append(point)
                else:
                    best_point = None
                    best_distance = float('inf')
                    
                    for point in candidates:
                        distance = np.sqrt((point['x'] - x_theory)**2 + (point['y'] - y_theory)**2)
                        if distance < best_distance:
                            best_distance = distance
                            best_point = point
                    
                    if best_point is not None:
                        best_point['theory_distance'] = best_distance
                        best_point['deduplicated'] = True
                        unified_points.append(best_point)
            else:
                interp_point = {
                    'row': row,
                    'col': col,
                    'x': x_theory,
                    'y': y_theory,
                    'interpolated': True
                }
                unified_points.append(interp_point)
                interpolated_points.append(interp_point)
    
    return unified_points, interpolated_points

def detect_filling(contour_info: List[Dict], binary_eroded: np.ndarray, roi_size: Tuple[int, int] = (20, 20), 
                  fill_threshold: float = 0.4) -> Tuple[np.ndarray, np.ndarray]:
    """检测每个点位的填涂情况"""
    fill_matrix = np.zeros((5, 5), dtype=float)
    fill_status = np.zeros((5, 5), dtype=int)
    
    roi_w, roi_h = roi_size
    
    for info in contour_info:
        if 'row' in info and 'col' in info and 'center' in info:
            row = info['row']
            col = info['col']
            cx, cy = info['center']
            
            # 确保坐标为整数
            cx, cy = int(cx), int(cy)
            
            # 计算ROI边界
            x1 = int(cx - roi_w // 2)
            y1 = int(cy - roi_h // 2)
            x2 = int(cx + roi_w // 2)
            y2 = int(cy + roi_h // 2)
            
            # 确保ROI在图像范围内
            h, w = binary_eroded.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 > x1 and y2 > y1:
                roi = binary_eroded[y1:y2, x1:x2]
                if roi.size > 0:
                    fill_ratio = cv2.countNonZero(roi) / roi.size
                    fill_matrix[row, col] = fill_ratio
                    fill_status[row, col] = 1 if fill_ratio > fill_threshold else 0
    
    return fill_matrix, fill_status

def process_single_region(region_img: np.ndarray, region_id: int) -> Dict:
    """处理单个区域"""
    # 转换为灰度
    if len(region_img.shape) == 3:
        gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = region_img.copy()
    
    # CLAHE增强
    clahe = cv2.createCLAHE(clipLimit=REGION_PARAMS['clahe_clip'], tileGridSize=REGION_PARAMS['clahe_grid'])
    clahe_result = clahe.apply(gray)
    
    # 自适应二值化
    binary = cv2.adaptiveThreshold(clahe_result, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, REGION_PARAMS['adaptive_block'], REGION_PARAMS['adaptive_c'])
    
    # 闭运算
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, REGION_PARAMS['close_kernel'])
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 边缘置黑
    border_width = REGION_PARAMS['border_width']
    h, w = closed.shape
    if border_width > 0:
        closed[0:border_width, :] = 0
        closed[h-border_width:h, :] = 0
        closed[:, 0:border_width] = 0
        closed[:, w-border_width:w] = 0
    
    # 轮廓检测
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 过滤轮廓
    valid_contours = []
    contour_info = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w_rect, h_rect = cv2.boundingRect(cnt)
        aspect_ratio = w_rect / h_rect if h_rect > 0 else 0
        
        if (REGION_PARAMS['min_area'] <= area <= REGION_PARAMS['max_area'] and
            REGION_PARAMS['min_aspect_ratio'] <= aspect_ratio <= REGION_PARAMS['max_aspect_ratio']):
            
            valid_contours.append(cnt)
            
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx = x + w_rect // 2
                cy = y + h_rect // 2
            
            contour_info.append({
                'contour': cnt,
                'bbox': (x, y, w_rect, h_rect),
                'area': area,
                'aspect_ratio': aspect_ratio,
                'center': (cx, cy)
            })
    
    # 可疑区域检测
    suspicious = len(contour_info) < REGION_PARAMS['suspicious_point_threshold']
    if suspicious:
        return {
            'region_id': region_id,
            'suspicious': True,
            'suspicious_reason': f"点数不足 ({len(contour_info)} < {REGION_PARAMS['suspicious_point_threshold']})"
        }
    
    # 行列聚类
    centers = [(info['center'][0], info['center'][1]) for info in contour_info]
    clustered_points = cluster_row_col(centers, REGION_PARAMS['n_rows'], REGION_PARAMS['n_cols'])
    
    # 将行列标签添加到contour_info
    for i, point in enumerate(clustered_points):
        if 'original_index' in point and point['original_index'] < len(contour_info):
            idx = point['original_index']
            contour_info[idx]['row'] = point['row']
            contour_info[idx]['col'] = point['col']
    
    # 离群点检测和剔除
    kept_points, outlier_points = detect_and_remove_outliers(
        clustered_points, REGION_PARAMS['n_rows'], REGION_PARAMS['n_cols'], 8.0, 8.0
    )
    
    # 统一的去重和插值
    unified_points, interpolated_points = unify_grid_points(kept_points, REGION_PARAMS['n_rows'], REGION_PARAMS['n_cols'])
    
    # 更新contour_info
    new_contour_info = []
    for point in unified_points:
        if 'original_index' in point:
            idx = point['original_index']
            if idx < len(contour_info):
                info = contour_info[idx].copy()
                info['row'] = point['row']
                info['col'] = point['col']
                if 'deduplicated' in point:
                    info['deduplicated'] = True
                new_contour_info.append(info)
        else:
            new_contour_info.append({
                'row': point['row'],
                'col': point['col'],
                'center': (point['x'], point['y']),
                'interpolated': True
            })
    
    contour_info = new_contour_info
    
    # 填涂检测
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, REGION_PARAMS['erode_kernel'])
    binary_eroded = cv2.erode(binary, kernel_erode)
    fill_matrix, fill_status = detect_filling(contour_info, binary_eroded, 
                                             REGION_PARAMS['roi_size'], REGION_PARAMS['fill_threshold'])
    
    return {
        'region_id': region_id,
        'suspicious': suspicious,
        'contour_info': contour_info,
        'fill_matrix': fill_matrix,
        'fill_status': fill_status,
        'binary': binary,
        'closed': closed,
        'binary_eroded': binary_eroded
    }

# ==================== 可视化函数 ====================
def visualize_preprocessing(resized: np.ndarray, masked: np.ndarray, gray: np.ndarray, 
                           gray_enhanced: np.ndarray, binary: np.ndarray, closed: np.ndarray, 
                           labels: np.ndarray, regions: List[Dict]) -> None:
    """可视化预处理流程，包含连通域彩色标签"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 调整标题间距
    plt.subplots_adjust(top=0.92, hspace=0.2, wspace=0.2)
    
    # 1. 原始缩放图片
    axes[0,0].imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    axes[0,0].set_title('1. 原始缩放图片', fontsize=12, pad=8)
    axes[0,0].axis('off')
    
    # 2. 掩码后图片
    axes[0,1].imshow(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB))
    axes[0,1].set_title('2. 掩码后图片', fontsize=12, pad=8)
    axes[0,1].axis('off')
    
    # 3. 灰度+CLAHE增强
    axes[0,2].imshow(gray_enhanced, cmap='gray')
    axes[0,2].set_title('3. 灰度+CLAHE增强', fontsize=12, pad=8)
    axes[0,2].axis('off')
    
    # 4. CLAHE+自适应阈值
    axes[1,0].imshow(binary, cmap='gray')
    axes[1,0].set_title('4. CLAHE+自适应阈值', fontsize=12, pad=8)
    axes[1,0].axis('off')
    
    # 5. 闭运算+右侧归零
    axes[1,1].imshow(closed, cmap='gray')
    axes[1,1].set_title('5. 闭运算+右侧归零', fontsize=12, pad=8)
    axes[1,1].axis('off')
    
    # 6. 连通域彩色标签
    ax = axes[1,2]
    label_colors = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255)
    ]
    
    for idx, region in enumerate(regions):
        color_idx = idx % len(colors)
        color = colors[color_idx]
        label_colors[labels == region['label']] = color
    
    ax.imshow(label_colors)
    ax.set_title(f'6. 连通域彩色标签 ({len(regions)}个)', fontsize=12, pad=8)
    ax.axis('off')
    
    plt.suptitle('完整的预处理流程', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.show()

def visualize_kmeans_clustering(resized: np.ndarray, regions: List[Dict], 
                               all_horizontal: List[float], all_vertical: List[float],
                               cropped_regions: List[Dict]) -> None:
    """可视化KMeans聚类结果"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 调整标题间距
    plt.subplots_adjust(top=0.85, wspace=0.3)
    
    # 1. 原始点分布（连通域质心分布）
    ax = axes[0]
    centroids_x = [r['centroid'][0] for r in regions]
    centroids_y = [r['centroid'][1] for r in regions]
    ax.scatter(centroids_x, centroids_y, s=20, c='blue', alpha=0.6)
    ax.set_xlabel('X坐标')
    ax.set_ylabel('Y坐标')
    ax.set_title('1. 连通域质心分布', fontsize=12, pad=10)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    
    # 2. 行列聚类结果+分割线
    ax = axes[1]
    if regions and 'row' in regions[0] and 'col' in regions[0]:
        # 为不同行分配颜色
        unique_rows = sorted(set(r['row'] for r in regions))
        unique_cols = sorted(set(r['col'] for r in regions))
        
        # 绘制行聚类结果
        for row_idx, row in enumerate(unique_rows):
            row_points = [r for r in regions if r['row'] == row]
            x_vals = [r['centroid'][0] for r in row_points]
            y_vals = [r['centroid'][1] for r in row_points]
            ax.scatter(x_vals, y_vals, s=40, 
                      label=f'行{row}', alpha=0.7)
        
        # 绘制列聚类结果
        for col_idx, col in enumerate(unique_cols):
            col_points = [r for r in regions if r['col'] == col]
            x_vals = [r['centroid'][0] for r in col_points]
            y_vals = [r['centroid'][1] for r in col_points]
            ax.scatter(x_vals, y_vals, s=20, marker='s',
                      label=f'列{col}', alpha=0.7)
        
        # 绘制分割线
        for y in all_horizontal[1:-1]:
            ax.axhline(y=y, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        for x in all_vertical[1:-1]:
            ax.axvline(x=x, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('X坐标')
    ax.set_ylabel('Y坐标')
    ax.set_title('2. 行列聚类结果+分割线', fontsize=12, pad=10)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    
    # 简化图例
    if regions and 'row' in regions[0] and 'col' in regions[0]:
        # 创建自定义图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='tab:blue', alpha=0.7, label='行聚类'),
            Patch(facecolor='tab:blue', alpha=0.7, label='列聚类（方形）')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # 3. 原图中区域划分
    ax = axes[2]
    img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    ax.imshow(img_rgb)
    
    # 绘制有效区域边界
    for region in cropped_regions:
        x1, y1, x2, y2 = region['x1'], region['y1'], region['x2'], region['y2']
        color = 'green'
        
        # 绘制矩形框
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                            linewidth=2, edgecolor=color, 
                            facecolor='none', alpha=0.8)
        ax.add_patch(rect)
        
        # 添加区域编号
        ax.text(x1 + (x2-x1)/2, y1 + (y2-y1)/2, f"{region['region_id']}", 
               ha='center', va='center', fontsize=10, 
               color=color, weight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_title(f'3. 区域划分结果 (共{len(cropped_regions)}个区域)', fontsize=12, pad=10)
    ax.axis('off')
    
    plt.suptitle('KMeans聚类与区域分割', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.show()

def visualize_region_processing(region_result: Dict) -> None:
    """可视化单个区域的处理过程"""
    region_id = region_result['region_id']
    contour_info = region_result['contour_info']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # 调整标题间距
    plt.subplots_adjust(top=0.9, hspace=0.3, wspace=0.3)
    
    # 1. 原始区域
    axes[0,0].imshow(cv2.cvtColor(globals()[f"region_{region_id:02d}"], cv2.COLOR_BGR2RGB))
    axes[0,0].set_title(f'1. 原始区域{region_id}', fontsize=12, pad=8)
    axes[0,0].axis('off')
    
    # 2. CLAHE+自适应阈值
    axes[0,1].imshow(region_result['binary'], cmap='gray')
    axes[0,1].set_title('2. CLAHE+自适应阈值', fontsize=12, pad=8)
    axes[0,1].axis('off')
    
    # 3. 闭运算
    axes[0,2].imshow(region_result['closed'], cmap='gray')
    axes[0,2].set_title('3. 闭运算', fontsize=12, pad=8)
    axes[0,2].axis('off')
    
    # 4. 腐蚀
    axes[1,0].imshow(region_result['binary_eroded'], cmap='gray')
    axes[1,0].set_title('4. 腐蚀', fontsize=12, pad=8)
    axes[1,0].axis('off')
    
    # 5. 点处理结果
    ax = axes[1,1]
    if contour_info:
        # 创建原始图像的副本
        contour_img = cv2.cvtColor(globals()[f"region_{region_id:02d}"].copy(), cv2.COLOR_BGR2RGB)
        
        # 统计点类型
        point_counts = {
            'original': 0,
            'deduplicated': 0,
            'interpolated': 0
        }
        
        for info in contour_info:
            if 'interpolated' in info and info['interpolated']:
                point_counts['interpolated'] += 1
            elif 'deduplicated' in info and info['deduplicated']:
                point_counts['deduplicated'] += 1
            else:
                point_counts['original'] += 1
        
        # 在图像上方显示统计信息
        stats_text = f"点统计:\n"
        stats_text += f"原始点: {point_counts['original']}个\n"
        stats_text += f"去重点: {point_counts['deduplicated']}个\n"
        stats_text += f"插值点: {point_counts['interpolated']}个\n"
        stats_text += f"总点数: {len(contour_info)}个"
        
        ax.text(0.5, 0.95, stats_text, fontsize=10, ha='center', 
               transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 用不同颜色标记不同类型的点
        for info in contour_info:
            if 'center' in info:
                cx, cy = info['center']
                cx, cy = int(cx), int(cy)
                
                if 'interpolated' in info and info['interpolated']:
                    # 插值点：蓝色
                    cv2.circle(contour_img, (cx, cy), 8, (0, 0, 255), 2)  # 蓝色圆圈
                elif 'deduplicated' in info and info['deduplicated']:
                    # 去重点：橙色
                    cv2.rectangle(contour_img, (cx-6, cy-6), (cx+6, cy+6), (0, 165, 255), 2)  # 橙色方框
                else:
                    # 原始点：红色
                    cv2.drawMarker(contour_img, (cx, cy), (255, 0, 0), 
                                 markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)  # 红色十字
        
        ax.imshow(contour_img)
        ax.set_title('5. 点处理结果', fontsize=12, pad=8)
    ax.axis('off')
    
    # 6. 填涂检测结果
    ax = axes[1,2]
    if contour_info:
        # 创建原始图像的副本
        contour_img = cv2.cvtColor(globals()[f"region_{region_id:02d}"].copy(), cv2.COLOR_BGR2RGB)
        
        # 绘制填涂检测结果
        roi_w, roi_h = REGION_PARAMS['roi_size']
        fill_matrix = region_result['fill_matrix']
        
        for info in contour_info:
            if 'row' in info and 'col' in info and 'center' in info:
                row = info['row']
                col = info['col']
                cx, cy = info['center']
                cx, cy = int(cx), int(cy)
                
                x1 = int(cx - roi_w // 2)
                y1 = int(cy - roi_h // 2)
                x2 = int(cx + roi_w // 2)
                y2 = int(cy + roi_h // 2)
                
                fill_ratio = fill_matrix[row, col]
                is_filled = fill_ratio > REGION_PARAMS['fill_threshold']
                color = (0, 255, 0) if is_filled else (0, 0, 255)
                
                # 绘制框
                cv2.rectangle(contour_img, (x1, y1), (x2, y2), color, 2)
                
                # 在框内标注填涂比例（百分比）
                fill_percent = int(fill_ratio * 100)
                text = f"{fill_percent}%"
                font_scale = 0.4
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
                text_x = cx - text_size[0] // 2
                text_y = cy + text_size[1] // 2
                
                # 添加文字背景
                bg_x1 = text_x - 2
                bg_y1 = text_y - text_size[1] - 2
                bg_x2 = text_x + text_size[0] + 2
                bg_y2 = text_y + 2
                cv2.rectangle(contour_img, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), -1)
                
                # 添加文字
                cv2.putText(contour_img, text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1)
        
        ax.imshow(contour_img)
        
        # 生成答案字符串
        fill_status = region_result['fill_status']
        answers = []
        
        for q_col in range(5):  # 5道题
            # 统计该题填涂的选项
            filled_options = []
            for q_row in range(1, 5):  # 4个选项
                if fill_status[q_row, q_col] == 1:
                    option_label = ['A', 'B', 'C', 'D'][q_row-1]
                    filled_options.append(option_label)
            
            if len(filled_options) == 0:
                answers.append("_")  # 未填涂
            elif len(filled_options) == 1:
                answers.append(filled_options[0])  # 一个选项
            else:
                answers.append("?")  # 多个选项
        
        # 将答案连接成字符串
        answer_str = ''.join(answers)
        
        # 添加答案文本
        ax.text(0.5, 0.05, f"答案: {answer_str}", fontsize=12, 
                ha='center', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        ax.set_title('6. 填涂检测结果', fontsize=12, pad=8)
    ax.axis('off')
    
    plt.suptitle(f'区域{region_id}处理流程可视化', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.show()

def visualize_all_regions_filling(region_results: List[Dict]) -> None:
    """可视化所有区域的填涂结果"""
    n_rows = 5
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*2.5))
    
    # 调整标题间距
    plt.subplots_adjust(top=0.92, hspace=0.4, wspace=0.3)
    
    for idx, result in enumerate(region_results):
        row = idx // n_cols
        col = idx % n_cols
        
        region_id = result['region_id']
        
        if result['suspicious']:
            axes[row, col].text(0.5, 0.5, f"可疑\n区域{region_id}", 
                              ha='center', va='center', fontsize=12, color='red')
        else:
            # 创建可视化图像
            if 'region_img' in result:
                img = result['region_img']
            else:
                img = globals()[f"region_{region_id:02d}"]
            
            # 复制图像
            display_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
            
            # 在图像上绘制填涂结果
            roi_w, roi_h = REGION_PARAMS['roi_size']
            fill_matrix = result['fill_matrix']
            contour_info = result['contour_info']
            
            for info in contour_info:
                if 'row' in info and 'col' in info and 'center' in info:
                    row_idx = info['row']
                    col_idx = info['col']
                    cx, cy = info['center']
                    cx, cy = int(cx), int(cy)
                    
                    fill_ratio = fill_matrix[row_idx, col_idx]
                    is_filled = fill_ratio > REGION_PARAMS['fill_threshold']
                    color = (0, 255, 0) if is_filled else (0, 0, 255)
                    
                    x1 = int(cx - roi_w // 2)
                    y1 = int(cy - roi_h // 2)
                    x2 = int(cx + roi_w // 2)
                    y2 = int(cy + roi_h // 2)
                    
                    cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 1)
                    
                    # 在框内标注填涂比例
                    fill_percent = int(fill_ratio * 100)
                    text = f"{fill_percent}%"
                    font_scale = 0.3
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
                    text_x = cx - text_size[0] // 2
                    text_y = cy + text_size[1] // 2
                    
                    cv2.putText(display_img, text, (text_x, text_y), 
                              cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
            
            axes[row, col].imshow(display_img)
            
            # 生成答案字符串
            fill_status = result['fill_status']
            answers = []
            
            for q_col in range(5):  # 5道题
                # 统计该题填涂的选项
                filled_options = []
                for q_row in range(1, 5):  # 4个选项
                    if fill_status[q_row, q_col] == 1:
                        option_label = ['A', 'B', 'C', 'D'][q_row-1]
                        filled_options.append(option_label)
                
                if len(filled_options) == 0:
                    answers.append("_")  # 未填涂
                elif len(filled_options) == 1:
                    answers.append(filled_options[0])  # 一个选项
                else:
                    answers.append("?")  # 多个选项
            
            # 将答案连接成字符串
            answer_str = ''.join(answers)
            
            # 计算题号范围
            start_question = (region_id - 1) * 5 + 1
            end_question = start_question + 4
            
            axes[row, col].set_title(f'区域{region_id}\n{start_question}-{end_question}: {answer_str}', 
                                    fontsize=9, pad=6)
        
        axes[row, col].axis('off')
    
    for idx in range(len(region_results), n_rows*n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle('所有区域填涂检测结果 (答案格式: AB?C_)', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.show()

def visualize_question_filling(region_results: List[Dict], original_img: np.ndarray, cropped_regions: List[Dict]) -> Tuple[Dict, np.ndarray]:
    """可视化各题作答情况，并在原图上标记"""
    marked_img = original_img.copy()
    h, w = marked_img.shape[:2]
    
    # 计算题号映射
    answer_dict = {}
    total_questions = 0
    
    for result in region_results:
        if not result['suspicious']:
            region_id = result['region_id']
            fill_status = result['fill_status']
            fill_matrix = result['fill_matrix']
            
            # 计算该区域的第一题题号
            start_question = (region_id - 1) * 5 + 1
            
            for col in range(5):  # 5道题
                question_num = start_question + col
                answer_dict[question_num] = {}
                total_questions = max(total_questions, question_num)
                
                for row in range(1, 5):  # 4个选项
                    option_idx = row - 1
                    option_label = ['A', 'B', 'C', 'D'][option_idx]
                    is_filled = fill_status[row, col] == 1
                    answer_dict[question_num][option_label] = is_filled
    
    # 查找区域边界
    region_boundaries = {}
    for region in cropped_regions:
        region_id = region['region_id']
        region_boundaries[region_id] = {
            'x1': region['x1'],
            'y1': region['y1'],
            'x2': region['x2'],
            'y2': region['y2']
        }
    
    # 在原图上标记各题作答情况
    for result in region_results:
        if not result['suspicious']:
            region_id = result['region_id']
            fill_status = result['fill_status']
            fill_matrix = result['fill_matrix']
            contour_info = result['contour_info']
            
            if region_id in region_boundaries:
                region_info = region_boundaries[region_id]
                x1, y1, x2, y2 = region_info['x1'], region_info['y1'], region_info['x2'], region_info['y2']
                
                # 计算该区域的第一题题号
                start_question = (region_id - 1) * 5 + 1
                
                # 绘制区域外框（黄色）
                cv2.rectangle(marked_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                
                # 在区域左上角标注区域编号
                cv2.putText(marked_img, f"R{region_id}", 
                          (x1 + 5, y1 + 20), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # 计算区域尺寸
                region_width = x2 - x1
                region_height = y2 - y1
                
                # 计算每个网格的尺寸
                grid_width = region_width / 5
                grid_height = region_height / 5
                
                # 绘制每个点位的框
                for info in contour_info:
                    if 'row' in info and 'col' in info and 'center' in info:
                        row = info['row']
                        col = info['col']
                        cx, cy = info['center']
                        
                        # 计算框在原始图像中的位置
                        # 首先找到这个点在区域内的相对位置
                        region_img = globals()[f"region_{region_id:02d}"]
                        region_h, region_w = region_img.shape[:2]
                        
                        # 点在区域内的相对位置
                        rel_x = cx / region_w
                        rel_y = cy / region_h
                        
                        # 点在原始图像中的绝对位置
                        abs_x = int(x1 + rel_x * region_width)
                        abs_y = int(y1 + rel_y * region_height)
                        
                        # 计算ROI大小
                        roi_w, roi_h = REGION_PARAMS['roi_size']
                        
                        # 计算ROI在原始图像中的大小
                        scale_x = region_width / region_w
                        scale_y = region_height / region_h
                        
                        scaled_roi_w = int(roi_w * scale_x)
                        scaled_roi_h = int(roi_h * scale_y)
                        
                        # 确保ROI大小合理
                        scaled_roi_w = max(10, min(scaled_roi_w, 30))
                        scaled_roi_h = max(8, min(scaled_roi_h, 20))
                        
                        # 计算框的边界
                        box_x1 = abs_x - scaled_roi_w // 2
                        box_y1 = abs_y - scaled_roi_h // 2
                        box_x2 = abs_x + scaled_roi_w // 2
                        box_y2 = abs_y + scaled_roi_h // 2
                        
                        # 确定颜色
                        if row == 0:  # 题号行
                            color = (255, 0, 0)  # 蓝色
                            
                            # 在框上方标注题号
                            question_num = start_question + col
                            cv2.putText(marked_img, f"Q{question_num}", 
                                      (abs_x - 15, abs_y - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                        else:  # 选项行
                            fill_ratio = fill_matrix[row, col]
                            is_filled = fill_ratio > REGION_PARAMS['fill_threshold']
                            color = (0, 255, 0) if is_filled else (0, 0, 255)  # 绿色:填涂, 红色:未填涂
                        
                        # 绘制框
                        cv2.rectangle(marked_img, (box_x1, box_y1), (box_x2, box_y2), color, 2)
                        
                        # 在选项框中标注填涂比例
                        if row > 0:  # 选项行
                            fill_percent = int(fill_ratio * 100)
                            text = f"{fill_percent}%"
                            font_scale = 0.3
                            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
                            text_x = abs_x - text_size[0] // 2
                            text_y = abs_y + text_size[1] // 2
                            
                            cv2.putText(marked_img, text, (text_x, text_y), 
                                      cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
    
    # 显示标记后的图片
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(marked_img, cv2.COLOR_BGR2RGB))
    plt.title('答题卡作答情况标记 (蓝色:题号, 绿色:填涂, 红色:未填涂)', fontsize=16, pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return answer_dict, marked_img

# ==================== 主程序 ====================
def main():
    """主程序"""
    print("="*60)
    print("答题卡识别与处理系统")
    print("="*60)
    
    # 1. 读取和预处理
    img_path = 'images/test3.jpg'
    resized, masked, gray, gray_enhanced, binary, closed = preprocess_image(img_path)
    
    # 2. 连通域分析
    labels, regions = analyze_connected_components(closed)
    
    # 3. 可视化预处理流程
    visualize_preprocessing(resized, masked, gray, gray_enhanced, binary, closed, labels, regions)
    
    # 4. KMeans聚类和区域分割
    all_horizontal, all_vertical = kmeans_clustering_regions(regions, resized)
    cropped_regions = crop_regions(resized, all_horizontal, all_vertical, regions)
    
    # 5. 可视化KMeans聚类结果
    visualize_kmeans_clustering(resized, regions, all_horizontal, all_vertical, cropped_regions)
    
    # 6. 创建区域变量
    print(f"\n创建区域图像变量")
    for i, region in enumerate(cropped_regions, 1):
        var_name = f"region_{i:02d}"
        globals()[var_name] = region['cropped'].copy()
        print(f"{var_name}: 尺寸{region['cropped'].shape}")
    
    # 7. 处理所有区域
    print(f"\n处理所有区域")
    print(f"可疑区域阈值: 点数<{REGION_PARAMS['suspicious_point_threshold']}")
    print(f"填涂检测: ROI大小{REGION_PARAMS['roi_size']}, 阈值{REGION_PARAMS['fill_threshold']}")
    
    region_results = []
    suspicious_regions = []
    normal_regions = []
    
    for i in range(1, len(cropped_regions) + 1):
        var_name = f"region_{i:02d}"
        if var_name in globals():
            region_img = globals()[var_name]
            result = process_single_region(region_img, i)
            
            # 保存区域图像
            result['region_img'] = region_img.copy()
            
            region_results.append(result)
            
            if result['suspicious']:
                suspicious_regions.append(i)
            else:
                normal_regions.append(i)
    
    print(f"\n正常区域: {len(normal_regions)}个 ({normal_regions})")
    print(f"可疑区域: {len(suspicious_regions)}个 ({suspicious_regions})")
    
    # 8. 选择一个代表区域进行详细可视化
    if normal_regions:
        representative_region = region_results[normal_regions[0] - 1]
        visualize_region_processing(representative_region)
    
    # 9. 可视化所有区域的填涂结果
    visualize_all_regions_filling(region_results)
    
    # 10. 可视化各题作答情况
    answer_dict, marked_img = visualize_question_filling(region_results, resized, cropped_regions)
    
    # 打印统计信息
    print(f"\n答题统计:")
    print(f"总题数: {len(answer_dict)}")
    
    filled_count = 0
    multiple_count = 0
    empty_count = 0
    
    for q in answer_dict:
        filled_options = []
        for option in ['A', 'B', 'C', 'D']:
            if answer_dict[q][option]:
                filled_options.append(option)
        
        if len(filled_options) == 0:
            empty_count += 1
        elif len(filled_options) == 1:
            filled_count += 1
        else:
            multiple_count += 1
    
    print(f"正常填涂: {filled_count}题")
    print(f"多选项: {multiple_count}题")
    print(f"未填涂: {empty_count}题")
    print(f"填涂比例: {(filled_count+multiple_count)/len(answer_dict)*100:.1f}%")
    
    # 12. 返回结果
    return {
        'original_image': resized,
        'marked_image': marked_img,
        'cropped_regions': cropped_regions,
        'region_results': region_results,
        'answer_dict': answer_dict,
        'suspicious_regions': suspicious_regions,
        'normal_regions': normal_regions
    }

# ==================== 运行程序 ====================
if __name__ == "__main__":
    results = main()
    
    # 保存处理结果
    processed_results = results
    print(f"\n处理结果已保存到变量 'processed_results'")
    print("="*60)
    print("答题卡识别与处理完成！")
    print("="*60)