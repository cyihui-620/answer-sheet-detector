'''
Step1: 提取答题区域和学号填涂区域，并透视变换转正视图
Author: 李明珠
Time: 2025-12-18
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

def order_points(pts):
    """func:排序角点"""
    rect = np.zeros((4, 2), dtype="float32")
    sum = pts.sum(axis=1)
    rect[0] = pts[np.argmin(sum)]  # 左上：x+y最小
    rect[2] = pts[np.argmax(sum)]  # 右下：x+y最大
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上：x-y最小
    rect[3] = pts[np.argmax(diff)]  # 左下：x-y最大
    
    return rect

def perspective_transform(img, corners, show=False):
    """func:对指定区域进行透视变换，转为正视图"""
    pts = corners.reshape(4, 2)
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # 计算新图像宽高
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # 目标矩形的四个角点（正视图）
    dst = np.array([
        [0, 0],                    # 左上
        [maxWidth - 1, 0],         # 右上
        [maxWidth - 1, maxHeight - 1],  # 右下
        [0, maxHeight - 1]         # 左下
    ], dtype="float32")
    
    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(rect, dst)

    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    
    if show:
        plt.imshow(warped, cmap='gray')
        plt.title('perspective transform')
        plt.show()

    return warped, M

def find_answer_sheet_corners(img_gray, show=False):
    """func:找到答题区域和学号填涂区域的角点"""
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 最大的轮廓是答题卡区域
    largest_contour = max(contours, key=cv2.contourArea)
    # 次大的轮廓是学号填涂区
    second_largest_contour = sorted(contours, key=cv2.contourArea, reverse=True)[1]
    
    # 使用最小外接矩形获取角点
    rect_ans = cv2.minAreaRect(largest_contour)
    ans_box = cv2.boxPoints(rect_ans).astype("float32")
    
    rect_id = cv2.minAreaRect(second_largest_contour)
    id_box = cv2.boxPoints(rect_id).astype("float32")
    
    if show: # debug用
        img_contour = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)  # 转换为BGR
        
        # 最大轮廓（红色加粗）
        cv2.drawContours(img_contour, [largest_contour], -1, (0, 0, 255), 3)
        # 次大轮廓（蓝色加粗）
        cv2.drawContours(img_contour, [second_largest_contour], -1, (255, 0, 0), 3)
        # 其他轮廓（浅绿色）
        for i, contour in enumerate(contours):
            if contour is not largest_contour:
                cv2.drawContours(img_contour, [contour], -1, (0, 255, 0), 1)
        
        # 绘制最小外接矩形（黄色、青色）
        ans_box_int = ans_box.astype(np.int32)
        cv2.polylines(img_contour, [ans_box_int], True, (0, 255, 255), 2)
        
        id_box_int = id_box.astype(np.int32)
        cv2.polylines(img_contour, [id_box_int], True, (255, 255, 0), 2)

        # 显示轮廓
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(img_contour, cv2.COLOR_BGR2RGB))
        plt.title('Contours Detection (Yellow: Answer Sheet, Cyan: ID Region)')
        plt.axis('off')
        plt.show()
    
    return ans_box, id_box

def process_answer_sheet(image_path, show=False):
    """func:功能集合实现"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ans_box, id_box = find_answer_sheet_corners(gray, show=show)
    
    sheet_warped, _ = perspective_transform(img, ans_box, show=show)
    id_region_warped, _ = perspective_transform(img, id_box, show=show)
    
    return sheet_warped, id_region_warped

# debug用
if __name__ == "__main__":
    sheet_warped, id_region_warped = process_answer_sheet('./images/sheet.jpg', show=True)
        