'''
Step2: 识别11位学号
Author: 李明珠
Time: 2025-12-18
'''
from find_region import process_answer_sheet
import cv2
import numpy as np

# 全局变量
confidence = 0.09

def process(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # 手动阈值：THRESH_BINARY_INV 表示大于阈值的变0（黑），小于阈值的变255（白）
    _, binary = cv2.threshold(blurred, 75, 255, cv2.THRESH_BINARY_INV)
    
    # 形态学运算
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return opened

def recognize_student_id(img, num_digits=11, visualize=False):
    binary = process(img)
    h, w = binary.shape

    # col_region 和 row_region 不是完整的整个图像
    row_bias = 110
    col_bias_left = 10
    col_bias_right = 95
    col_region = binary[:, col_bias_left:w-col_bias_right]
    row_region = binary[row_bias:h, :]

    col_width = col_region.shape[1] // num_digits
    row_height = row_region.shape[0] // 10
    student_id = []
    
    # 创建可视化
    if visualize:
        vis_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    for col_idx in range(num_digits):
        if col_idx < num_digits - 1:
            start_x = col_idx * col_width
            end_x = (col_idx + 1) * col_width
        else:
            start_x = col_idx * col_width
            end_x = col_region.shape[1]
        
        current_col = col_region[row_bias:, start_x:end_x]
        
        max_fill_ratio = 0
        detected_digit = None
        
        # 遍历每一行（数字0-9）
        for row_idx in range(10):
            cell_start_y = row_idx * row_height
            cell_end_y = (row_idx + 1) * row_height if row_idx < 9 else current_col.shape[0]
            
            cell = current_col[cell_start_y:cell_end_y, :]
            fill_ratio = cv2.countNonZero(cell) / cell.size
            
            # 可视化：绘制每个单元格的边框
            if visualize:
                abs_start_x = col_bias_left + start_x
                abs_end_x = col_bias_left + end_x
                abs_start_y = row_bias + row_idx * row_height
                abs_end_y = row_bias + (row_idx + 1) * row_height if row_idx < 9 else h
                
                color = (0, 255 - row_idx * 25, row_idx * 25)  # 从蓝到红的渐变
                cv2.rectangle(vis_img, (abs_start_x, abs_start_y), 
                             (abs_end_x, abs_end_y), color, 1)
                
                # 显示行号（数字0-9）
                cv2.putText(vis_img, f'{row_idx}', 
                           (abs_start_x + 2, abs_start_y + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            if fill_ratio > max_fill_ratio:
                max_fill_ratio = fill_ratio
                detected_digit = row_idx
        
        # 可视化：标记检测到的数字
        if visualize:
            abs_start_x = col_bias_left + start_x
            abs_mid_x = col_bias_left + (start_x + end_x) // 2
            if detected_digit is not None and max_fill_ratio > confidence:
                abs_mid_y = row_bias + row_height * detected_digit + row_height // 2
                # 绘制绿色圆圈标记检测到的数字
                cv2.circle(vis_img, (abs_mid_x, abs_mid_y), 8, (0, 255, 0), 2)
                cv2.putText(vis_img, str(detected_digit), 
                           (abs_mid_x - 8, abs_mid_y + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # # 调试信息：打印第11列的检测结果
        # if col_idx == 10:
        #     print(f"\n第11列调试信息:")
        #     print(f"  col_region宽度={col_region.shape[1]}, col_width={col_width}")
        #     print(f"  start_x={start_x}, end_x={end_x}, 列宽度={end_x-start_x}")
        #     print(f"  current_col形状={current_col.shape}")
        #     print(f"  max_fill_ratio={max_fill_ratio:.3f}, detected_digit={detected_digit}")
        #     print(f"  confidence阈值={confidence}, 是否通过={max_fill_ratio > confidence}")
        
        if max_fill_ratio > confidence:
            student_id.append(str(detected_digit))
        else:
            student_id.append('?')
    
    # 显示可视化结果
    if visualize:
        # 绘制列分割线
        for col_idx in range(num_digits + 1):
            x = col_bias_left + col_idx * col_width
            cv2.line(vis_img, (x, 0), (x, h), (255, 255, 0), 1)
        
        # 绘制行分割线
        for row_idx in range(11):
            y = row_bias + row_idx * row_height
            cv2.line(vis_img, (0, y), (w, y), (255, 0, 255), 1)
        
        cv2.imshow('Region Visualization', vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return ''.join(student_id)

# 主程序
if __name__ == "__main__":
    sheet_warped, id_region_warped = process_answer_sheet('./images/sheet.jpg')
    
    # 需要调试时开 visualie=True
    student_id = recognize_student_id(id_region_warped, num_digits=11, visualize=True)
    print(f"识别到的学号: {student_id}")
    
