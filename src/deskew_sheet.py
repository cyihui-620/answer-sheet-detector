'''
Funtion: 检测答题卡水平黑线指示倾斜角度, 旋转校正倾斜页面, 便于后续处理
Note: 输入答题卡倾斜角不超过45°
Author: 蔡奕辉
Time: 2025-12-18
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_angle_of_black_line(img):
    '检测答题卡中间和底部两条黑线的倾斜角'
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    #二值化（检测黑线）, 手定阈值:30
    _, binary= cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    #霍夫变换直线检测，检测到多条水平线（分布于两条黑线的上下边缘）
    lines = cv2.HoughLinesP(edges,
                        rho=1,                
                        theta=np.pi/180,      
                        threshold=150,         # 投票阈值 -> 控制检测灵敏度（值小检测更多直线）
                        minLineLength=200,     # 最小线段长度 
                        maxLineGap=100)        # 最大线段间隙
    angle_list = []
    if lines is not None:
        print(f"检测到 {len(lines)} 条直线")
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle)<45:  #  检测基本要求:图像倾斜角<45°
                angle_list.append(angle)
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            print(f"线段 {i}: ({x1},{y1})-({x2},{y2}), "
                f"长度={length:.1f}px, 角度={angle:.1f}°")
    else:
        print("未检测到直线")

    sorted_angle_list = sorted(angle_list)
    rotation_angle = sorted_angle_list[len(sorted_angle_list)//2]
    print(f"建议答题卡逆时针旋转校正角度:{rotation_angle:.1f}°") 

    return rotation_angle   

def deskew_sheet(img_path, debug = True):
    img = cv2.imread(img_path)
    
    deskew_angle = detect_angle_of_black_line(img)
   
    angle_threshold =  1 
    if abs(deskew_angle) > angle_threshold: #倾斜角小于1°不校正
        #旋转变换
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, deskew_angle, 1.0)
        deskewed = cv2.warpAffine(img, M, (w, h),
                                    flags=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(255, 255, 255))
    else:
        print(f"原图形倾斜角小于{angle_threshold}°, 无需旋转校正")     

    if debug:
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Original')
        plt.axis('off')
        plt.show()

        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(deskewed, cv2.COLOR_BGR2RGB))
        plt.title('Deskewed')
        plt.axis('off')
        plt.show()

    return deskewed

if __name__ == "__main__":
    deskew_sheet('images/test3.jpg', debug = True) 
