# 图像处理原理说明

## 1. 图像二值化处理

### 使用示例
```python
_, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
```

### 原理说明
**大津法（OTSU）自适应阈值**是一种自动确定最佳阈值的算法，特别适用于双峰直方图的图像。

**算法原理：**

1. 计算图像的灰度直方图
2. 遍历所有可能的阈值（0-255）
3. 对于每个阈值t，将像素分为两类：$C_1（≤t）$和$C_2（＞t）$
4. 计算类间方差$σ²(t) = ω₁(t)ω₂(t)[μ₁(t)-μ₂(t)]²$
5. 选择使类间方差最大的阈值作为最优阈值

**数学公式：**
- 类间方差：$\sigma^2(t) = \omega_1(t)\omega_2(t)[\mu_1(t)-\mu_2(t)]^2$
- 其中：
  - $\omega_1(t) = \sum_{i=0}^{t}p(i)$，$\omega_2(t) = \sum_{i=t+1}^{255}p(i)$
  - $\mu_1(t) = \frac{\sum_{i=0}^{t}i·p(i)}{\omega_1(t)}$，$\mu_2(t) = \frac{\sum_{i=t+1}^{255}i·p(i)}{\omega_2(t)}$

**代码作用：** 将灰度图像转换为二值图像，便于后续的轮廓检测。

**详细过程推导**：https://zhuanlan.zhihu.com/p/649435540

## 2. 形态学闭运算

### 使用示例
```python
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
```

### 原理说明
**形态学闭运算**是先膨胀后腐蚀的操作，用于填充小孔洞和连接断裂的区域。

**数学定义：**
- 闭运算：$A \bullet B = (A \oplus B) \ominus B$
- 其中：
  - $A$是输入图像
  - $B$是结构元素
  - $\oplus$表示膨胀操作
  - $\ominus$表示腐蚀操作

**膨胀操作原理：**
- $A \oplus B = \{z | (\hat{B})_z \cap A \neq \varnothing\}$
- 用结构元素B扫描图像的每个像素，用结构元素与其覆盖的二值图像做"或"操作

**腐蚀操作原理：**
- $A \ominus B = \{z | (B)_z \subseteq A\}$
- 用结构元素B扫描图像的每个像素，用结构元素与其覆盖的二值图像做"与"操作

**代码作用：** 消除答题卡区域内的噪声点和小断裂，确保轮廓的完整性。

![](images\1.png)

![](images/2.png)



## 3. 轮廓检测

### 使用示例
```python
contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

### 原理说明
**轮廓检测算法**基于Suzuki85的边界跟踪算法，用于提取图像中的连通区域边界。

**算法步骤：**
1. **二值化预处理**：输入必须是二值图像
2. **边界跟踪**：从左上角开始扫描，遇到白色像素时开始边界跟踪
3. **轮廓层次构建**：记录轮廓的父子关系（RETR_EXTERNAL只检测最外层轮廓）
4. **轮廓点简化**：CHAIN_APPROX_SIMPLE仅保留轮廓的端点，压缩水平、垂直和对角线段

**数学原理：**
- 使用8邻域或4邻域连通性进行边界跟踪
- 通过链码表示轮廓的走向
- 轮廓点集满足：$C = \{p_1, p_2, ..., p_n\}$，其中$p_{i+1}$是$p_i$的邻域点

**代码作用：** 检测图像中的所有闭合轮廓，为后续的区域识别做准备。

**边界跟踪算法Suzuki85具体原理解析**：https://zhuanlan.zhihu.com/p/397588540

## 4. 最小外接矩形

### 使用示例
```python
rect_ans = cv2.minAreaRect(largest_contour)
ans_box = cv2.boxPoints(rect_ans).astype("float32")
```

### 原理说明
**最小外接矩形算法**寻找能够完全包围轮廓且面积最小的旋转矩形。

**算法原理（基于旋转卡壳法）：**

1. 计算轮廓的凸包
2. 使用旋转卡壳法寻找凸包的最小面积外接矩形
3. 遍历凸包的所有边作为矩形的基准边
4. 对于每条边，计算：
   - 与该边平行的对跖点（最远点）
   - 与该边垂直的最远点
5. 记录面积最小的矩形

**数学公式：**
- 矩形面积：$Area = width \times height$
- 旋转角度：$\theta = \arctan(\frac{y_2-y_1}{x_2-x_1})$
- 矩形中心：$center = \frac{1}{4}\sum_{i=1}^{4} vertex_i$

**代码作用：** 获取答题区域和学号区域的精确边界框，用于透视变换。

**旋转卡壳法详细讲解：**https://blog.csdn.net/hanchengxi/article/details/8639476

## 5. 透视变换

### 使用示例
```python
M = cv2.getPerspectiveTransform(rect, dst)
warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
```

### 原理说明
**透视变换**将图像从任意视角投影到正视图，消除透视畸变。

**数学原理（单应性矩阵）：**
透视变换使用3×3的单应性矩阵进行坐标映射：

$$\begin{bmatrix}
x' \\ y' \\ w'
\end{bmatrix} = 
\begin{bmatrix}
h_{11} & h_{12} & h_{13} \\
h_{21} & h_{22} & h_{23} \\
h_{31} & h_{32} & h_{33}
\end{bmatrix}
\begin{bmatrix}
x \\ y \\ 1
\end{bmatrix}$$

归一化后的坐标：$x'' = \frac{x'}{w'}$，$y'' = \frac{y'}{w'}$

**求解单应性矩阵：**
给定4组对应点，通过解线性方程组求H矩阵：
- 每对点提供2个方程：$x' = \frac{h_{11}x + h_{12}y + h_{13}}{h_{31}x + h_{32}y + h_{33}}$
- 共8个未知数，需要至少4对点

**重映射过程：**
1. 对目标图像的每个像素(x,y)
2. 计算在原图像中的对应位置：$[x', y', w']^T = H^{-1}[x, y, 1]^T$
3. 通过双线性插值获取像素值

**代码作用：** 将倾斜的答题卡区域校正为规整的正方形视图，便于后续的选项识别。

![](images/3.png)