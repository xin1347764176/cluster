# 陈列识别

## 简介

集成 `getboxes.py` 即可，该文件是接口文件。`test_main.py` 是测试代码，调用 `process_interface` 函数，输入标签和位置坐标，输出对应的簇和位置坐标，并返回 `ans`。

## 函数说明

### `process_interface`

- **输入**：
  - `boxes` 格式为 `List[Tuple[str, Tuple[float, float, float, float]]]`，其中每个元素包含标签和位置坐标 `(label, (x_min, y_min, x_max, y_max))`。

- **输出**：
  - `ans` 的格式为 `List[Tuple[int, int]]`，表示对应输出 `(第几排, 第几列)`。`(-1, -1)` 表示被过滤掉的项。

## 评价指标

### `test_metric.py`

`test_metric.py` 是用于验证评价指标的代码。

## 聚类算法实现

![最终效果示意图](image.png)  
![聚类算法示意图](./pic_single/output_image.jpg)  

## 陈列识别 Demo 支持

### 前提

限定图片中为货架上基本整齐摆放的商品。

### 输入

所有前排商品的检测框。

### 输出

总计有几排商品；每个商品（对应一个输入的检测框）是放置在第几排的第几个。

## 目标

通过一段策略实现商品检测框的聚类。以下是主要步骤和代码实现的详细说明。

## 步骤

1. **提取中心点坐标**：
   - 获取每个商品的检测框的中心点坐标，构成集合 P。

2. **定义点的距离度量**：
   - 设点的距离度量 \( D_{P2P} = \sqrt{dx^2 + k \cdot dy^2} \)，其中 \( dx \) 和 \( dy \) 分别为两个点的横纵坐标的差值，\( k \) 为可调参数，暂定为 10。

3. **定义簇的距离度量**：
   - 设簇的距离度量 $$
D_{C2C} = \min_{p1 \in C1, p2 \in C2} [D_{P2P}(p1, p2)]
$$，其中 \( C1 \) 和 \( C2 \) 表示两个簇。

4. **设定簇距离阈值**：
   - 给定簇距离阈值 \( 0.002233 \cdot \text{average_area} + 173.80 \)。

5. **初始化簇**：
   - 初始化每个点为一个簇。

6. **合并最近的簇**：
   - 取距离最小的两个簇，若它们的距离小于 \( d \)，则合并这两个簇。重复此过程，直至最小簇距离大于或等于 \( d \)。

7. **过滤小簇**：
   - 过滤掉少于 3 个元素的簇。对每个簇的上边缘拟合直线，计算每个框的点到拟合直线的距离（y 方向）与框高度的比值，求平均值。作为杂乱簇的过滤指标。

8. **合并同一层的簇**：
   - 根据拟合直线的斜率和截距合并同一层的簇。

9. **排序和绘制**：
   - 对簇按 y 坐标均值排序，对簇内的点按 x 坐标排序。
   - 遍历所有点，找到相等中心点，绘制聚类结果。


