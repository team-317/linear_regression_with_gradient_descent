# 通过逻辑斯蒂回归完成简单的室内外图像二分类任务

## 任务概要

给定一个室内外图像的训练集和测试集，通过训练集训练一个分类器，使其能正确对测试集中的室内外图像进行分类。



## 实现过程

### 1.数据预处理

### 1.1 图像统一化

在给定的数据集中，各类图像的大小不一致，将每张图像的大小缩放为64x64x3，使其大小统一。

### 1.2 图像特征提取

一张64x64x3的图像，如果将每个像素值作为一个特征进行处理，将带来较大的计算量，因而对图像进行特征提取，用少量的特征来代表一张图像。在一张图像中，各个像素值的大小各有差异，比如下面这张图像，其中深色的山体（像素值较低）和灰白色的江水（像素值较高）是图像的两种主要成分，统计图像中各类像素值的分布情况，得到下方对应的像素分布直方图。

![histogram_sample.jpg](https://docs.opencv.org/4.x/histogram_sample.jpg)

在室内外图像分类中，我们可以假定室外图像多为深色图像（低值像素偏多），室内多为有灯光的亮色图像（高值像素偏多），这正好可以通过像素分布直方图来反映。

因而对于一张图像，这里统计该图在RGB三个通道上的像素分布直方图，由于原始图像像素取值范围为0~255，因而直方图具有256x3个特征。为简化计算，将每个通道的256个特征以分桶的测量划分到16个区间，得到48个特征。

### 1.3 图像归一化

由于每张图像中像素值的取值范围各有差异，对其进行归一化，将其限制在[0, 1]。

## 2.建立分类模型

记某一图像的特征为$x\in R^{1 \times 48}$，对应的标签为$y$将室外图像记为正例，取值为1，室内图像记为负例，取值为0。该任务的简单线性回归模型如下：
$$
y = w^T x + b\quad\quad\quad\quad(1)
$$
普通回归模型的二分类能力不强，引入逻辑回归函数：
$$
y = \frac{1}{1 + e^{-(w^T x + b)}}\quad\quad\quad\quad(2)
$$
若将$y$表示为样本$x$为正例的可能性，则$1-y$为其返反例的可能性，两者的比值$y/(1-y)$称为“几率”，反映了$x$作为整理的相对可能性，对几率取对数得到“对数几率”$ln(y/(1-y))$，将(2)式代入得：
$$
ln\frac{y}{1-y}=ln(\frac{1}{1 + e^{-(w^T x + b)}}/\frac{e^{-(w^T x + b)}}{1 + e^{-(w^T x + b)}}) = ln\frac{1}{e^{-(w^T x + b)}} = w^Tx + b\quad \quad (3)
$$
通过“最大似然法”可估计$w$和$b$的值，给定数据集$\{(x_i, y_i)\}^m_{i=1}$，记$\hat{w} = (w, b),\ \hat{x}=(x, 1)$，最大化“对数似然”函数求解参数$\hat{w}$（推导过程参考南瓜书，对应西瓜书公式3.21~3.27）：
$$
L(\hat{w}) = \max \sum^m_{i=1}ln\ p(y_i|\hat{x_i}; \hat{w})\\
\quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad
= \max \sum^m_{i=1}ln\{ y_ip_1(\hat{x}_i; \hat{w})  + (1 - y_i)p_0(\hat{x}_i; \hat{w})\}	\\
\quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad 
= \min \sum^m_{i=1}-y_i \hat{w}^T \hat{x_i} + ln(1 + e^{ \hat{w}^T \hat{x_i}})  \quad \quad(4)
$$

## 3.梯度下降求解优化模型

记$\hat{X} = (X, 1)$计算(4)式对$\hat{w}$的梯度，得到$\hat{w}$的梯度：
$$
L(\hat{w})^{'} = \frac{e^{\hat{w}^T_{k} \  \hat{X}}}{1 + e^{\hat{w}^T_{k} \  \hat{X}}} \ \hat{X} - Y \hat{X}  \\
\hat{w}_{k+1} = \hat{w}_k - \alpha L(\hat{w})^{'} 
$$
其中$\alpha$为更新步长。

## 4.参考资料

【1】周志华——西瓜书

【2】[矩阵求导符号计算网站](http://www.matrixcalculus.org/)

【3】[cv2直方图计算]([OpenCV: Histograms - 1 : Find, Plot, Analyze !!!](https://docs.opencv.org/4.x/d1/db7/tutorial_py_histogram_begins.html))