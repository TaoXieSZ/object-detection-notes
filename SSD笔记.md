# SSD笔记

参考知乎文章：[https://www.zhihu.com/search?type=content&q=ssd%20%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B](https://www.zhihu.com/search?type=content&q=ssd 目标检测)

## 首先，再次复习一下目标检测算法的两大主流：

1. two-stage：先通过启发式方法（selective search）经典的如RCNN，或者如CNN网络，如RPN，产生一组候选框，随后对这些候选框进行分类与回归；
2. one-stage：其中经典的方法有SSD和YOLO，其主要思路是均匀地在图片的不同位置上，进行不同尺度和长宽比的密集抽样，然后利用CNN提取特征后直接进行分类和回归。

对比一下，two-stage的方法在准确率上较有优势，one-stage的方法在速度上更占优势，然而由于均匀的密集采样的一个很大的缺点是训练比较困难，这主要因为正样本和负样本（即背景）的及其不平衡，导致准确率较低。这一点上，在Focal Loss的论文有所研究，作为下一篇要看的论文！

<img src="https://pic2.zhimg.com/v2-f143b28b7a7a1f912caa9a99c1511849_b.jpg" alt="img" style="zoom:67%;" />



SSD算法全称为Single shot multibos detector：

1. single shot：表明SSD算法属于one-stage
2. MultiBox：表明SSD是多框检测

## 相比于YOLO，SSD的区别有：

1. 采用CNN直接进行检测，而不是如YOLO中在全连接层之后做检测。
2. SSD企图不同尺度的特征图来做检测，大尺度特征图（高清）用来检测小物体，小尺度特征图（模糊）用来检测大物体
3. 采用不同尺度和长宽比的先验框，这一概念的名字因模型而异（Prior bexes, default boxes, anchor boxes）。

<img src="https://pic3.zhimg.com/v2-4c1d4d1b857a88b347549e54e15f322e_b.jpg" alt="img" style="zoom:80%;" />

## 设计理念：

基本框架：

![img](https://pic1.zhimg.com/v2-bfaaa064fb1f0c1c7a11a4ce79962e84_b.jpg)

### 1. 采用多尺度特征图用于检测

多尺度即采用不同大小的特征图，CNN网络浅层输出特征图较大，随着层数增加，会用ConV或者Pooling降低特征图的大小。

<img src="https://pic2.zhimg.com/v2-4e827b166ba5cbaa5ac8b428a32b885d_b.jpg" alt="img" style="zoom:67%;" />

如图所示，8x8的特征图可以用来检测较小的物体，4x4的特征图用来检测较大的物体，而两者的先验框大小也是不一样的。

### 2. 采用卷积层进行检测

YOLOv1中采用全连接层来进行检测，而SSD直接用卷积操作对特征图进行检测，从这一点上提高速度。



### 3. 设置先验框

在YOLOv1中，每个grid输出多个bbox，这些bbox的大小位置是相对于这个grid而计算的，但由于图像中，目标的形状显然是多变的，那么在训练的时候就需要自适应目标的形状。

而SSD则引入了Faster RCNN中anchor理念，一定程度上可以减少训练的难度。如下图所示，SSD对于特征图中的每个grid，设置4个大小、长宽比不一的anchor boxes。

![img](https://pic1.zhimg.com/v2-f6563d6d5a6cf6caf037e6d5c60b7910_b.jpg)

SSD的预测值分为两个部分：

1. 每个类别的置信度向量，长度为$c+1$，其中$c$是数据集类别，1是模型增加的背景类别。
2. bbox的位置，包含4个值，$(cx, cy, w, h)$，表达bbox的中心位置坐标和长宽。

而SSD预测的时候，输出的是预测框相对于anchor box的offset，文章中指出可能用transformation更恰当：

用$d = (d^{cx}, d^{cy}, d^{w}, d^{h})$表示anchor的回归值，用$b = (b^{cx}, b^{cy},b^{w}, b^{h})$表示边界框的回归值，那么预测值$l$：
$$
l^{cx} = \frac{b^{cx} - d^{cx}}{d^w}\\
l^{cy} = \frac{b^{cy} - d^{cy}}{d^h}\\
l^w = log(\frac{b^w}{d^w})\\
b^h = log(\frac{b^h}{d^h})
$$
那么输出分类和回归值的时候，从预测值$l$中得到边界框的值：
$$
b^{cx} = d^w\cdot l^{cx} + d^{cx}\\
b^{cy} = d^h\cdot l^{cy} + d^{cy}\\
b^w = d^w\cdot exp(l^w)\\
b^h = d^h\cdot exp(l^h)
$$
而在源码中，还有设置超参数$variance$来调整预测值，通过该参数来控制两种模式：

1. $variance=True$:  预测值为以上算法；

2. $variance=False$:需要手动设置超参数，来对$l$进行缩放：
   $$
   b^{cx} = d^w\cdot (variance[0]*l^{cx}) + d^{cx}\\
   b^{cy} = d^h\cdot (variance[1]*l^{cy}) + d^{cy}\\
   b^w = d^w\cdot exp(variance[2]*l^w)\\
   b^h = d^h\cdot exp(variance[3]*l^h)
   $$
    

综上所述，对于一个大小为$m\times n$的特征图，划分为$mn$个grid，每个grid设置k个anchor boxes，那么每个grid要输出$(c+1+4)k$个预测值，所有的grid输出$(c+1+4)kmn$个预测值，因此需要$(c+1+4)k$个卷积核来完成该层的预测。



## 网络结构

SSD采用VGG16作为backbone，并加入卷积层来获得更多的特征图用来检测。

![img](https://pic1.zhimg.com/v2-a43295a3e146008b2131b160eec09cd4_b.jpg)

首先VGG16在ILSVRC CLS-LOC数据集预训练。然后借鉴了DeepLab的LargeFOV，分别将全连接层中的fc6和fc7转换成$3\times3$的卷积层conv6和$1\times1$的卷积层conv7，同时将池化层由原来的stride=5的$2\times2$改成strede=1的$3\times 3$ 。

为了配合这种操作，引入一种Atrous算法，其实就是conv6采用扩展卷积或者带孔卷积（dilation conv），在不增加参数与模型复杂度的条件下指数级地扩大卷积的视野（以参数——扩张率（dilation rate），表示扩张的大小）。

1. (a)是普通的$3\times3$ 卷积，其视野为3x3；
2. (b)是扩张率为2的带孔卷积，其视野扩展为所谓的7x7；
3. (c)是扩张率为4的卷积，视野为15x15，但特征更加系数。

而conv6采用的是3x3但扩张率为6的带孔卷积。

![img](https://pic4.zhimg.com/v2-e3201dedee06e7793affbed4504add17_b.jpg)

然后移除droupout层和fc8层，并新增一系列的卷积层，在检测数据集上做finetuning。

其中VGG16中的Conv4_3层的输出作为检测的第一张特征图。Conv4_3的特征图大小为$38\times38$，因为该层比较靠近底层，其norm较大，所以在后面新增一个L2Norm层，来保证和后面的检测层差异不大。其与BN不同，仅仅对每个像素点在channel维度上做归一化，而BN则是在[batch_size, width, height]三个维度上做归一化。归一化后一般设置一个可训练的放缩变量gamma。TF实现如下：

```python
# l2norm (not bacth norm, spatial normalization)
def l2norm(x, scale, trainable=True, scope="L2Normalization"):
    n_channels = x.get_shape().as_list()[-1]
    l2_norm = tf.nn.l2_normalize(x, [3], epsilon=1e-12)
    with tf.variable_scope(scope):
        gamma = tf.get_variable("gamma", shape=[n_channels, ], dtype=tf.float32,
                                initializer=tf.constant_initializer(scale),
                                trainable=trainable)
        return l2_norm * gamma
```

从新增的卷积层中提取Conv7, Conv8_2, Conv9_2, Conv10_2, Conv11_2作为检测所用的特征图，加上之前提到的Conv4_3，总共6个特征图。其大小分别为$(38,38),(19,19),(10,10),(5,5),(3,3),(1,1)$，在同一张特征图上grid之间的anchor数量相等，不同的特征图设置的anchor数量不同。

### anchor的设置：

包括尺度和长宽比的设置。尺度上，遵循线性递增：随着特征图的大小降低，anchor尺度线性增加：
$$
s_k = s_{min} + \frac{s_{max} - s_{min}}{m-1}\times (k - 1), k\in [1,m]
$$
其中：

1. $m$指的是特征图的个数，但代码中是5，因为最大的一层，即Conv4_3，是单独设置的；
2. $s_k$是先验框大小相对于图片的比例；3.
3. $s_{max}$和$s_{min}$表示比例的最大值和最小值，分别取0.2，0.9；

对于第一个特征图，anchor尺度比例一般设置为$\frac{s_{min}}{2}=0.1$，那么尺度为300 * 0.1 = 30。对于后面的特征图，anchor的尺度按照上式线性增加，但是先将尺度比例扩大100倍，此时增长步长为：
$$
\lfloor\frac{\lfloor s_{max}\times 100\rfloor - \lfloor s_{min}\times 100\rfloor}{m-1}\rfloor=17
$$
各个特征图的尺度为60，111，162，213，264。

综上，各个特征图的anchor尺度30，60，111，162，213，264。

对于长宽比，一般选取$a_r\in \{1,2,3,\frac{1}{2},\frac{1}{3}\}$，对于特定的长宽比，按下列公式计算：
$$
w_k^a = s_k\sqrt{a_r},\\
h_k^a = \frac{s_k}{\sqrt{a_r}}
$$
**Note**：这里的$s_k$为实际的尺度。

默认情况下，每个特征图会有一个$a_r = 1$且尺度为$s_k$的anchor，除此以外，还会设置一个尺度为$s_k'=\sqrt{s_ks_{k+1}}$且$a_r=1$的anchor，这样每个特征图都设置了两个长宽比为1但大小不同的正方形anchor。

**Note:** 最后一个特征图需要参考一个虚拟$s_{m+1}=300\times\frac{105}{100}=315$来计算$s_m'$。因此，每个特征图一共有6个anchor:$\{1,2,3,\frac{1}{2},\frac{1}{3},1'\}$，但在实现时，Conv4_3，Conv10_2和Conv11_2层仅使用4个anchor，除去了长宽比为3，和1/3的anchor。每个grid的anchor中心分布在各个grid的中心，即：
$$
(\frac{i + 0.5}{|f_k|}, \frac{j+0.5}{|f_k|}),i,j\in [0,|f_k|)
$$
其中，$|f_k|$为特征图的大小。

得到特征图后，需要对特征图进行卷积得到检测结果，下图为一个$5\times 5$大小的特征图的检测过程：

![img](https://pic1.zhimg.com/v2-2d20292e51ef0ce4008ef8a1cf860030_b.jpg)

其中：

1. Priorbox（第一个）是得到anchor box；
2. 检测值包含两部分，类别置信度和边界框位置，令$n_k$为该特征图所采用的anchor数量，那么：
   1. 类别置信度需要的卷积核数量为$n_k\times c$，这里的c应该是总数量；
   2. 边界框位置需要的卷积核数量为$n_k\times 4$。

由于每个anchor都输出一个bbox，所以SSD300一共输出了：
$$
38\times 38\times 4 + 19\times 19\times 6 + 10\times 10\times 6 + 5\times 5\times 6 + 3\times 3\times 4 + 1\times 1\times4 = 8732
$$
数量巨大，所以SSD本质上是密集采样。

## 训练过程

### Anchor匹配

在训练时，首先要确认的是训练图片中的GT与哪个anchor匹配，而匹配上的anchor所对应的bbox负责预测其位置和类别。

1. 在YOLO中，GT中心点落在哪个grid，就由这个grid上的IOU最大的anchor负责预测；

2. 而在SSD中，anchor与GT的匹配原则：

   1. 首先，对每个GT，找到与其IOU最大的anchor box，则将该anchor与GT匹配。这样保证了每个GT与某一个anchor匹配，这样与GT匹配的anchor称为正样本（其实是anchor对应的用来预测bbox），反之，如果一个anchor没有与GT进行匹配，那么该anchor只能与背景匹配，成为负样本。

      由于数据集中的图片上的GT相对而言少很多，如果仅仅按照这一原则，那么会导致正负样本极其不平衡，所以有下一个原则-》

   2. 对于剩余的未匹配的anchor，如果与某个GT的IoU大于某个threshold，那么该anchor也与GT相匹配：

      1. 一个GT可对应多个anchor；
      2. 但是一个anchor只能对应一个GT，解决方法为：如果一个anchor于多个GT的IoU都大于threshold，那么该anchor选择IoU最大的GT做匹配。

   原则2一定在原则1之后考虑，因为如果某个GT对应最大IOU小于threshold，但所匹配的anchor反而与另外一个GT的IOU大于thres，那么就会带来矛盾。所以优先运行原则1，GT优先。

   然而，这一情况很少发生，因为由于anchor很多，对某个GT能重合的anchor，两者的IOU很有可能大于threshold**（我这里存怀疑态度）**。所以只需要实施第二个原则。

   下图为匹配示意图，其中绿色的GT是ground truth，红色为先验框，FP表示负样本，TP表示正样本。

![img](https://pic3.zhimg.com/v2-174bec5acd695bacbdaa051b730f998a_b.jpg)

尽管如此，负样本还是相对多很多。因此SSD采用hard negative mining，对负样本按照**置信度误差**进行降序排序，即预测背景的置信度越小，误差越大，即选取最难的样本用来作为负样本进行训练，因为容易和物体混淆。然后选取误差最大的k个负样本用来训练，使得正负样本的比例接近1:3。

### 损失函数

SSD的损失函数定义为定位损失（locatization loss，loc）与置信度误差（confidence loss，conf）的加权和：
$$
L(x,c,l,g)=\frac{1}{N}(L_{conf}(x,c) + \alpha L_{loc}(x,l,g))
$$
其中：

1. $N$是anchor的正样本数量；
2. $x_{ij}^p\in \{1,0\}$为指示参数：其为1时，表示第$i$个anchor与第$j$个GT匹配，并且GT的类别为$p$；
3. $l$为anchor的所对应bbox的位置预测值；
4. $g$是GT的位置参数。

**对于定位损失：**SSD采用smooth L1 loss：
$$
L_{loc}(x,l,g)=\sum_{i\in Pos}^N \sum_{m\in\{cx,cy,w,h\}} x_{ij}^k \cdot smooth_{L1}\\

\hat{g}_j^{cx}= \frac{(g^{cx}_j - d^{cx}_i)}{d^w_i} \\

\hat{g}_j^{cy}= \frac{(g^{cy}_j - d^{cy}_i)}{d^h_i} \\

\hat{g}_i^w = \log(\frac{g^w_j}{d^w_i}) \\

\hat{g}_i^h = \log(\frac{g^h_j}{d^h_i}) \\

smooth_{L1}(x) = 
\begin{cases}
0.5x^2 & \text{if} \ |x| < 1\\
|x| - 0.5 & \text{otherwise}
\end{cases}
$$
由于$x_{ij}^p$的存在，定位误差只对正样本进行计算。

**Note:** $\hat{g}$为GT编码后的值，因为预测值$l$也是编码值，若设置variance_encoded_in_target=True，编码时要加上variance：
$$
\hat{g}_j^{cx}= \frac{(g^{cx}_j - d^{cx}_i)}{d^w_i\times variance[0]} \\

\hat{g}_j^{cy}= \frac{(g^{cy}_j - d^{cy}_i)}{d^h_i \times variance[1]} \\

\hat{g}_i^w = \frac{\log(\frac{g^w_j}{d^w_i})}{variance[2]} \\

\hat{g}_i^h = \frac{\log(\frac{g^h_j}{d^h_i})}{variance[3]} \\
$$
对于置信度误差，其采用softmax loss：
$$
L_{conf}(x, c) = -\sum_{i\in Pos}^N x_{ij}^p \log(\hat{c}^p_i) - \sum_{i\in Neg} \log(\hat{c}^0_i),\\
\text{where } \hat{c}^p_i = \frac{\exp(c^p_i)}{\sum_p\exp(c^p_i)}
$$
权重系数$\alpha$通过交叉验证，设置为1。



### 数据增强

SSD采用的数据增强有：

1. 水平翻转（horizontal flip）
2. 随机裁剪+颜色扭曲（random crop + color distortion）
3. 随机采集块域（randomly sample a patch）

示意图如下：

<img src="https://pic1.zhimg.com/v2-616b035d839ec419ad604409d4abf6b0_b.jpg" alt="img" style="zoom: 67%;" />

<img src="https://pic1.zhimg.com/v2-616b035d839ec419ad604409d4abf6b0_b.jpg" alt="img" style="zoom:67%;"  />

## 预测过程

预测过程比较简单，对于每个预测框：

1. 首先根据类别置信度确定其类别（选择置信度最大的类别）与置信度值，并过滤掉属于背景的预测框；
2. 然后根据置信度阈值，来过滤掉阈值较低的预测框。
3. 对于剩下的预测框进行解码，根据先验框得到真正的位置参数；
4. 根据置信度进行降序排列，仅保留top-k个预测框。
5. 采用NMS算法，过滤掉那些重叠度较大的预测框。
6. 输出最终结果。

## 实验结果

### VOC2007，VOC2012，COCO

![img](https://pic4.zhimg.com/v2-295e1a51086205b89ef46e6847a29357_b.jpg)

### 对比

![img](https://pic1.zhimg.com/v2-09cf380f4fc0572c12acfd5f1372d310_b.jpg)

### Trick分析

1. 数据增强对mAP的提升很大
2. 使用不同长宽比的anchor可以得到更好的结果

![img](https://pic4.zhimg.com/v2-1f12aabd1577fb996cc471814f68b673_b.jpg)

多尺度提升也很大

![img](https://pic1.zhimg.com/v2-a4134f06b5d924067e3d262bc5d290d0_b.jpg)

## 总结：

SSD在YOLOv1的基础上作出了三种改进：

1. 多尺度特征图
2. 利用卷积进行检测
3. anchor

