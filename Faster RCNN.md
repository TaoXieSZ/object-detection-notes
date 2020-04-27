# Faster R-CNN

<img src="../image-20200401140941010.png" alt="image-20200401140941010" style="zoom: 25%;" />

![img](https://pic4.zhimg.com/80/v2-e64a99b38f411c337f538eb5f093bdf3_720w.jpg)

如上图，为Faster R-CNN的大致框架。

接下来，要从以下几个部分去理解Faster R-CNN：

## Region Proposal Network(RPN)

在此之前的目标检测框架中，哪怕是Fast RCNN，都需要先计算建议框，Fast RCNN与RCNN采用Selective Search（based on grouping super-pixels），OpenCV adaboost采用滑动窗口+图像金字塔。而两类方法都十分地耗时。因此Faster R-CNN直接采用神经网络来生成建议框。对于任意size的图片输入，RPN输出一组矩形的、带有objectness score的目标建议框。

![img](https://pic3.zhimg.com/80/v2-1908feeaba591d28bee3c4a754cca282_720w.jpg)

如图，上面的分支通过Softmax进行分类，将**anchor**分成positive与negative两类；下面分支计算bbox对**anchor**的offset，而不是直接计算坐标。最后的Proposal块，负责综合positive anchors和对应的offset，生成建议框，同时筛选掉太小或太大的建议框，并输入到ROI pooling层进行下一步的计算。

### Anchors

所谓的anchors，实际上就是一组预先定义的矩形框，在官方源码中的rpn/generate_anchors.py，运行demo可以生成：

```
[[ -84.  -40.   99.   55.]
 [-176.  -88.  191.  103.]
 [-360. -184.  375.  199.]
 [ -56.  -56.   71.   71.]
 [-120. -120.  135.  135.]
 [-248. -248.  263.  263.]
 [ -36.  -80.   51.   95.]
 [ -80. -168.   95.  183.]
 [-168. -344.  183.  359.]]
```

每一行的四个值为$(x_1,y_1,x_2,y_2)$，分别代表矩形左上与右下两点坐标，

此时有个问题，为什么会出现负数呢？

**解答**：anchor的生成步骤为：先生成9个base anchor，然后通过坐标偏移，通过坐标偏移在特征图上采样的每个点上放上这9个anchor，当左上角超出图片边界，坐标就产生了负数的坐标值。因此需要裁剪：

![img](https://pic1.zhimg.com/80/v2-9d67146e0cb10397d8c2170794412608_720w.jpg)

那么回到我们的坐标，9个矩形具有3种形状，长宽比为$w:h\in \{1:1, 1:2, 2:1\}$。

![img](https://pic4.zhimg.com/80/v2-7abead97efcc46a3ee5b030a2151643f_720w.jpg)

**注意：**关于anchor的size，Faster RCNN是根据图片大小，手动设置的。在上述的python demo中，首先任意大小的图片会先被reshape成$800\times600$的大小，而anchors的大小分别为$352\times704$，$736\times384$，基本包含了各个尺度和各个形状的建议框。



**那这些anchors的作用**：对卷积层输出的卷积特征图，为每一stride计算九个以anchor为模板的检测框！

<img src="/Users/chris/Library/Application Support/typora-user-images/image-20200401230356796.png" alt="image-20200401230356796" style="zoom:50%;" />



RPN如上图所示：

1. 在ZFmodel中，Conv5的卷积核个数为256个，因此生成了256张特征图，也可以视为一张特征图的深度为256.
2. 在Conv5之后，做了rpn_conv/$3\times3$卷积，同样num_output为256，因此生成256-d的intermediate layer；
3. 如果k=9，那么每个anchor要被分成positive和nagative，因此每个点由256-d划分为**cls=2 * 9** **scores**；每个anchor还有偏移值，即$(x,y,w,h)$，所以，**reg=4*9 coordinates**；
4. 不能把所有的anchor都拿来训练，而是适当地随机选取128个postive anchors和128个negative anchors，其中，与groundtruth的$IoU>0.7$的，认为是positive，而$IoU<0.3$的，认为是negative，中间的$0.3<IoU<0.7$的均不用来训练。

**可以看出，RPN实质上就是在原图尺寸上，设置了密密麻麻的候选框（anchor box），并用CNN来判断是negative还是positive，是一个二分类问题**。

有多密集呢，看下图：

![img](https://pic2.zhimg.com/80/v2-4b15828dfee19be726835b671748cc4d_720w.jpg)

### Softmax

在RPN中，采用Softmax来进行positive和negative的分类。在softmax之前，有一个$1\times 1$的卷积操作：

![img](https://pic4.zhimg.com/80/v2-1ab4b6c3dd607a5035b5203c76b078f3_720w.jpg)

用来将$w\times h \times 256$的特征图计算得到$w\times h \times (9\times2)$，因为有p和n两个“类别”。而前后的Reshape操作，是为了方便做softmax。



### Bounding Box regression原理

<img src="https://pic4.zhimg.com/80/v2-93021a3c03d66456150efa1da95416d3_720w.jpg" alt="img" style="zoom: 150%;" />

绿色为真实值（GT），红色为生成的positive anchor。对于每个窗口，用一个四元组来表示，$(x,y,w,h)$，表示中心点与框高。训练的过程就是要让红色的anchor box尽可能接近绿色的真实值。如图所示，将A进行转换得到更接近G的G'：

![img](https://pic2.zhimg.com/80/v2-ea7e6e48662bfa68ec73bdf32f36bb85_720w.jpg)

那怎么做呢，对于给定的anchor $(A_X,A_y,A_w,A_h)$和真实框$(G_x,G_y,G_w,G_h)$，我们要找到一个变换$F$，使得$F(A_X,A_y,A_w,A_h)=(G_x',G_y',G_w',G_h')$，其中$(G_x',G_y',G_w',G_h')\approx(G_x,G_y,G_w,G_h)$：

先做平移：
$$
G_x'=A_w\cdot d_x(A) + A_x\\
G_y' = A_h \cdot d_y(A)+ A_y
$$
再做缩放：
$$
G_w' = A_w \cdot exp(d_w(A))\\
G_h' = A_h \cdot exp(D_h(A))
$$
我们需要学习的就是$d_x(A),d_y(A),d_w(A),d_h(A)$这四个变换。

**注意：**当输入的anchor与GT接近，可以认为是线性变换，就可以用线性回归来求解。



问题又来了，那我们要怎么通过线性回归来求解呢？首先，理清楚线性回归的概念，线性回归即给定输入的特征向量$X$，学习一组参数$W$，使得$Y=WX$。

那么在求解anchor的问题中，输入X为CNN特征图，记为$\phi$，同时还有训练传入A与GT的变换量，记为$(t_x,t_y,t_w,t_h)$。输出是$d_x(A),d_y(A),d_w(A),d_h(A)$四个变换矩阵。那么目标函数可以表示为：
$$
d_*(A) = W_*^T\cdot \phi(A)
$$
其中，$\phi(A)$是anchor A对应的特征图中的特征向量，$W_*$是需要学习的参数，$d_*(A)$是预测值。为了使预测值$d_*(A)$与真实值$t_*$接近，作者设计L1损失函数：
$$
Loss = \sum_i^N{|t_*^i-W_*^T\cdot\phi(A_i)|}
$$
目标函数为：
$$
\hat{W_*}=argmin_{W_*}\sum_i^n{|t_*^i-W_*^T\cdot \phi(A)|+\lambda||W_*||}
$$
真实情况是使用smooth-L1 loss（如Fast RCNN）：
$$
L_{loc}(t^u, v) = \sum_{i\in \{x,y,w,h\}} smooth_{L_1}(t^u_i-v_i) \\
smooth_{L_1}(x) = \begin{cases} 0.5 x^2, \quad if \ |x| < 1 \\ |x| -0.5,\quad othersise\end{cases}
$$
**当然，此处假设anchor位置比较接近，才可以使用线性变换的假设**；

在Faster RCNN中，positive anchor与GT的平移量$(t_X,t_y)$与尺度因子$t_w,t_h$为：
$$
t_x=\frac{x-x_a}{w_a} \\
t_y = \frac{y-y_a}{h_a} \\
t_w = log(\frac{w}{w_a})\\
t_h = log(\frac{h}{h_a})
$$
对于训练bounding box regression网络回归分支，输入是Conv feature $\phi$，监督信号是anchor与GT的变换$(t_x,t_y,t_w,t_h)$，即训练目标是：输入$\phi$的情况下，极可能使得输出接近于监督信号。当测试时，网络输出的就是anchor相对于真实值的平移量和变换因子。



### 对proposal进行bounding box regression

在框架中，对bbox的回归分支味：

![img](https://pic3.zhimg.com/80/v2-8241c8076d60156248916fe2f1a5674a_720w.jpg)

首先，源码中，对该$1\times1$的定义为：

```
layer {
  name: "rpn_bbox_pred"
  type: "Convolution"
  bottom: "rpn/output"
  top: "rpn_bbox_pred"
  convolution_param {
    num_output: 36   # 4 * 9(anchors)
    kernel_size: 1 pad: 0 stride: 1
  }
}
```

其中，num_output为36，即该卷积输出的维度为$W\times H\times 36$，然后存储为$[1,4\times 9,H, W]$，表示在该特征图的每个点，有9个anchor，每一个都有4个用来做回归的变换量，即：
$$
[d_x(A),d_y(A),d_w(A),d_h(A)]
$$
在下图中，

![img](https://pic2.zhimg.com/80/v2-4b15828dfee19be726835b671748cc4d_720w.jpg)

我们可以看到，VGG输出$\frac{800}{16}\times \frac{600}{16} \times 512$维度的特征图，对应设置$50\times 38 \times k$个anchor，而RPN输出：

1. 大小为$50\times 38 \times 2k$的positve/negati softmax分类特征矩阵；
2. 大小为$50\times 38 \times 4k$的regression坐标回归特征矩阵。

至此，完成了RPN的分类+回归任务。

### Proposal Layer

​		Proposal layer的作用是综合所有的$[d_x(A),d_y(A),d_w(A),d_h(A)]$变换量和positive anchors，计算出精准的proposal，随后输入到RoI pooling层。首先，看源码中的定义：

```
layer {
  name: 'proposal'
  type: 'Python'
  bottom: 'rpn_cls_prob_reshape'
  bottom: 'rpn_bbox_pred'
  bottom: 'im_info'
  top: 'rois'
  python_param {
    module: 'rpn.proposal_layer'
    layer: 'ProposalLayer'
    param_str: "'feat_stride': 16"
  }
}
```

在该层中，有三个输入以及一个参数：

1. 正类或负类——rpn_cls_prob_reshape；
2. bbox regression——rpn_bbox_pred；
3. im_info
4. feat_stride=16

**im_info**：对于任意大小$P\times Q$的图像，输入到Faster RCNN前已经先被缩放为$M\times N$，那么im_info存储的就是$[M,N,scale\_factor]$，保存着缩放的信息。经过Conv Net时，经过了4层pooling层，因此大小变为$W\times H = \frac{M}{16} \times \frac{N}{16}$，其中feat_stride即feature stride，就保存着这个**16**，用来计算anchor的偏移量。如下图所示：

![img](https://pic2.zhimg.com/80/v2-1e43500c7cc9a9de211d737bc347ced9_720w.jpg)

在Proposal Layer的前向传播中，按照下列顺序进行：

1. 生成anchor，利用$[d_x(A),d_y(A),d_w(A),d_h(A)]$，对所有的anchor进行bbox regression回归；
2. 按照输入的positive softmax scores，对anchor进行排序（递减），按照pre_nms_topN=6000来提取前6000个anchor，即提取了计算了偏移后的positive anchor；
3. 限定超出图像边界的positive anchor为图像边界，即坐标为负数，防止在后续的ROI pooling时，建议框超出了边界；
4. 去掉小于尺寸阈值的positive anchor；
5. 对剩余的positive anchor进行NMS；
6. 输出proposal = $[x_1,y_1,x_2,y_2]$。

此时，可认为检测部分已经结束，接下来属于识别任务。



**对RPN进行总结：**

1. **生成anchor**
2. **softmax提取positive anhor**
3. **bbox reg回归positive anchors**
4. **Proposal Layer生成proposal（建议框）**



## ROI Pooling

​	RoI pooling负责收集PRN输出的建议框，并计算得到建议框相应的特征图，输入到全连接层中进行分类回归。我们要看到，RoI pooling层有两个输入：

1. 原始的特征图

2. RPN输出的建议框

   

   接下来，我们从两个角度分析ROI pooling：

3. 为何需要RoI pooling

4. RoI Pooling的原理

### 为何需要ROI pooling

​	对于当时的CNN，如AlexNet和VGG，当网络训练好之后，测试输入的图片尺寸必须时固定的，同时网络输入也是固定大小的矩阵。如果输入大小不定：有两种方法：

1. 图像裁剪（cropping）
2. 图像弯曲（warpping）



​	而这两种方法都会对原有的图像信息造成损害，（完整结构与原始形状），而RPN网络生成的建议框中，同样有尺寸不一的特点，而RoI Pooling可以解决这一问题。



### 原理

同样的，先对源码进行分析：

```
layer {
  name: "roi_pool5"
  type: "ROIPooling"
  bottom: "conv5_3"
  bottom: "rois"
  top: "pool5"
  roi_pooling_param {
    pooled_w: 7
    pooled_h: 7
    spatial_scale: 0.0625 # 1/16
  }
}
```

其中有两个新的参数，    pooled_w: 7与pooled_h：7。

其前向传播的流程为：

1. 由于建议框是对应了$M\times N$的大小，所以首先根据spatial_scale将其映射回，大小为$\frac{M}{16} \times \frac{N}{16}$的特征图上；
2. 将每个建议框对应的特征图区域水平分为pooled_w * pooled_h数量的网格；
3. 对每个网格进行**max** **pooling**。

这样处理后，即使是在大小不一的建议框，输出都是pooled_w * pooled_h，实现了固定长度的输出！

具体如下图所示：

![img](https://pic1.zhimg.com/80/v2-e3108dc5cdd76b871e21a4cb64001b5c_720w.jpg)



## Classification

在Classification部分，利用已经获得的 **proposal feature map**，通过全连接层与softmax计算每个建议框属于什么物体（类别），输出cls_prob概率向量；同时再次利用bounding box regression获得的每个建议框的偏移值，**bbox_pred**，得到目标检测框。

![img](https://pic2.zhimg.com/80/v2-9377a45dc8393d546b7b52a491414ded_720w.jpg)

全连接层就很熟悉了，示意图如下：

![img](https://pic2.zhimg.com/80/v2-38594a97f33ff56fc72542a20a78116d_720w.jpg)

计算公式如：

![img](https://pic4.zhimg.com/80/v2-f56d3209f9a7d5f27d77ead7489ab70f_720w.jpg)



## Training

在预训练模型上（VGG，ZF，etc），实际上的训练分为以下6个步骤：

1. 在预训练好的model上，先训练RPN网络，对应于**stage1_rpn_train.pt**；
2. 利用训练的RPN，收集好建议框，对应于**rpn_test.pt**；
3. 第一次训练Fast R-CNN，对应于**stage1_fast_rcnn_train.py**；
4. 第二次训练RPN，对应于**state2_rpn_train.py**；
5. 再次计算建议框，对应于**rpn_test.pt**；
6. 第二次训练Fast RCNN，对应**state2_fast_rcnn_train.pt**。



之所以只循环两次，因为作者实验证明，循环次数增加也不会有提升。流程如下图所示：

<img src="https://pic2.zhimg.com/80/v2-ed3148b3b8bc3fbfc433c7af31fe67d5_720w.jpg" alt="img" style="zoom:150%;" />



### 训练RPN网络

​		原论文基于VGG预训练模型，用图片进行训练：

<img src="https://pic2.zhimg.com/80/v2-c39aef1d06e08e4e0cec96b10f50a779_720w.jpg" alt="img" style="zoom:150%;" />

​		整个网络使用的损失函数为：
$$
L(\{p_i\},\{t_i\}) = \frac{1}{N_{cls}}\sum_i{L_{cls}(p_i,p_i^*)} + \lambda\cdot\frac{1}{N_{reg}}\sum_i{p_i^*L_{reg}(t_i,t_i^*)}
$$
其中：

1. $i$ 表示anchor box的index，$p_i$ 表示positive softmax probability，$p_i^*$ 代表是positive或者negative（该anchor如果与ground truth的交互比大于0.7，为positive，$p_i^*=1$；小于0.3，为negative，即$p_i^*=0$；那么在0.3与0.7之间的，则不参与训练）；
2. $t$ 代表predict bounding box，$t^*$代表对应的positive anchor对应的ground truth box。

对损失函数进行整体分析：

1. cls loss：为rpn_cls_loss层计算的softmax loss，用于对anchor分类为positve与negative的网络训练；
2. reg loss：为rpn_loss_bbox层计算的smooth L1 loss，用于对坐标进行回归，但是仅仅对positve anchor进行计算，因为无意义（negative）的anchor也不需要用来计算。



要注意的是，实际上，$N_{cls}$与$N_{reg}$差距往往较大，因此采用平衡因子$\lambda$，使得总的损失函数对两者均有涉及。smooth L1 loss 和Fast RCNN一致：
$$
L_{loc}(t^u, v) = \sum_{i\in \{x,y,w,h\}} smooth_{L_1}(t^u_i-v_i) \\
smooth_{L_1}(x) = \begin{cases} 0.5 x^2, \quad if \ |x| < 1 \\ |x| -0.5,\quad othersise\end{cases}
$$
重看下图：

![img](https://pic2.zhimg.com/80/v2-ed3148b3b8bc3fbfc433c7af31fe67d5_720w.jpg)

我们可以看到：

1. 在RPN训练阶段，rpn-data层会和test阶段一样生成anchor；
2. 对于rpn_loss_cls，输入分别有Ron_cls_score_reshape和Ron_labels分别对应于$p$与$p^*$；
3. 对于rpn_loss_bbox，输入的rpn_bbox_pred和rpn_bbox_targets分别代表$t$与$t^*$，rpn_bbox_inside_weigths对应于$p*$，rpn_bbox_outside_weigths则用不上

**在训练和检测阶段中，生成和存储anchor的顺序需要完全一致。**



### 通过RPN网络收集建议框

利用训练好的RPN网络，获取proposal rois，同时获取positive softmax probability，如下图：

<img src="https://pic1.zhimg.com/80/v2-1ac5f8a2899ee413464ecf7866f8f840_720w.jpg" alt="img"  />

把输出存储在磁盘文件中。

### 训练Faster RCNN网络

读取上一步得到的文件，获取proposal与positive probability。从data层输入网络，然后进行：

1. 将提取得到的proposal作为rois传入网络，如下图中的蓝色框；
2. 计算bbox_inside_weights+bbox_outside_weights，作用与RPN一样，传入到smooth_L1_loss layer，如下图绿色框。

![img](https://pic3.zhimg.com/80/v2-fbece817952865689187e68f0af86792_720w.jpg)

而stage2就大同小异了。



## QA：Anchor与网络输出如何对应

VGG网络输出$50\times 38 \times 512$的特征，对应设置$50\times 38 \times k$个anchors，而RPN输出$50\times 38 \times 2k$的分类特征矩阵和$50\times 38 \times 4k$的坐标回归特征矩阵，对应关系如下图：

<img src="https://pic1.zhimg.com/80/v2-82196feb7b528d76411feb90bfec2af4_720w.jpg" alt="img" style="zoom:150%;" />

