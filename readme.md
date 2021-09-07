训练命令：
python train_feat_all.py --config configs/zwh_hard.yaml --gpu 1 --tag 0907_1

目前的主流方法是取bag内最置信的clip（即异常视频中异常分数最大的clip）来构造损失函数，这种做法无法充分利用异常视频中的正常帧以及难以区分的异常帧。在进行训练时，异常视频中的正常帧和难以区分的异常帧其实是非常有助于异常的检测。在此把异常分数值在0.3～0.7范围内的样本称为困难样本，因为这种样本很难被准确地分类。**本文的重点是要围绕如何充分利用异常视频中的正常帧以及困难样本，借此来更好地区分容易分错的视频帧，从而能够提高检测精度**。由此对模型进行修改凝练：

1、去掉memory模块，因为不是重点；

2、重点在于Fusion模块：

Fusion模块的作用为挖掘视频中的困难样本，以此来更好地训练attention模块以及分类器网络。从下面几个方面对困难样本进行构建：

1、如前所述，把异常视频中异常分数值在0.3～0.7范围内的样本称为困难样本，此时从光流的角度将其划分为异常或者正常帧；

2、把正常视频和异常视频中在flow角度与其他帧差异较大的一些视频帧作为困难样本；

3、每个batch从不同的视频片段中挑选出一些帧，与当前视频帧进行mixup(这里还没有mixup，只是简单的concat)人工构建困难样本。

具体实施：

一个batch输入ref-frame，nor-frame，abn-frame三个不同的视频。

当ref frame为normal时

 对于1，挑选出与rgb_score与0.5最接近的一些帧作为hard sample，然后从flow_score给其分类。对于2，在视频内部从光流角度计算视频帧之间的相似性，挑出K个运动模式差异大的ref frame。对于3，充分利用nor-frame和abn-frame中的置信度高的正常帧组成new-nor-frame。将K个ref frame和new-nor-frame送入fusion模块中进行fusion获得混合的正常帧fusion-nor-frame。将fusion-nor-frame送入attention模块中进行进一步混合得到最终特征，以标签0构建BCE损失函数。同时也充分利用abn-frame中的置信度高的异常帧的特征，与fusion-nor-frame构成rank loss，拉远二者之间的距离。具体看代码

通过这个方式既用到了异常视频中置信度不高的困难样本，又用到了置信度高的正常和异常帧。

当ref frame为abnormal时，差不多，具体看代码。



todo：

fusion模块尝试mixup，即将特征进行相加，而不是concat，不知道效果如何

添加rank loss，拉大异常帧与正常帧的特征差异

添加光流特征，目前都是把rgb特征当作光流特征传入函数的。以及弄清楚RTFM的特征是怎么来的。

前面两个都好做，主要问题在于要把精度提上去，目前最好精度为96.6，加上了RTFM的attention模块（attention需要放到fusion模块的前面，代码注释有说明）。第三个需要看情况找找合适的光流特征。



