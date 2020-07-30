# voxsrc2020

- 提供一些案例 Notebook
- 提供整合的工具包

## base.py

文件说明：提供高度抽象的功能

1. 增加数据增益：修改 `loadWAV` 方法的功能

2. 模型设计：自定义 `net` 与 `top` 结构，定义方法参考 `__main__` 中的如下两行：

   ```python
   # 定义说话人嵌入提取模型
   net = ResNetSE34L(nOut=512, num_filters=[16, 32, 64, 128])
   # 定义顶层分类器模型
   top = AMSoftmax(in_feats=512, n_classes=5994, m=0.2, s=30)s
   ```

3. 部分数据集测试：替换 `trainlst` 即可

