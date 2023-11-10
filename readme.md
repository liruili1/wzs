# Readme preprocess
##                   preprocess-> 2dchuli-img  -mask->pixel_check->cls_pro->训练测试集->train_net
### 五步 先预处理 再转换2dimg mask 再检查像素 再处理分类标签 再分离训练测试集（用处理标签后的文件夹）  再训练
### 分割  先运行seg_pro.py 分出数据集来以后  再分离训练测试  再用seg_train.py 训练  现在已经能训练了 但是结果不好 你自己调调超参数试试， 网络是seg_net.py文件夹
