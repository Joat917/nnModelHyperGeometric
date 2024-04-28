这个项目旨在构建一个神经网络，来拟合如下函数：
₂F₁(a,b,c;x), a从-5到-1，b从0到5，c从1到7，x从-0.9到0.9。

此外，由于输入的数据没有任何误差，所以无需担心过拟合发生，不需要测试集。

model1: 
4-15-15-15-15-10-10-1
上面的是各层的神经元数，神经元全部使用Sigmoid函数。


文件说明：
dataManager：生成和管理训练所需的数据。
functionShow：展示我们所拟合的函数的样子。由于这是一个四元函数，所以只能在给定a和b的情况下展示它在三维空间中的切片。
params：训练数据。
train_model1: 训练模型，输出日志到./model-log1中，并把模型保存在pth文件中。
show_model1: 展示模型并和真值进行比对。由于这是一个四元函数，所以只能在给定a和b的情况下展示它在三维空间中的切片。
StartCode.cmd: 打开VSCode。
startTensorBoard-model1.cmd: 运行tensorboard并打开浏览器。
readme: 这是readme: 这是readme: 这是readme: 这是readme: 这是readme: 这是readme: 这是readme: 这是readme: 这是readme: 这是readme: 这是readme: 这是readme: 这是readme: Traceback (most recent call last): 
  File "readme.txt", line 16, in <module>
StackOverflow


训练日志：

数据量512，轮数5000
使用Adam，初始学习率0.05，最终损失函数在558左右，学不动了。
使用Adam，初始学习率0.005，最终损失函数在0.2左右，偶尔出现一个小尖峰。拟合效果不错。

加大数据量到1024，最终损失函数在0.2左右。
加大数据量到4096，最终损失函数在100左右，但是问题是训练完这个数据集耗费的时间高达10分钟。虽然损失函数很大，但是可以明显看出这个模型比其它模型都要好。


