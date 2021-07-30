English Version:
### XPARL QMIX
* QMIX algorithm implementation using PARL framework of PaddlePaddle
* Original algorithm is from PARL official example
* Add XPARL support for training on muitiple CPUs，in theory training speed will be much faster.
### Quick Start
* setup ubuntu
* install PaddlePaddle 2.x+ version
* install PARL 2.x+ version
* install opencv-python
* install SC2
* install smac&Maps
* pull this code
* start xparl using command line: xparl start --port 8010 --cpu_num x
* start training using command line: python train.py
![image]https://github.com/ouyangshixiong/XPARL_QMIX/blob/master/xparl_qmix.jpg

中文版：
### XPARL QMIX算法实现
* 用百度飞桨的PARL框架实现了QMIX算法
* 代码是基于百度飞桨PARL的example目录中的原QMIX算法改进而来
* 实现了基于xparl集群的多CPU训练，理论上可以大大提升训练速度

### 快速开始
* 安装ubuntu
* 安装飞桨2.x以上版本，最好是基于GPU的版本。
* 安装PARL2.x版本
* 安装opencv-python，以支持cv2
* 在本机安装SC2
* 安装smac环境和对应的训练地图，这个按smac的github指导安装和验证即可。
* 拉这份代码到本地
* 先启动xparl集群，开启分布式cpu训练：xparl start --port 8010 --cpu_num x 。其中x为需要启动的cpu数，需要跟实际cpu数目匹配。
* 在命令行执行： python train.py 开启训练。可以同步开启visualdl，查看训练效果。

![image]https://github.com/ouyangshixiong/XPARL_QMIX/blob/master/xparl_qmix.jpg



