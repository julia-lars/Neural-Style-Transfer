# Neural-Style-Transfer

## lapstyle环境配置&运行
conda create -n lap_style python=3.6
conda activate lap_style
pip install tensorflow==1.15 pillow numpy scipy

此外还需要下载vgg19.npy的模型权重放入lapstyle文件夹中。地址如下：
https://disk.pku.edu.cn/link/AAA5896F3C7E0B41C7869E516079341FFE
文件夹名：vgg
有效期限：2027-02-01 17:31
提取码：GXLf

python lapstyle.py