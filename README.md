# Introduction  
**diff_tf1.x_tf2.x_pytorch** means ***Differ Among TensorFlow 1.x, TensorFlow 2.x, and Pytorch***.  
Previously, I struggled to summarize some differences among these three frameworks in words. But this is really hard for me to make a list about it, as I have decided to only use **Pytorch**. So you see, the ***lenet5_pytorch.py*** is more complete.  
Well, in ***github.com*** you must be a coder (or programmer sounds nicer?), hence you'd better read the code lines and find the differences by yourself. (This is the best explaination I can come up with to hidden my laziness. LOL.)  
# Details  
Folder ***tensorflow1.x*** is an old-old-old LeNet5_MNIST project based on TensorFlow 1.9.  
````bash  
cd tensorflow1.x  
python3 lenet5_backward.py  
python3 lenet5_test.py  
````  
Script ***lenet5_tensorflow2.x.py*** is based on TensorFlow 2.4.  
````bash  
python3 lenet5_tensorflow2.x.py  
````  
Script ***lenet5_pytorch.py*** is based on Pytorch 1.8. This one is more complete, maybe you can learn something about **argparse** and **tqdm**.  
````bash  
python3 lenet5_pytorch.py --epochs 10 --batch-size 64 --device 0,1  
````  
# Requirements  
Try  
````bash  
pip install -r requirements.txt  
````  
If error occurs, try to fix it by yourself.  
In folder ***tensorflow1.x***, there is another ***requirements.txt*** due to different version of TensorFlow. Uninstall the TensorFlow 2.4 first if you want to try ***tensorflow1.x***.  
# About me  
Just call me **Al** (not ai but al. LOL.) / Albert / Ling Feng (in Chinese, pronounces like ***lin-phone***).  
Rabbit is my favorite animal.  
E-mail: ling@stu.pku.edu.cn  
Gitee: https://gitee.com/lingff  
CSDN: https://blog.csdn.net/weixin_43214408  
