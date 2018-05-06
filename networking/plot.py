#将训练结果绘制出来
import matplotlib.pyplot as plt
import sys
import re

filename = 'plot.txt'
f = open(filename,'r')
y = []

while 1:
    line = f.readline()
    if not line:
        break
    x = line.split()[2]
    x1 = re.findall('\d',x)
    x2 = int(x1[0] + x1[1] + x1[2] + x1[3])
    y.append(x2)
    if not line:
        break
f.close()
x = list(range(1,70))
print(x)
print(y)
#设置横纵坐标轴的范围  
plt.xlabel("x")  
plt.ylabel("y")  
#设置标题  
plt.title("mytitle")  
plt.legend()  
#画网格线，默认横纵坐标轴都画，grid(axis="y")表示只画y轴  
plt.grid()  
plt.plot(x,y,label="mylabel")
plt.show()