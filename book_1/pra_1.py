#簡単な分類の仕組み

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data_1 = {'weight': [3.0, 1.0], 'height': [1.0, 3.0], 'target': ['A', 'B']}
df = pd.DataFrame(data_1)

plt.scatter(df['weight'], df['height'])
plt.xlim(0, 5)
plt.ylim(0, 5)
plt.title('plot')
plt.show()

a = 0.25
b = 0
#minからmaxまでの値を100個生成(numpy配列)
x = np.linspace(0, 5, 100)
y = a * x + b

plt.scatter(df['weight'], df['height'])
plt.plot(x, y, color='red')
plt.xlim(0, 5)
plt.ylim(0, 5)
plt.title('line: y = 0.25x')
plt.show()

#AとBが分類できていない
#目標：Aとの誤差 x=3.0のときy=1.1
tar_x = 3.0
er = 1.1 - (a * tar_x + b)

#誤差を最小化するための線形関数
_a = 0
t = (a + _a) * tar_x + b
# er = t - yよって、er == _a * x となる
_a = er / tar_x
new_a = a + _a

y = (new_a) * x + b
plt.scatter(df['weight'], df['height'])
plt.plot(x, y, color='red')
plt.xlim(0, 5)
plt.ylim(0, 5)
plt.title('learned line_x')
plt.show()

#Bとの誤差　
# 目標：x=1.0のときy=2.9
tar_x = 1.0
er = 2.9 - (new_a * tar_x + b)
print(er)
_a = er / tar_x
new_a = a + _a

y = (new_a) * x + b
plt.scatter(df['weight'], df['height'])
plt.plot(x, y, color='red')
plt.xlim(0, 5)
plt.ylim(0, 5)
plt.title('learned line_y')
plt.show()

#更新を和らげる
#Lは学習率
L = 0.5
a = 0.25

#Lを入れてみて同じことを試してみる
y = a * x + b

plt.scatter(df['weight'], df['height'])
plt.xlim(0, 5)
plt.ylim(0, 5)
plt.plot(x, y, color='red')
plt.title('line: y = 0.25x')
plt.show()

tar_x = 3.0
er = 1.1 - (a * tar_x + b)
_a = 0
t = (a + _a) * tar_x + b
# er = t - yよって、er == _a * x となる
#更新をLで和らげる
_a = L * (er / tar_x)
new_a = a + _a
print(new_a)

y = (new_a) * x + b
plt.scatter(df['weight'], df['height'])
plt.plot(x, y, color='red')
plt.xlim(0, 5)
plt.ylim(0, 5)
plt.title('learned 1st')
plt.show()

tar_x = 1.0
er = 2.9 - (new_a * tar_x + b)
print(er)
_a = L * (er / tar_x)
new_a = new_a + _a
print(new_a)

y = (new_a) * x + b
plt.scatter(df['weight'], df['height'])
plt.plot(x, y, color='red')
plt.xlim(0, 5)
plt.ylim(0, 5)
plt.title('learned 2nd')
plt.show()