import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# データの入力を辞書形式で扱えるように修正
data_str = input('(dict形式)例{x: [3.0, 1.0], y: [1.0, 3.0], target: ["A", "B"]} : ')
data = eval(data_str)  # evalを使用して文字列を辞書に変換する
learn = float(input('学習率 : '))
learn_count = int(input('学習回数 : '))
df = pd.DataFrame(data)

# 初期データのプロット
plt.scatter(df['x'], df['y'])
plt.xlim(0, 5)
plt.ylim(0, 5)
plt.title('Initial Data')
plt.show()

# パラメータの初期化
a = 0.25
b = 0.0
x = np.linspace(0, 5, 100)
y = a * x + b

#学習
for i in range(learn_count):
    for j in range(len(df)):
        tar_x = df['x'][j]
        tar_y = df['y'][j]
        target = df['target'][j]
        if target == 'A':
            tar_y += 0.1
        else:
            tar_y -= 0.1
        er = tar_y - (a * tar_x + b)
        _a = er / tar_x
        a += learn * _a

# 学習後のプロット
y = a * x + b
plt.scatter(df['x'], df['y'])
plt.plot(x, y, color='red')
plt.xlim(0, 5)
plt.ylim(0, 5)
plt.title('Learned Line')
plt.show()
print(a)