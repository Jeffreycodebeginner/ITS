import numpy as np
import matplotlib.pyplot as plt

def f(x):
    y = -15 * (np.sin(2*x))**2 - (x-2)**2 + 160
    return y

# 產生 x 的範圍
x = np.arange(-10, 10, 0.01)

# 計算對應的 y 值
y = f(x)

# 找出最大值和對應的 x
max_index = np.argmax(y)    # 找最大 y 的索引
y_max = y[max_index]        # 最大的 y
x_max = x[max_index]        # 對應的 x

print("最大值 y_max =", y_max)
print("對應的 x =", x_max)

# 畫圖
plt.plot(x, y)
plt.scatter(x_max, y_max, color='red', label=f'Max at x={x_max:.3f}')  # 標出最大點
plt.title('Plot of f(x)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()



