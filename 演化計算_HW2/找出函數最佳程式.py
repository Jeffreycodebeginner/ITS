import numpy as np
import matplotlib.pyplot as plt

CHROM_LEN = 10

def decimal_to_x(P_decimal):
    return -10 + (20 / (2**CHROM_LEN - 1)) * P_decimal

# 全域最佳解計算
P_dec = np.arange(2**CHROM_LEN)
x_all = decimal_to_x(P_dec)
y_all = -15 * (np.sin(2*x_all))**2 - (x_all-2)**2 + 160
best_idx = np.argmax(y_all)
best_x = x_all[best_idx]
best_y = y_all[best_idx]
print(f"全域最佳解 x = {best_x:.4f}, y = {best_y:.4f}")

# 繪圖
plt.figure(figsize=(6, 4))
plt.plot(x_all, y_all, label='f(x)')
plt.scatter(best_x, best_y, label=f'Global Max at x={best_x:.3f}, f(x)={best_y:.3f}')
plt.title('f(x) ')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# 1. 建立布林遮罩 (boolean mask)
mask = (x_all > 2.5) & (x_all < 4.0)   # 這裡同時滿足大於2.5且小於4

# 2. 在遮罩區間內找出對應的 y 子陣列
y_sub = y_all[mask]

# 3. 找到子陣列中的最大值及其在子陣列中的索引
sub_max_idx = np.argmax(y_sub)        # argmax：回傳最大值的索引
sub_max_y   = y_sub[sub_max_idx]

# 4. 將子陣列的索引轉回原陣列的索引
orig_indices = np.where(mask)[0]      # where：回傳 True 元素的原始索引陣列
orig_max_idx = orig_indices[sub_max_idx]

# 5. 取得對應的 x 值
sub_max_x = x_all[orig_max_idx]

print(f"在 2.5 < x < 4 區間內，最大值出現在 x = {sub_max_x:.4f}, y = {sub_max_y:.4f}")



# 複製 y_all，並把全域最大值（global maximum）的位置設為 −∞，避免再被選中
y_copy = y_all.copy()
y_copy[best_idx] = -np.inf             # best_idx 是你先前算出的全域最大值位置 (global max index)

# 在更新後的 y_copy 上直接找 argmax
second_idx = np.argmax(y_copy)         # 這次找到的就是第二大值的位置
second_x   = x_all[second_idx]
second_y   = y_all[second_idx]

print(f"第二大值在 x = {second_x:.4f}, y = {second_y:.4f}")