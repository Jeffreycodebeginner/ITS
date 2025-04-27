import numpy as np

# 生成所有1024個點的 x 值
CHROM_LEN = 10
P_decimal_all = np.arange(2**CHROM_LEN)  # 0 到 1023
x_all = -10 + (20 / (2**CHROM_LEN - 1)) * P_decimal_all

# 目標函數
def target_function(x):
    return -15 * (np.sin(2*x))**2 - (x - 2)**2 + 160

# 計算所有點的函數值
y_all = target_function(x_all)

# 找出最佳解
best_idx = np.argmax(y_all)
best_x = x_all[best_idx]
best_y = y_all[best_idx]

# 印出結果
print(f"在1024個離散點中，最大值出現在索引 {best_idx}")
print(f"對應的 x = {best_x:.6f}")
print(f"對應的 f(x) = {best_y:.6f}")

# 顯示前 5 名最佳點
top5_indices = np.argsort(y_all)[-5:][::-1]
print("\n前五名最佳點：")
for idx in top5_indices:
    print(f" 索引 {idx:>4d} | x = {x_all[idx]:>8.6f} | f(x) = {y_all[idx]:>8.6f}")
