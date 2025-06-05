import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. 資料生成
# -----------------------------
np.random.seed(42)
a = 1
N = 400
X = np.random.uniform(-0.8, 0.7, size=N)
Y = np.random.uniform(-0.8, 0.7, size=N)
random_data = np.stack([X, Y], axis=1)  # shape = (400, 2)

# -----------------------------
# 2. 資料切分
# -----------------------------
Train_X = random_data[:300]  # (300, 2)
Test_X  = random_data[300:]  # (100, 2)

# -----------------------------
# 3. 目標函數與正／反正規化
# -----------------------------
def target_function(X, Y):
    return 5 * np.sin(np.pi * X**2) * np.sin(2 * np.pi * Y) + 1

def normalize(Z, z_min=-4, z_max=6, low=0.2, high=0.8):
    return (Z - z_min) / (z_max - z_min) * (high - low) + low

def denormalize(Z_norm, z_min=-4, z_max=6, low=0.2, high=0.8):
    return (Z_norm - low) / (high - low) * (z_max - z_min) + z_min

# 計算並正規化目標值
dZ = target_function(X, Y)     # (400,)
dZ = normalize(dZ)             # (400,)

# 拆分為訓練／測試集
dZ_train = dZ[:300].reshape(-1, 1)  # (300,1)
dZ_test  = dZ[300:].reshape(-1, 1)  # (100,1)

# -----------------------------
# 4. 初始化權重（不含 bias）
# -----------------------------
np.random.seed(42)
h_num = 100
W1 = np.random.randn(h_num, 2)   # (隱藏層神經元數,2)
W2 = np.random.randn(1, h_num)   # (1,隱藏層神經元數)

def phi(x):
    return 1 / (1 + np.exp(-a * x))

def phi_derieve(x):
    s = phi(x)
    return s * (1 - s)

# -----------------------------
# 5. SGD + Batch‐Mode 參數
# -----------------------------
epochs = 50000
eta = 0.01         # 基本學習率
E_list = []        # 紀錄每個 epoch 的 MSE

# -----------------------------
# 6. 訓練迴圈 (必須在計算 dW2, dW1 前先累加、然後「除以 300」)
# -----------------------------
for epoch in range(1, epochs + 1):
    # 每 10 個 epoch 做一次 learning‐rate 衰減
    lr_t = eta * (0.95 ** (epoch // 100000))
    lr_t = eta

    # ----- Forward -----
    V1 = Train_X @ W1.T       # 隱藏層加權和
    Y1 = phi(V1)              # 隱藏層激活輸出

    V2 = Y1 @ W2.T            # 輸出層加權和
    Y2 = phi(V2)              # 最終預測 (正規化後)

    # 計算誤差 e(n) = target(n) - prediction(n)
    e = dZ_train - Y2         # (300, 1)

    # ----- Backward -----
    
    delta2 = e * phi_derieve(V2)
    delta1 = phi_derieve(V1) * (delta2 @ W2)  # (300,10)

    # 計算「未平均的」梯度累加：
    grad_W2_accum = (delta2.T @ Y1)      #   sum_{n=1..300} 
    grad_W1_accum = (delta1.T @ Train_X) #   sum_{n=1..300}

    #Batch‐Mode 要先「累加全部樣本」再「除以 300」**
    dW2 = (grad_W2_accum / 300.0)  # 「平均梯度」
    dW1 = (grad_W1_accum / 300.0)  

    # 最後再乘上學習率 lr_t 來更新
    W2 += lr_t * dW2
    W1 += lr_t * dW1

    # 記錄當前平均 MSE
    E = 0.5 * np.mean(e**2) 
    E_list.append(E)

# -----------------------------
# 7. 定義預測函式
# -----------------------------
def predict_system(X_input):
    V1 = X_input @ W1.T  
    Y1 = phi(V1)         
    V2 = Y1 @ W2.T       
    Y2 = phi(V2)       
    return Y2

# -----------------------------
# 8. 在訓練／測試集上預測並還原
# -----------------------------
Z_pred_train = predict_system(Train_X)  # (300,1)
Z_pred_test  = predict_system(Test_X)   # (100,1)

# 把預測值從 [0.2,0.8] 反正規化回 [-4,6]
Z_pred_train_R = denormalize(Z_pred_train)  # (300,1)
Z_pred_test_R  = denormalize(Z_pred_test)   # (100,1)

# 也把真實值還原
dZ_train_R = denormalize(dZ_train)  # (300,1)
dZ_test_R  = denormalize(dZ_test)   # (100,1)

# 計算還原後的 MSE 作為最終評估
E_train = 0.5 * np.mean((dZ_train_R - Z_pred_train_R)**2)
E_test  = 0.5 * np.mean((dZ_test_R - Z_pred_test_R)**2)
print(f"E_train: {E_train:.4f}")
print(f"E_test:  {E_test:.4f}")

# -----------------------------
# 9. 繪製訓練誤差收斂曲線
# -----------------------------
plt.figure(figsize=(8, 5))
plt.plot(E_list)
plt.xlabel("Epoch (n)")
plt.ylabel("Average Error E")
plt.title("Training Error Curve (Batch SGD)")
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# 10. 3D 曲面圖：真實 vs. 預測
# -----------------------------
from mpl_toolkits.mplot3d import Axes3D

draw_x, draw_y = np.meshgrid( 
    np.linspace(-0.8, 0.7, 100), 
    np.linspace(-0.8, 0.7, 100)
)
draw_x_flat = draw_x.ravel().reshape(-1, 1)  # (10000, 1)
draw_y_flat = draw_y.ravel().reshape(-1, 1)  # (10000, 1)
grid_input  = np.hstack([draw_x_flat, draw_y_flat])  # (10000, 2)

# 真實曲面
draw_z_true = target_function(draw_x_flat, draw_y_flat).reshape(100, 100)

# 神經網路輸出（正規化後），再反正規化
draw_z_pred_norm = predict_system(grid_input)         # (10000,1)
draw_z_pred = denormalize(draw_z_pred_norm).reshape(100, 100)

fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(draw_x, draw_y, draw_z_true, cmap='viridis', edgecolor='none')
ax1.set_title("Target Function (True)")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("F(x, y)")

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(draw_x, draw_y, draw_z_pred, cmap='plasma', edgecolor='none')
ax2.set_title("BPNN Prediction (Batch SGD)")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("F_pred(x, y)")

plt.tight_layout()
plt.show()
