import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- 1. 資料生成 ---
np.random.seed(488)   # 固定隨機種子以便重現結果
a = 1
N = 400
X = np.random.uniform(-0.8, 0.7, size=N)
Y = np.random.uniform(-0.8, 0.7, size=N)
random_data = np.stack([X, Y], axis=1)  # shape = (400, 2)

# --- 2. 資料切分 ---
Train_X = random_data[:300]   # (300, 2)
Test_X  = random_data[300:]   # (100, 2)

def target_function(X, Y):
    return 5 * np.sin(np.pi * X**2) * np.sin(2 * np.pi * Y) + 1

def normalize(Z, z_min=-4, z_max=6, low=0.2, high=0.8):
    # 把 Z 從 [z_min, z_max] 線性映射到 [low, high]
    return (Z - z_min) / (z_max - z_min) * (high - low) + low

def denormalize(Z_norm, z_min=-4, z_max=6, low=0.2, high=0.8):
    # 把 Z_norm 從 [low, high] 還原回 [z_min, z_max]
    return (Z_norm - low) / (high - low) * (z_max - z_min) + z_min

# 計算全體目標值並正規化
dZ = target_function(X, Y)          # (400,)
dZ = normalize(dZ)                  # (400,)

dZ_train = dZ[:300].reshape(-1, 1)   # (300, 1)
dZ_test  = dZ[300:].reshape(-1, 1)   # (100, 1)

# --- 3. 初始化權重（不含 bias） ---
np.random.seed(42)
h_num = 5
W1 = np.random.randn(h_num, 2)   # shape = (10, 2)
W2 = np.random.randn(1, h_num)   # shape = (1, 10)

def phi(x):
    return 1 / (1 + np.exp(-a*x))

def phi_derieve(x):
    s = phi(x)
    return s * (1 - s)

# --- 4. Adam 參數初始化 ---
beta1, beta2 = 0.9, 0.999
eps = 1e-8

mW1 = np.zeros_like(W1)
vW1 = np.zeros_like(W1)
mW2 = np.zeros_like(W2)
vW2 = np.zeros_like(W2)

# --- 5. 訓練參數與儲存容器 ---
epochs = 50000
eta = 0.01
E_list = []

# --- 6. Batch‐Mode 訓練（含 Adam 更新） ---
for epoch in range(1, epochs + 1):
    # 每 1000 個 epoch 衰減一次學習率
    lr_t = eta * (0.95 ** (epoch // 1000))

    # --- Forward ---
    V1 = Train_X @ W1.T        # (300, 10)
    Y1 = phi(V1)               # (300, 10)
    V2 = Y1 @ W2.T             # (300, 1)
    Y2 = phi(V2)               # (300, 1)  # 網路最終輸出

    e = dZ_train - Y2          # (300, 1)

    # --- Backward (計算梯度) ---
    delta2 = e * phi_derieve(V2)             # (300, 1)
    delta1 = phi_derieve(V1) * (delta2 @ W2) # (300, 10)

    N_train = Train_X.shape[0]               # 300
    dW2 = (delta2.T @ Y1) / N_train          # (1, 10)
    dW1 = (delta1.T @ Train_X) / N_train     # (10, 2)

    # --- Adam 更新 W2 ---
    mW2 = beta1 * mW2 + (1 - beta1) * dW2
    vW2 = beta2 * vW2 + (1 - beta2) * (dW2 ** 2)
    mW2_hat = mW2 / (1 - beta1 ** epoch)
    vW2_hat = vW2 / (1 - beta2 ** epoch)
    W2 += lr_t * mW2_hat / (np.sqrt(vW2_hat) + eps)

    # --- Adam 更新 W1 ---
    mW1 = beta1 * mW1 + (1 - beta1) * dW1
    vW1 = beta2 * vW1 + (1 - beta2) * (dW1 ** 2)
    mW1_hat = mW1 / (1 - beta1 ** epoch)
    vW1_hat = vW1 / (1 - beta2 ** epoch)
    W1 += lr_t * mW1_hat / (np.sqrt(vW1_hat) + eps)

    # --- 記錄訓練誤差 MSE ---
    E = 0.5 * np.mean(e**2)
    E_list.append(E)

# --- 7. 定義預測函式（Forward）---
def predict_system(X_input):
    V1 = X_input @ W1.T        # (N, 10)
    Y1 = phi(V1)               # (N, 10)
    V2 = Y1 @ W2.T             # (N, 1)
    Y2 = phi(V2)               # (N, 1)
    return Y2

# --- 8. 在訓練與測試集上做預測並還原 ---
Z_pred_train = predict_system(Train_X)  # (300, 1)
Z_pred_test  = predict_system(Test_X)   # (100, 1)

Z_pred_train_R = denormalize(Z_pred_train)  # 還原到 [-4, 6]
Z_pred_test_R  = denormalize(Z_pred_test)

dZ_train_R = denormalize(dZ_train)
dZ_test_R  = denormalize(dZ_test)

E_train = 0.5 * np.mean((dZ_train_R - Z_pred_train_R)**2)
E_test  = 0.5 * np.mean((dZ_test_R  - Z_pred_test_R)**2)

print("E_train:", E_train)
print("E_test: ", E_test)

# --- 9. 繪製訓練誤差收斂曲線 ---
plt.figure(figsize=(8, 5))
plt.plot(E_list)
plt.xlabel("Epoch (n)")
plt.ylabel("Average Error E")
plt.title("Training Error Curve (Adam)")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 10. 3D 曲面圖：真實 vs. Neural Net 預測 ---
draw_x, draw_y = np.meshgrid(
    np.linspace(-0.8, 0.7, 100),
    np.linspace(-0.8, 0.7, 100)
)
draw_x_flat = draw_x.ravel().reshape(-1, 1)    # (10000, 1)
draw_y_flat = draw_y.ravel().reshape(-1, 1)    # (10000, 1)
grid_input  = np.hstack([draw_x_flat, draw_y_flat])  # (10000, 2)

draw_z_true = target_function(draw_x_flat, draw_y_flat).reshape(100, 100)
draw_z_pred_norm = predict_system(grid_input)         # (10000, 1)
draw_z_pred = denormalize(draw_z_pred_norm).reshape(100, 100)

fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(draw_x, draw_y, draw_z_true, cmap='viridis', edgecolor='none')
ax1.set_title("Target Function")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("F(x, y)")

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(draw_x, draw_y, draw_z_pred, cmap='plasma', edgecolor='none')
ax2.set_title("BPNN Prediction (with Adam)")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("F_pred(x, y)")

plt.tight_layout()
plt.show()
