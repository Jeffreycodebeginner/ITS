import numpy as np
import matplotlib.pyplot as plt
# --- 1. 資料生成 ---
a=1
N = 400
X = np.random.uniform(-0.8, 0.7, size=N)
Y = np.random.uniform(-0.8, 0.7, size=N)
random_data = np.stack([X, Y], axis=1)  # shape = (400, 2)

# --- 2. 資料切分 ---
Train_X= random_data[:300]
Test_X  = random_data[300:]

def target_function(X, Y):
    return 5 * np.sin(np.pi * X**2) * np.sin(2 * np.pi * Y) + 1



def normalize(Z, z_min=-4, z_max=6, a=0.2, b=0.8):
    return (Z - z_min) / (z_max - z_min) * (b - a) + a

def denormalize(Z_norm, z_min=-4, z_max=6, a=0.2, b=0.8):
    return (Z_norm - a) / (b - a) * (z_max - z_min) + z_min


dZ = target_function(X, Y)
dZ = normalize(dZ)    #在這邊需要將正規化

dZ_train = dZ[:300]   #訓練集期望輸出
dZ_test  = dZ[300:]   #測試集期望輸出

# --- 3. 初始化權重（可加入種子以便重現） ---
np.random.seed(42)


h_num=5
W1 = np.random.randn(h_num, 2)   # shape = (5, 2)
W2 = np.random.randn(1, h_num)    # shape = (1, 5)

def phi(x):
    return 1 / (1 + np.exp(-a*x))

def phi_derieve(x):
    s = phi(x)
    return s*(1-s)

    
E_list = []
epochs=10000
eta =0.01   #學習率
#-----BPNN演算法
for epoch in range(1, epochs+1):
    eta_t=eta* (0.95 ** (epoch // 1000))
    
    # --- Forward ---
    V1 = Train_X @ W1.T        # (300, 5)
    Y1 = phi(V1)               # (300, 5)
    V2 = Y1 @ W2.T             # (300, 1)
    Y2 = phi(V2)                # (300, 1)  #Y2就是Z(輸出)

    e = dZ_train.reshape(-1, 1) - Y2  # (300, 1)

    # --- Backward ---
    delta2 = e * phi_derieve(V2)        # (300, 1)
    delta1 = phi_derieve(V1) * (delta2 @ W2)  # (300, 5)

    dW2 = eta_t * (delta2.T @ Y1)         # (1, 5)
    dW1 = eta_t * (delta1.T @ Train_X)    # (5, 2)

    # --- 更新權重 ---
    W2 += dW2
    W1 += dW1
    
    E = 0.5 * np.mean(e**2)
    E_list.append(E)


def predict_system(Train_X):  #訓練完成後的系統:將初始輸入帶進
    V1 = Train_X @ W1.T
    Y1 = phi(V1)
    V2 = Y1 @ W2.T
    Y2 = phi(V2)     
    return Y2  #就是Z

Z_pred_train = predict_system(Train_X)
Z_pred_test = predict_system(Test_X)

Z_pred_train_R = denormalize(Z_pred_train)
Z_pred_test_R = denormalize(Z_pred_test)

dZ_train_R = denormalize(dZ_train)
dZ_test_R =denormalize(dZ_test)
    
E_train = 0.5 * np.mean((dZ_train_R - Z_pred_train_R)**2)
E_test = 0.5 * np.mean((dZ_test_R  - Z_pred_test_R)**2)

print("E_train:", E_train)
print("E_test:", E_test)



plt.figure(figsize=(8, 5))
plt.plot(E_list)
plt.xlabel("Epoch (n)")
plt.ylabel("Average Error E")
plt.title("Error function for Training")
plt.grid(True)
plt.tight_layout()
plt.show()
