import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# GA 全域參數
# ----------------------------
POP_SIZE   = 10    # 族群大小
CHROM_LEN  = 10    # 染色體長度（bit 數）
MR         = 0.01  # 突變率（這裡簡化為每代固定 1 bit 變異）
CR         = 0.8   # 交配率
MAX_GEN    = 300   # 最大世代

# ----------------------------
# 工具函式
# ----------------------------
def init_population(pop_size, chrom_len):
    """隨機初始化二進位族群矩陣 (pop_size × chrom_len)"""
    return np.random.randint(0, 2, size=(pop_size, chrom_len))

def binary_to_decimal(P):
    """將二進位染色體矩陣轉為十進位 array"""
    weights = 2 ** np.arange(CHROM_LEN-1, -1, -1)
    return P.dot(weights)

def decimal_to_x(P_decimal):
    """把十進位值映射到實數區間 [-10,10]"""
    return -10 + (20 / (2**CHROM_LEN - 1)) * P_decimal

def calculate_fitness(choice, x):
    """三種 fitness function 選擇"""
    f = -15 * (np.sin(2*x))**2 - (x-2)**2 + 160
    if choice == 1:
        return f
    elif choice == 2:
        return f**2
    else:
        return 2*f + 1

def roulette_wheel_selection(fit, cr):
    """輪盤選擇，回傳配對 indices 與 elites indices"""
    num_mate = int(len(fit) * cr)
    cum_fit   = np.cumsum(fit)
    total     = cum_fit[-1]
    spins     = np.random.rand(len(fit)) * total
    selected  = [np.searchsorted(cum_fit, s) for s in spins]
    pairs     = np.array(selected[:num_mate]).reshape(-1, 2)
    elites    = selected[num_mate:]
    return pairs, elites

def single_point_crossover(P, pairs):
    """單點交叉"""
    children = []
    for i1, i2 in pairs:
        p1, p2 = P[i1], P[i2]
        cp     = np.random.randint(1, CHROM_LEN-1)
        c1     = np.concatenate([p1[:cp], p2[cp:]])
        c2     = np.concatenate([p2[:cp], p1[cp:]])
        children.extend([c1, c2])
    return np.array(children)

def mutate_one_bit(offspring):
    """隨機翻轉 1 個 bit"""
    i = np.random.randint(len(offspring))
    j = np.random.randint(offspring.shape[1])
    offspring[i,j] ^= 1
    return offspring

def form_new_population(P, offspring, elites):
    """將 offspring 與 elites 重組成新族群"""
    elite_inds = P[elites]
    return np.vstack([offspring, elite_inds])

# ----------------------------
# 全域最優與門檻計算
# ----------------------------
def target_function():
    """在離散化的 2^CHROM_LEN 點上找到全域最大 f(x)"""
    P_dec = np.arange(2**CHROM_LEN)
    x     = decimal_to_x(P_dec)
    y     = -15 * (np.sin(2*x))**2 - (x-2)**2 + 160
    max_index = np.argmax(y) 
    y_max = y[max_index]        # 最大的 y
    x_max = x[max_index]        # 對應的 x
    return x,y,x_max,y_max

_, _, _, OPtimum = target_function()

def compute_target_and_tol(choice):
    """根據 choice 設定停止門檻 target 和 tolerance"""
    err = 0.01
    if choice == 1:
        return OPtimum,           err
    elif choice == 2:
        return OPtimum**2,        err**2 / 10
    else:
        return 2*OPtimum + 1,  2*err + 1

# ----------------------------
# GA 主程式
# ----------------------------
def run_ga(choice, max_gen=MAX_GEN):
    """
    執行一輪 GA：
      - choice: 1/2/3 對應三種 fitness function
      - 回傳每代的 max_hist, avg_hist
    """
    P           = init_population(POP_SIZE, CHROM_LEN)
    target, tol = compute_target_and_tol(choice)
    max_hist, avg_hist = [], []
    homogeneous_count   = 0

    for gen in range(1, max_gen+1):
        # 編碼→解碼→計算 fitness
        dec       = binary_to_decimal(P)
        x_real    = decimal_to_x(dec)
        fit_value = calculate_fitness(choice, x_real)

        # 紀錄本代最大與平均
        max_fit = np.max(fit_value)
        avg_fit = np.mean(fit_value)
        max_hist.append(max_fit)
        avg_hist.append(avg_fit)

        # 停止條件 1：連續 5 代族群同質化
        if np.all(P == P[0]):
            homogeneous_count += 1
        else:
            homogeneous_count = 0
        if homogeneous_count >= 5:
            break

        # 停止條件 2：平均達到／超越 target
        if avg_fit >= target:
            break

        # 停止條件 3：平均 fitness 收斂
        if abs(target - avg_fit) <= tol:
            break

        # 演化步驟
        pairs, elites = roulette_wheel_selection(fit_value, CR)
        children      = single_point_crossover(P, pairs)
        children      = mutate_one_bit(children)
        P             = form_new_population(P, children, elites)

    return max_hist, avg_hist

# ----------------------------
# 主程式：各自分開畫三張圖
# ----------------------------
if __name__ == "__main__":
    labels = {1:"f(x)", 2:"f(x)^2", 3:"2f(x)+1"}

    for choice in (1, 2, 3):
        max_hist, avg_hist = run_ga(choice)
        target, _          = compute_target_and_tol(choice)

        plt.figure(figsize=(6,4))
        plt.plot(max_hist, label="Max Fitness")
        plt.plot(avg_hist, label="Average Fitness")
        plt.axhline(target, color="k", linestyle="--", label="Target")
        plt.title(f"GA 收斂曲線 — {labels[choice]}")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()




# 先呼叫 target_function 取得所有資料
x_all, y_all, x_max, y_max = target_function()

# 畫出 f(x) 曲線
plt.plot(x_all, y_all, label="f(x)")

# 標出全域最大點
plt.scatter(x_max, y_max, color='red',
            label=f'Max at x={x_max:.3f}, f(x)={y_max:.3f}')

plt.title('Plot of f(x)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()