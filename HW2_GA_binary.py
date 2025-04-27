import numpy as np
import matplotlib.pyplot as plt
import os

seed = int.from_bytes(os.urandom(4), 'little')
print(f"使用的 seed = {seed}")

# ----------------------------
# GA 全域參數
# ----------------------------
POP_SIZE   = 10    # 族群大小
CHROM_LEN  = 10    # 染色體長度（bit 數）
MR         = 0.01  # 突變率（簡化為每代固定 1 bit 變異）
CR         = 0.8   # 交配率
MAX_GEN    = 30000 # 最大世代數

labels = {1: "f(x)", 2: "[f(x)]^2", 3: "2f(x)+1"}
a, b   = 2, 1      # 用於第三類 fitness = a*f + b

# ----------------------------
# 工具函式
# ----------------------------
def init_population(pop_size, chrom_len):
    return np.random.randint(0, 2, size=(pop_size, chrom_len))

def binary_to_decimal(P):
    weights = 2 ** np.arange(CHROM_LEN-1, -1, -1)
    return P.dot(weights)

def decimal_to_x(P_decimal):
    return -10 + (20 / (2**CHROM_LEN - 1)) * P_decimal

def calculate_fitness(choice, x):
    f = -15 * (np.sin(2*x))**2 - (x-2)**2 + 160
    if choice == 1:
        return f
    elif choice == 2:
        return f**2
    else:
        return a*f + b

def roulette_wheel_selection(fit, cr):
    num_mate = int(len(fit) * cr)
    cum_fit   = np.cumsum(fit)
    total     = cum_fit[-1]
    spins     = np.random.rand(len(fit)) * total
    selected  = [np.searchsorted(cum_fit, s) for s in spins]
    pairs     = np.array(selected[:num_mate]).reshape(-1, 2)
    elites    = selected[num_mate:]
    return pairs, elites

def single_point_crossover(P, pairs):
    children = []
    for i1, i2 in pairs:
        p1, p2 = P[i1], P[i2]
        cp     = np.random.randint(1, CHROM_LEN-1)
        c1     = np.concatenate([p1[:cp], p2[cp:]])
        c2     = np.concatenate([p2[:cp], p1[cp:]])
        children.extend([c1, c2])
    return np.array(children)

def mutate_one_bit(offspring):
    i = np.random.randint(len(offspring))
    j = np.random.randint(offspring.shape[1])
    offspring[i,j] ^= 1
    return offspring

def form_new_population(P, offspring, elites):
    elite_inds = P[elites]
    return np.vstack([offspring, elite_inds])

# ----------------------------
# 全域最優與門檻計算
# ----------------------------
def target_function():
    P_dec = np.arange(2**CHROM_LEN)
    x_all = decimal_to_x(P_dec)
    y_all = -15 * (np.sin(2*x_all))**2 - (x_all-2)**2 + 160
    idx   = np.argmax(y_all)
    return x_all, y_all, x_all[idx], y_all[idx]

_, _, _, OPtimum = target_function()
print("Global Optimum f(x) =", OPtimum)

def compute_target_and_tol(choice):
    tol = 0.02
    tol2 = 1.0
    if choice == 1:
        return OPtimum,        tol,      tol2
    elif choice == 2:
        return OPtimum**2,     tol**2/10, tol2**2/10
    else:
        return a*OPtimum + b,  a*tol + b,  a*tol2 + b


# ----------------------------
# 停止條件封裝函式
# ----------------------------
def check_stopping_conditions(
    P,
    max_fit,
    avg_fit,
    avg_hist,
    max_hist,      
    target,
    tol,
    tol2,
    homo_count,
    max_tol_count,
    homo_limit=2
):
    # 1) 族群同質化
    if np.all(P == P[0]):
        homo_count += 1
    else:
        homo_count = 0
    if homo_count >= homo_limit:
        return True, f"連續 {homo_limit} 代族群同質化", homo_count

    # 2) 平均達標
    if avg_fit >= target:
        return True, f"平均適應度達到目標 {target:.3f}", homo_count

    # 3) 當 max_fit 進入容差區間，且最近 3 代 avg_fit 波動 < 0.02
    if abs(max_fit - target) <= tol2 and len(avg_hist) >= 4:
        recent = avg_hist[-4:]
        if max(recent) - min(recent) < tol2:
            return True, (
                f"max_fit 已與目標誤差 ≤ {tol:.3f}，"
                "且最近 4 代 avg_fit 收斂 (Δ < 0.02)"
            ), homo_count
    

        
        
        
    # 4) 平均接近容差
    if abs(target - avg_fit) <= tol:
        return True, f"平均適應度與目標誤差 ≤ {tol:.3f}", homo_count
    return False, None, homo_count

    # 5) **max_fit 連續 5 代進入 tol**, 同時 avg_fit 也在 tol 以內
    if abs(max_fit - target) <= tol:
        max_tol_count += 1
    else:
        max_tol_count = 0

    if max_tol_count >= 5 and abs(avg_fit - target) <= tol:
        return True, f"最高適應度連續 {max_tol_count} 代在誤差 ≤ {tol:.3f}，且平均也在容差內", homo_count, max_tol_count

    # 若都沒達到，繼續
    return False, None, homo_count, max_tol_count

    
    


# ----------------------------
# GA 主程式
# ----------------------------
def run_ga(choice, max_gen=MAX_GEN):
    P = init_population(POP_SIZE, CHROM_LEN)
    target, tol, tol2 = compute_target_and_tol(choice)  # ← 這裡要有 tol2
    max_hist, avg_hist = [], []
    homo_count = 0
    max_tol_count    = 0

    for gen in range(1, max_gen+1):
        # 計算 fitness
        dec       = binary_to_decimal(P)
        x_real    = decimal_to_x(dec)
        fit_value = calculate_fitness(choice, x_real)

        max_fit = fit_value.max()
        avg_fit = fit_value.mean()
        max_hist.append(max_fit)
        avg_hist.append(avg_fit)

        # 檢查停止條件（傳入 max_fit、avg_fit）
        stop, reason, homo_count = check_stopping_conditions(
                P, max_fit, avg_fit, avg_hist,max_hist,
                target, tol, tol2, homo_count,max_tol_count,homo_limit=2
            )

        if stop:
            print(f"{labels[choice]} | Gen {gen:4d} | 停止：{reason}")
            break

        # 演化步驟
        pairs, elites = roulette_wheel_selection(fit_value, CR)
        children      = single_point_crossover(P, pairs)
        children      = mutate_one_bit(children)
        P             = form_new_population(P, children, elites)

    return max_hist, avg_hist

# ----------------------------
# 主程式：畫收斂曲線並顯示
# ----------------------------
if __name__ == "__main__":
    for choice in (1, 2, 3):
        np.random.seed(seed)
        max_hist, avg_hist = run_ga(choice)
        target, tol, tol2 = compute_target_and_tol(choice)
        
        if choice == 2:
            max_hist = np.sqrt(max_hist)
            avg_hist = np.sqrt(avg_hist)
            target   = np.sqrt(target)
            
        if choice  == 3:
            max_hist = (np.array(max_hist) - b) / a
            avg_hist = (np.array(avg_hist) - b) / a
            target   = (target - b) / a

        plt.figure(figsize=(6,4))
        plt.plot(max_hist, label="Max Fitness")
        plt.plot(avg_hist, label="Average Fitness")
        plt.axhline(target, color="k", linestyle="--", label="Target")
        plt.title(f"GA Converge Cuve — {labels[choice]}")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()




    # 最後再畫一次 f(x) 函數與全域最大點
    x_all, y_all, x_max, y_max = target_function()
    plt.figure(figsize=(6,4))
    plt.plot(x_all, y_all, label="f(x)")
    plt.scatter(x_max, y_max, color='red',
    label=f'Max at x={x_max:.3f}, f(x)={y_max:.3f}')
    plt.title('Plot of f(x)')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
