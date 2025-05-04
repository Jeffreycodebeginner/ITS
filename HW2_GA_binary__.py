import numpy as np
import matplotlib.pyplot as plt
import random

# ==========================
# GA 二元編碼程式
# ==========================

# --- 1. 全域參數設定 ---
POP_SIZE   = 10    # 族群大小
CHROM_LEN  = 10    # 染色體長度（bit 數）
MR         = 0.01  # 突變率（固定每代一位 bit 可能翻轉）
CR         = 0.8   # 交配率
MAX_GEN    = 10000   # 最大世代數
k =2# 第二類適應度 指數大小
a, b   = 2, 0   # 第三類適應度 a*f + b
labels = {1: "f(x)", 2: f"[f(x)]^{k}", 3: f"{a}f(x)+{b}"}

# --- 2. 初始化族群 ---
def init_population(pop_size, chrom_len):
    return np.random.randint(0, 2, size=(pop_size, chrom_len))

# --- 3. 編碼/解碼 ---
def binary_to_decimal(P):
    weights = 2 ** np.arange(CHROM_LEN - 1, -1, -1)
    return P.dot(weights)

def decimal_to_x(P_decimal):
    return -10 + (20 / (2**CHROM_LEN - 1)) * P_decimal

# --- 4. 適應度計算 ---
def calculate_fitness(choice, x):
    f = -15 * (np.sin(2*x))**2 - (x-2)**2 + 160
    if choice == 1:
        return f
    elif choice == 2:
        return f**k
    else:
        return a * f + b

# --- 5. 目標水準函數 ---
def compute_best_and_second_y(choice):
    if choice == 1:
        return ans_y, second_best
    elif choice == 2:
        return ans_y**k, second_best**k
    else:
        return a * ans_y + b, a * second_best + b

# --- 6. 選擇策略封裝 ---
def selection_wrapper(method, fit, cr):
    if method == 'A':
        return roulette_wheel_selection(fit, cr)
    elif method == 'B':
        return tournament_selection(fit, cr)
    else:
        print("未識別選擇策略，將使用輪盤賭")
        return roulette_wheel_selection(fit, cr)

# --- 6.1. 輪盤賭選擇 ---
def roulette_wheel_selection(fit, cr):
    num_mate = int(len(fit) * cr)
    cum_fit   = np.cumsum(fit)
    total     = cum_fit[-1]
    spins     = np.random.rand(len(fit)) * total
    selected  = [np.searchsorted(cum_fit, s) for s in spins]
    pairs     = np.array(selected[:num_mate]).reshape(-1, 2)
    losers    = selected[num_mate:]
    return pairs, losers

# --- 6.2. 競賽式選擇 ---
def tournament_selection(fit, cr, k=3):
    num_mate = int(len(fit) * cr)
    indices = []
    for _ in range(num_mate):
        competitors = np.random.choice(len(fit), k, replace=False)
        winner = competitors[np.argmax(fit[competitors])]
        indices.append(winner)
    pairs = np.array(indices).reshape(-1, 2)
    sorted_idx = np.argsort(fit)
    losers = sorted_idx[-(len(fit)-len(indices)):]
    return pairs, losers

# --- 7. 交配（單點） ---
def single_point_crossover(P, pairs):
    children = []
    for i1, i2 in pairs:
        p1, p2 = P[i1], P[i2]
        cp     = np.random.randint(1, CHROM_LEN)
        c1 = np.concatenate([p1[:cp], p2[cp:]])
        c2 = np.concatenate([p2[:cp], p1[cp:]])
        children.extend([c1, c2])
    return np.array(children)

# --- 8. 突變（翻轉一位） ---
def mutate_one_bit(offspring):
    i = np.random.randint(len(offspring))
    j = np.random.randint(offspring.shape[1])
    offspring[i, j] ^= 1
    return offspring

# --- 9. 新族群組成（含 losers） ---
def form_new_population(P, offspring, losers):
    loser_inds = P[losers]
    return np.vstack([offspring, loser_inds])

# --- 10. 全域最佳解計算 ---
P_dec = np.arange(2**CHROM_LEN)
x_all = decimal_to_x(P_dec)
y_all = -15 * (np.sin(2*x_all))**2 - (x_all-2)**2 + 160
best_idx = np.argmax(y_all)
ans_x   = x_all[best_idx]
ans_y   = y_all[best_idx]
y2 = y_all.copy()
y2[best_idx] = -np.inf
second_best = np.max(y2)
print(f"DEBUG: second_best = {second_best:.6f}")  
print(f"全域最佳解 x = {ans_x:.4f}, y = {ans_y:.4f}")

# --- 11. 停止條件門檻函式 ---
def conditon_value(choice):
    threshold = 145 
    tol=0.01
    if choice == 1:
        return threshold,tol
    elif choice == 2:
        return threshold**k,tol**k
    else:
        return a * threshold + b,a*tol

print(">>> IN compute_best_and_second_y:", compute_best_and_second_y(3))

# --- 12. 執行 GA 主流程 ---
def run_ga(choice, selection_method, max_gen=MAX_GEN):
    P = init_population(POP_SIZE, CHROM_LEN)
    max_hist, avg_hist = [], []
    count, best_P, best_x = 0, None, None
    for gen in range(1, max_gen+1):
        dec       = binary_to_decimal(P)
        x_real    = decimal_to_x(dec)
        fit_value = calculate_fitness(choice, x_real)
        max_hist.append(fit_value.max())
        avg_hist.append(fit_value.mean())
        

        # 停止條件：累積 avg_fit > threshold 五次且當前 max_fit ≥ target
        threshold,tol = conditon_value(choice)
        target, snd = compute_best_and_second_y(choice)
        print(f"DEBUG: choice={choice}, target={target:.6f}, snd={snd:.6f}")
        if fit_value.mean() >= threshold:
            count += 1
        # 當累積 avg_fit 次數≥5 且當前 max_fit ≥ snd 時停止
        if count >= 5 and fit_value.max() >= snd:
            idx     = np.argmax(fit_value)
            best_P  = P[idx].copy()
            best_x  = decimal_to_x(np.array([binary_to_decimal(best_P)]))[0]
            print(f"Choice {choice} Gen {gen}: avg ≥ {threshold} x5 且 max ≥ 158.8041, 停止")
            break
        
        
        
        pairs, losers = selection_wrapper(selection_method, fit_value, CR)
        children      = single_point_crossover(P, pairs)
        children      = mutate_one_bit(children)
        P             = form_new_population(P, children, losers)
        
    if best_P is None:
        idx    = np.argmax(fit_value)
        best_P = P[idx].copy()
        best_x = decimal_to_x( np.array([ binary_to_decimal(best_P) ]) )[0]
        
        
        
    return max_hist, avg_hist,best_P,best_x

# --- 13. 範例執行與繪圖 ---
if __name__ == '__main__':
    base_seed = random.randint(1, 100000)
    selection_method = input("請選擇選擇策略 (A: 輪盤賭, B: 競賽式): ").upper()
    for choice in (1, 2, 3):
        np.random.seed(base_seed)
        max_h, avg_h ,best_P,best_x= run_ga(choice, selection_method)
        target,second_best = compute_best_and_second_y(choice)
        print(f"[Choice {choice}] 最佳染色體: {best_P}, x = {best_x:.4f}\n")
        
        plt.figure(figsize=(6,4))
        plt.plot(max_h, label='Max Fitness')
        plt.plot(avg_h, label='Average Fitness')
        plt.axhline(target, color='k', linestyle='--', label='Target')
        plt.title(f"GA Converge Curve Data not deel— {labels[choice]}")
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        
        
        # 還原適應度
        if choice == 2:
            max_h = np.array(max_h) ** (1/k)
            avg_h = np.array(avg_h) ** (1/k)
            target = target**(1/k)
        elif choice == 3:
            max_h = (np.array(max_h) - b) / a
            avg_h = (np.array(avg_h) - b) / a
            target = (target - b) / a
        plt.figure(figsize=(6,4))
        plt.plot(max_h, label='Max Fitness')
        plt.plot(avg_h, label='Average Fitness')
        plt.axhline(target, color='k', linestyle='--', label='Target')

        plt.title(f"GA Converge Curve — {labels[choice]}")
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(6,4))
        plt.plot(x_all, y_all, label='f(x)')
        plt.axvline(best_x, color='k', linestyle='--', label='solution x')
        plt.scatter(ans_x, ans_y, color='red', label=f'Max at x={ans_x:.3f}, f(x)={ans_y:.3f}')
        plt.title('Plot of f(x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
