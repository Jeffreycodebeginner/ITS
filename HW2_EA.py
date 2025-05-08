import numpy as np
import matplotlib.pyplot as plt
import random

# --- GA 參數 ---
POP_SIZE  = 10
CR_RATE   = 0.8      # 80% 個體被選來交配
# MUT_SIGMA = 1.0
X_MIN, X_MAX = -10.0, 10.0
MAX_GEN=10000



k =2
a,b=2,1000
label_1 = {1: "f(x)", 2: f"[f(x)]^{k}", 3: f"{a}f(x)+{b}"}


def calculate_fitness(choice, x):
    f = -15 * (np.sin(2*x))**2 - (x-2)**2 + 160
    if choice == 1:
        return f
    elif choice == 2:
        return f**k
    else:
        return a * f + b


# --- 輪盤選擇，回傳配對 indices 跟 losers indices ---
def roulette_wheel_selection(fitness, cr_rate):
    POP_SIZE = len(fitness)
    cum_fit = np.cumsum(fitness)
    total   = cum_fit[-1]
    spins    = np.random.rand(POP_SIZE) * total
    selected = np.searchsorted(cum_fit, spins)
    num_mate = int(POP_SIZE * cr_rate)
    if num_mate % 2 == 1:
        num_mate -= 1
    pairs_idx  = selected[:num_mate].reshape(-1, 2)
    losers_idx = selected[num_mate:]
    return pairs_idx, losers_idx

def tournament_selection(fitness, cr_rate, tour_size=2):
    pop_size = len(fitness)
    # 1) 辦 pop_size 場比賽
    winners = []
    for _ in range(pop_size):
        participants = np.random.choice(pop_size, size=tour_size, replace=False)
        winner = participants[np.argmax(fitness[participants])]
        winners.append(winner)

    # 2) 計算要配對的人數（偶數）
    num_mate = int(pop_size * cr_rate)
    if num_mate % 2 == 1:
        num_mate -= 1

    # 3) 前面那些贏家做 pairs
    pairs_idx = np.array(winners[:num_mate]).reshape(-1, 2)
    # 4) 後面那些贏家就是 losers
    losers_idx = winners[num_mate:]
    
    return pairs_idx, losers_idx


def selection_wrapper(method, fitness, cr_rate):
    """
    method: 'A' = 輪盤賭, 'B' = 競賽式
    fitness: 1D array
    cr_rate: crossover rate
    """
    if method == 'A':
        return roulette_wheel_selection(fitness, cr_rate)
    elif method == 'B':
        return tournament_selection(fitness, cr_rate)
    else:
        print("未識別選擇策略，將使用輪盤賭")
        return roulette_wheel_selection(fitness, cr_rate)


# --- 交配，回傳 offspring 向量 (length = num_mate) ---
def ea_crossover(pairs_idx, pop):
    num_pairs = pairs_idx.shape[0]
    # 每對產 2 個子代
    offspring = np.empty(2 * num_pairs)
    for k, (i, j) in enumerate(pairs_idx):
        p1, p2 = pop[i], pop[j]
        r = np.random.rand()
        # 兩個子代
        offspring[2*k]   = r * p1 + (1 - r) * p2
        offspring[2*k+1] = r * p2 + (1 - r) * p1
    return np.clip(offspring, X_MIN, X_MAX)

def ea_mutation(offspring,gen):
    # 1. 抽一個全域比例 r
    r = np.random.rand()  # 範圍 ∈ (0,1)
    # 2. 產生與 offspring 同形狀的向量 d，每個元素 ∈ (−1,1)
    d = np.random.uniform(-1, 1, size=offspring.shape)
    # 3. 一次把所有基因都做突變 x' = x + r*d
    mutated = offspring + r * d
    # 4. 邊界截斷
    return np.clip(mutated, X_MIN, X_MAX)


# ---  目標水準函數 ---
def compute_best_and_second_y(choice):
    if choice == 1:
        return ans_y, second_best
    elif choice == 2:
        return ans_y**k, second_best**k
    else:
        return (a * ans_y + b), (a * second_best + b)

# --- 全域最佳解計算 ---
x_all = np.arange(-10,10,0.01)
y_all = -15 * (np.sin(2*x_all))**2 - (x_all-2)**2 + 160
best_idx = np.argmax(y_all)
ans_x   = x_all[best_idx]
ans_y   = y_all[best_idx]
y2 = y_all.copy()
y2[best_idx] = -np.inf
second_best = np.max(y2)
print(f"全域最佳解 x = {ans_x:.4f}, y = {ans_y:.4f}")


def condition_value(choice):
    threshold = 150 
    if choice == 1:
        return threshold
    elif choice == 2:
        return threshold**k
    else:
        return a * threshold + b


# --- EA 主程式 ---
selection_method = input("請選擇選擇策略 (A: 輪盤賭, B: 競賽式): ").upper()
def run_EA(choice, max_gen=MAX_GEN):
    P_x = np.random.uniform(X_MIN, X_MAX, size=POP_SIZE)
    max_hist, avg_hist = [], []
    count, best_P, best_x = 0, None, None
    for gen in range(1, max_gen+1):
        fitness = calculate_fitness(choice, P_x)
        max_hist.append(fitness.max())
        avg_hist.append(fitness.mean())
        
        # 停止條件：累積 avg_fit > threshold(150) 五次且當前 max_fit ≥ target
        threshold = condition_value(choice)
        target, snd = compute_best_and_second_y(choice)
        if fitness.mean() >= threshold:
            count += 1
        # 當累積 avg_fit 次數≥5 且當前 max_fit ≥ snd 時停止
        if count >= 5 and fitness.max() >= snd:
            idx     = np.argmax(fitness)
            best_P  = P_x[idx].copy()
            best_x  = best_P
            print(f"Choice {choice} Gen {gen}: avg ≥ 150 且 max ≥ {second_best:.4f}, 停止")
            break
        pairs, losers = selection_wrapper(selection_method, fitness, CR_RATE)

        # 1) 交配
        offspring = ea_crossover(pairs, P_x)
        # 2) 突變 (所有 offspring 都突變一次)
        offspring = ea_mutation(offspring, gen)
        
        # 3) 把 losers 直接帶到下一代
        children = list(offspring) + [P_x[i] for i in losers]
        P_x = np.array(children)
        
    if best_P is None:
        idx    = np.argmax(fitness)
        best_P = P_x[idx].copy()
        best_x  = best_P
    return max_hist, avg_hist,best_P,best_x,gen


#繪圖
seed = random.randint(1, 100000)
seed= 87927
print("種子碼:",seed)
for choice in (1, 2, 3):
    np.random.seed(seed)
    max_h, avg_h ,best_P,best_x,gen= run_EA(choice)
    target,_ = compute_best_and_second_y(choice)
    print(f"[Choice {choice}] 最佳染色體(實數)  x= {best_P:.4f}")
    
    plt.figure(figsize=(6,4))
    plt.plot(max_h, label='Max Fitness')
    plt.plot(avg_h, label='Average Fitness')
    plt.axhline(target, color='k', linestyle='--', label='Target')
    plt.title(f"EA Converge Curve Data not deel— {label_1[choice]} ; seed ={seed}")
    plt.xlabel(f"Generation (converge Gen:{gen})")
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
    plt.title(f"EA Converge Curve — {label_1[choice]} ; seed ={seed}")
    plt.xlabel(f"Generation (converge Gen:{gen})")
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6,4))
    plt.plot(x_all, y_all, label='f(x)')
    plt.axvline(best_x, color='k', linestyle='--', label=f'solution x , x= {best_P:.4f} ' )
    plt.scatter(ans_x, ans_y, color='red', label=f'Max at x={ans_x:.3f}, f(x)={ans_y:.3f}')
    plt.title(f'EA Plot of f(x) with {label_1[choice]}')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    
    