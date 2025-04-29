import numpy as np

# --- GA 參數 ---
POP_SIZE  = 10
CR_RATE   = 0.8      # 80% 個體被選來交配
MUT_SIGMA = 1.0
X_MIN, X_MAX = -10.0, 10.0
S0 = 0.1 * (X_MAX - X_MIN)   # 0.1*20 = 2.0  

# 目標函數
def f1(x):
    return -15 * np.sin(2*x)**2 - (x-2)**2 + 160

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

# --- 交配，回傳 offspring 向量 (length = num_mate) ---
def arithmetic_crossover(pairs_idx, pop):
    num_pairs = pairs_idx.shape[0]
    offspring = np.empty(num_pairs * 2)
    for k, (i, j) in enumerate(pairs_idx):
        p1, p2 = pop[i], pop[j]
        δ      = np.random.rand()    # δ ∈ [0,1]
        c1     = p1 + δ * (p2 - p1)
        c2     = p2 - δ * (p2 - p1)
        offspring[2*k]   = c1
        offspring[2*k+1] = c2
    return np.clip(offspring, X_MIN, X_MAX)

# --- 全量突變 ---
def mutate_all(offspring, gen):
    S     = S0 * np.exp(-0.01 * gen)                       # 問題依賴尺度
    noise = np.random.uniform(-1, 1, size=offspring.shape) # 每個子代一個 noise
    mutated = offspring + S * noise
    return np.clip(mutated, X_MIN, X_MAX)

# --- GA 主程式 ---
pop = np.random.uniform(X_MIN, X_MAX, size=POP_SIZE)
for gen in range(300):
    fitness      = f1(pop)
    pairs, losers = roulette_wheel_selection(fitness, CR_RATE)
    
    # 1) 交配
    offspring = arithmetic_crossover(pairs, pop)
    # 2) 突變 (所有 offspring 都突變一次)
    offspring = mutate_all(offspring, gen)
    
    # 3) 把 losers 直接帶到下一代
    children = list(offspring) + [pop[i] for i in losers]
    pop = np.array(children)

# --- 結果輸出 ---
fit_vals = f1(pop)
best_idx = np.argmax(fit_vals)
print(f"Best x = {pop[best_idx]:.4f}, f1(x) = {fit_vals[best_idx]:.4f}")
