import numpy as np
import matplotlib.pyplot as plt



# GA 參數
POP_SIZE   = 10
CHROM_LEN  = 10
MR         = 0.01
CR         = 0.8
homogeneous_count=0


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
    elif choice == 3:
        return 2*f + 1
    else:
        raise ValueError("Choice must be 1, 2 or 3")

def roulette_wheel_selection(fit, cr):
    pop_size = len(fit)
    num_mate = int(pop_size * cr)
    cum_fit = np.cumsum(fit)
    total = cum_fit[-1]
    spins = np.random.rand(pop_size) * total
    selected = [np.searchsorted(cum_fit, s) for s in spins]
    pairs = np.array(selected[:num_mate]).reshape(-1, 2)
    elites = selected[num_mate:]
    return pairs, elites

def single_point_crossover(P, pairs):
    children = []
    for i1, i2 in pairs:
        p1, p2 = P[i1], P[i2]
        cp = np.random.randint(1, CHROM_LEN-1)
        c1 = np.concatenate([p1[:cp], p2[cp:]])
        c2 = np.concatenate([p2[:cp], p1[cp:]])
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

def target_function():
    P_decimal = np.arange(2**CHROM_LEN)
    x         = decimal_to_x(P_decimal)
    y         = -15 * (np.sin(2*x))**2 - (x-2)**2 + 160
    best_idx  = np.argmax(y)          # 找到最大 y 的索引
    best_x    = x[best_idx]           # 对应的 x 值
    best_y    = y[best_idx]           # 对应的 y 值
    return x, y, best_x, best_y








OPtimum = target_function()[3]
print("Global Optimum f(x) =", OPtimum)

def allthesame(P):
    global homogeneous_count
    if np.all(P == P[0]):
        homogeneous_count += 1
    else:
        homogeneous_count = 0
    # 當發生所有人都一樣的情況次數至少10 次才判收敛
    return homogeneous_count >= 3

def has_converged_fitness(fit_history, error=1e-5, window=100):
    if len(fit_history) < window:
        return False
    recent = fit_history[-window:]
    return (max(recent) - min(recent)) < error


















def main():
    P = init_population(POP_SIZE, CHROM_LEN)
    choice = int(input("Select fitness function (1, 2, or 3): "))

    # 停止門檻
    error2=0.01
    if choice == 1:
        target = OPtimum
    elif choice == 2:
        target = OPtimum**2
        error2 = (error2**2)/10
    else:
        target = 2*OPtimum + 1
        error2 = 2*error2  + 1

    maxfit_history     = []
    average_fit_history = []

    for gen in range(1, 100001):
        # 編／解碼 + fitness
        b          = binary_to_decimal(P)
        x          = decimal_to_x(b)
        fit_value  = calculate_fitness(choice, x)

        # 最佳與平均
        best_idx    = np.argmax(fit_value)
        max_fit     = fit_value[best_idx]
        average_fit = np.mean(fit_value)
        print(f"世代 {gen:4d} | max={max_fit:8.3f} | avg={average_fit:8.3f}")

        # 紀錄歷史
        maxfit_history.append(max_fit)
        average_fit_history.append(average_fit)

        # 停止條件
        if allthesame(P):
            print(f"Stopped at gen {gen}: 發生所有人都一樣的次數=3.")
            break
        if average_fit >= target:
            print(f"Stopped at gen {gen}: 平均達到最佳目標函數 {target:.3f}.")
            break
        if has_converged_fitness(average_fit_history):
            print(f"Stopped at gen {gen}: 平均適應值已經收斂")
            break
        if abs(target - average_fit) <= error2:
            print(f"Stopped at gen {gen}: 平均適應值與目標的誤差小於 0.01 ")
            break

        # 演化
        pairs, elites = roulette_wheel_selection(fit_value, CR)
        children      = single_point_crossover(P, pairs)
        children      = mutate_one_bit(children)
        P             = form_new_population(P, children, elites)

    return maxfit_history, average_fit_history, target

if __name__ == "__main__":
    max_hist, avg_hist, target = main()

    # 畫收斂曲線
    plt.plot(max_hist, label="Max Fitness")
    plt.plot(avg_hist, label="Average Fitness")
    plt.axhline(target, color="k", linestyle="--", label="Target")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("GA Converge curve")
    plt.legend()
    plt.grid(True)
    plt.show()

    x, y, best_x, best_y = target_function()
    plt.plot(x, y, label="f(x)")
    plt.scatter(best_x, best_y, color='red',
    label=f'Max at x={best_x:.3f}, f(x)={best_y:.3f}')
    plt.title('Plot of f(x)')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()








