import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==== 隸屬函數定義 ====
def mu_temp_low(x): return -x / 4 if -4 <= x <= 0 else 0
def mu_temp_medium(x): return (x + 3) / 3 if -3 <= x <= 0 else (-x + 3) / 3 if 0 < x <= 3 else 0
def mu_temp_high(x): return x / 4 if 0 <= x <= 4 else 0

def mu_weight_light(y): return -y + 1 if 0 <= y <= 1 else 0
def mu_weight_medium(y): return 2*y - 1 if 0.5 <= y <= 1 else -2*y + 3 if 1 < y <= 1.5 else 0
def mu_weight_heavy(y): return y - 1 if 1 <= y <= 2 else 0

def mu_time_short(z): return -z / 5 + 1 if 0 <= z <= 5 else 0
def mu_time_medium(z): return z / 5 if 0 <= z <= 5 else -z / 5 + 2 if 5 < z <= 10 else 0
def mu_time_long(z): return z / 5 - 1 if 5 <= z <= 10 else 0

def mu_power_low(w): return -1/200 * (w - 800) if 600 <= w <= 800 else 0
def mu_power_medium(w):
    if 700 <= w <= 900: return 1/200 * (w - 700)
    elif 900 < w <= 1100: return -1/200 * (w - 1100)
    else: return 0
def mu_power_high(w): return 1/200 * (w - 1000) if 1000 <= w <= 1200 else 0

# ==== 規則設定 ====
rules = [
    {"x": mu_temp_low, "y": mu_weight_heavy, "z": mu_time_long, "w": mu_power_high},
    {"x": mu_temp_low, "y": mu_weight_medium, "z": mu_time_medium, "w": mu_power_high},
    {"x": mu_temp_low, "y": mu_weight_light, "z": mu_time_short, "w": mu_power_high},
    {"x": mu_temp_medium, "y": mu_weight_heavy, "z": mu_time_long, "w": mu_power_medium},
    {"x": mu_temp_medium, "y": mu_weight_medium, "z": mu_time_medium, "w": mu_power_medium},
    {"x": mu_temp_medium, "y": mu_weight_light, "z": mu_time_short, "w": mu_power_medium},
    {"x": mu_temp_high, "y": mu_weight_heavy, "z": mu_time_long, "w": mu_power_low},
    {"x": mu_temp_high, "y": mu_weight_medium, "z": mu_time_medium, "w": mu_power_low},
    {"x": mu_temp_high, "y": mu_weight_light, "z": mu_time_short, "w": mu_power_low}
]

# ==== 解模糊方法（修正版本） ====
def defuzz_cog(x, mf): return np.sum(x * mf) / np.sum(mf) if np.sum(mf) > 0 else 0

def defuzz_mom_user(x, mf):
    max_val = np.max(mf)
    if max_val == 0:
        return 0
    indices = np.where(np.isclose(mf, max_val, rtol=1e-4))[0]
    return np.mean(x[indices])

def defuzz_modified_mom_user(x, mf):
    max_val = np.max(mf)
    if max_val == 0:
        return 0
    indices = np.where(np.isclose(mf, max_val, rtol=1e-4))[0]
    return (np.max(x[indices]) + np.min(x[indices])) / 2

def defuzz_center_average_precise(x, mf_func, alpha):
    mf_vals = np.array([mf_func(xi) for xi in x])
    clipped = np.minimum(mf_vals, alpha)
    if np.max(clipped) == 0:
        return 0
    indices = np.where(clipped > 0)[0]
    i_left, i_right = indices[0], indices[-1]
    x_left, x_right = x[i_left], x[i_right]
    if i_left > 0:
        x0, x1 = x[i_left - 1], x[i_left]
        y0, y1 = clipped[i_left - 1], clipped[i_left]
        x_left = x0 + (alpha - y0) * (x1 - x0) / (y1 - y0) if y1 != y0 else x1
    if i_right < len(x) - 1:
        x0, x1 = x[i_right], x[i_right + 1]
        y0, y1 = clipped[i_right], clipped[i_right + 1]
        x_right = x0 + (alpha - y0) * (x1 - x0) / (y1 - y0) if y1 != y0 else x0
    return (x_left + x_right) / 2

# ==== 推論主函式 ====
def process_fuzzy_inference_user(x_in, y_in, rules, z_range, w_range):
    z_agg = np.zeros_like(z_range)
    w_agg = np.zeros_like(w_range)
    z_ca_sum = z_ca_weight = 0
    w_ca_sum = w_ca_weight = 0

    for rule in rules:
        alpha = min(rule["x"](x_in), rule["y"](y_in))
        if alpha > 0:
            z_cut = np.array([min(alpha, rule["z"](z)) for z in z_range])
            w_cut = np.array([min(alpha, rule["w"](w)) for w in w_range])
            z_agg = np.maximum(z_agg, z_cut)
            w_agg = np.maximum(w_agg, w_cut)

            z_center = defuzz_center_average_precise(z_range, rule["z"], alpha)
            w_center = defuzz_center_average_precise(w_range, rule["w"], alpha)
            z_ca_sum += alpha * z_center
            z_ca_weight += alpha
            w_ca_sum += alpha * w_center
            w_ca_weight += alpha

    return {
        "z_cog": defuzz_cog(z_range, z_agg),
        "z_mom": defuzz_mom_user(z_range, z_agg),
        "z_mmom": defuzz_modified_mom_user(z_range, z_agg),
        "z_ca": z_ca_sum / z_ca_weight if z_ca_weight > 0 else 0,
        "w_cog": defuzz_cog(w_range, w_agg),
        "w_mom": defuzz_mom_user(w_range, w_agg),
        "w_mmom": defuzz_modified_mom_user(w_range, w_agg),
        "w_ca": w_ca_sum / w_ca_weight if w_ca_weight > 0 else 0
    }

# ==== 範圍定義 ====
x_range = np.linspace(-4, 4, 81)
y_range = np.linspace(0, 2, 21)
z_range = np.linspace(0, 10, 101)
w_range = np.linspace(600, 1200, 601)

X_vals, Y_vals = [], []
z_cog_vals, z_mom_vals, z_mmom_vals, z_ca_vals = [], [], [], []
w_cog_vals, w_mom_vals, w_mmom_vals, w_ca_vals = [], [], [], []

for x in x_range:
    for y in y_range:
        result = process_fuzzy_inference_user(x, y, rules, z_range, w_range)
        X_vals.append(x)
        Y_vals.append(y)
        z_cog_vals.append(result["z_cog"])
        z_mom_vals.append(result["z_mom"])
        z_mmom_vals.append(result["z_mmom"])
        z_ca_vals.append(result["z_ca"])
        w_cog_vals.append(result["w_cog"])
        w_mom_vals.append(result["w_mom"])
        w_mmom_vals.append(result["w_mmom"])
        w_ca_vals.append(result["w_ca"])

df = pd.DataFrame({
    "x": X_vals,
    "y": Y_vals,
    "z_COG": z_cog_vals,
    "z_MOM": z_mom_vals,
    "z_ModifiedMOM": z_mmom_vals,
    "z_CA": z_ca_vals,
    "w_COG": w_cog_vals,
    "w_MOM": w_mom_vals,
    "w_ModifiedMOM": w_mmom_vals,
    "w_CA": w_ca_vals
})

def plot_surface(X, Y, Z, title, zlabel, cmap='viridis'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(X, Y, Z, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("x (°C)")
    ax.set_ylabel("y (kg)")
    ax.set_zlabel(zlabel)
    plt.tight_layout()
    plt.show()

plot_surface(X_vals, Y_vals, z_cog_vals, "COG Output z (Time)", "z (min)", cmap='viridis')
plot_surface(X_vals, Y_vals, z_mom_vals, "MOM Output z (Time)", "z (min)", cmap='plasma')
plot_surface(X_vals, Y_vals, z_mmom_vals, "Modified MOM Output z (Time)", "z (min)", cmap='cividis')
plot_surface(X_vals, Y_vals, z_ca_vals, "CA Output z (Time)", "z (min)", cmap='inferno')

plot_surface(X_vals, Y_vals, w_cog_vals, "COG Output w (Power)", "w (watt)", cmap='viridis')
plot_surface(X_vals, Y_vals, w_mom_vals, "MOM Output w (Power)", "w (watt)", cmap='plasma')
plot_surface(X_vals, Y_vals, w_mmom_vals, "Modified MOM Output w (Power)", "w (watt)", cmap='cividis')
plot_surface(X_vals, Y_vals, w_ca_vals, "CA Output w (Power)", "w (watt)", cmap='inferno')


print(df[(df["x"] == -2.0) & (df["y"] == 0.1)])
df.to_csv("bingle.csv", index=False)