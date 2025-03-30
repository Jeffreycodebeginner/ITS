import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === 隸屬函數定義 ===
def mu_temp_low(x): return -x / 4 if -4 <= x <= 0 else 0
def mu_temp_medium(x):
    if -3 <= x <= 0: return (x + 3) / 3
    elif 0 < x <= 3: return (-x + 3) / 3
    else: return 0
def mu_temp_high(x): return x / 4 if 0 <= x <= 4 else 0

def mu_weight_light(y): return -y + 1 if 0 <= y <= 1 else 0
def mu_weight_medium(y):
    if 0.5 <= y <= 1: return 2*y - 1
    elif 1 < y <= 1.5: return -2*y + 3
    else: return 0
def mu_weight_heavy(y): return y - 1 if 1 <= y <= 2 else 0

def mu_time_short(z): return -z / 5 + 1 if 0 <= z <= 5 else 0
def mu_time_medium(z):
    if 0 <= z <= 5: return z / 5
    elif 5 < z <= 10: return -z / 5 + 2
    else: return 0
def mu_time_long(z): return z / 5 - 1 if 5 <= z <= 10 else 0

def mu_power_low(w): return -1/200 * (w - 800) if 600 <= w <= 800 else 0
def mu_power_medium(w):
    if 700 <= w <= 900: return 1/200 * (w - 700)
    elif 900 < w <= 1100: return -1/200 * (w - 1100)
    else: return 0
def mu_power_high(w): return 1/200 * (w - 1000) if 1000 <= w <= 1200 else 0

# === 規則庫 ===
rules = [
    ("low", "heavy", "long", "high"),
    ("low", "medium", "medium", "high"),
    ("low", "light", "short", "high"),
    ("medium", "heavy", "long", "medium"),
    ("medium", "medium", "medium", "medium"),
    ("medium", "light", "short", "medium"),
    ("high", "heavy", "long", "low"),
    ("high", "medium", "medium", "low"),
    ("high", "light", "short", "low"),
]

z_values = np.arange(0, 10.1, 0.1)
w_values = np.arange(600, 1200.1, 0.1)

# === 主計算迴圈 ===
results = []
for x in np.round(np.arange(-4.0, 4.1, 0.1), 2):
    for y in np.round(np.arange(0.0, 2.1, 0.1), 2):
        µ_temp = {"low": mu_temp_low(x), "medium": mu_temp_medium(x), "high": mu_temp_high(x)}
        µ_weight = {"light": mu_weight_light(y), "medium": mu_weight_medium(y), "heavy": mu_weight_heavy(y)}

        fired_rules = []
        for temp_label, weight_label, time_label, power_label in rules:
            alpha = min(µ_temp[temp_label], µ_weight[weight_label])
            if alpha > 0:
                fired_rules.append({
                    "alpha": alpha,
                    "time_label": time_label,
                    "power_label": power_label
                })

        mu_z_agg = np.zeros_like(z_values)
        mu_w_agg = np.zeros_like(w_values)

        ca_z_numer, ca_z_denom = 0, 0
        ca_w_numer, ca_w_denom = 0, 0

        for rule in fired_rules:
            alpha = rule["alpha"]

            # z: time    #mu_z是每條規則中Z集合與alpha進行截斷後的值
            if rule["time_label"] == "short":
                mu_z = np.minimum([mu_time_short(z) for z in z_values], alpha)
            elif rule["time_label"] == "medium":
                mu_z = np.minimum([mu_time_medium(z) for z in z_values], alpha)
            elif rule["time_label"] == "long":
                mu_z = np.minimum([mu_time_long(z) for z in z_values], alpha)
            mu_z_agg = np.maximum(mu_z_agg, mu_z)  #COG,MOM,MODMOM的每項規則聯集輸出
            mu_z = np.array(mu_z)
            max_z = np.max(mu_z)
            z_support = z_values[mu_z == max_z]
            z_c = (np.min(z_support) + np.max(z_support)) / 2 if len(z_support) > 0 else 0  #CA取支撐點並找出支撐點的中心
            
            #CA計算跑每條規則後進行累加
            ca_z_numer += z_c * max_z   #max_z :裁剪後的最大值   * z_c:支撐點的中心點
            ca_z_denom += max_z         
            
            
            # w: power    #mu_w是每條規則中Z集合與alpha進行截斷後的值
            if rule["power_label"] == "low":
                mu_w = np.minimum([mu_power_low(w) for w in w_values], alpha)
            elif rule["power_label"] == "medium":
                mu_w = np.minimum([mu_power_medium(w) for w in w_values], alpha)
            elif rule["power_label"] == "high":
                mu_w = np.minimum([mu_power_high(w) for w in w_values], alpha)
            mu_w_agg = np.maximum(mu_w_agg, mu_w)
            mu_w = np.array(mu_w)
            max_w = np.max(mu_w)
            w_support = w_values[mu_w == max_w]
            w_c = (np.min(w_support) + np.max(w_support)) / 2 if len(w_support) > 0 else 0
            ca_w_numer += w_c * max_w
            ca_w_denom += max_w
        
        #解模糊化
        z_cog = np.sum(mu_z_agg * z_values) / np.sum(mu_z_agg) if np.sum(mu_z_agg) > 0 else 0
        w_cog = np.sum(mu_w_agg * w_values) / np.sum(mu_w_agg) if np.sum(mu_w_agg) > 0 else 0

        mu_max_z = np.max(mu_z_agg)
        z_mom = np.mean(z_values[mu_z_agg == mu_max_z]) if mu_max_z > 0 else 0
        mu_max_w = np.max(mu_w_agg)
        w_mom = np.mean(w_values[mu_w_agg == mu_max_w]) if mu_max_w > 0 else 0

        z_support = z_values[mu_z_agg == mu_max_z]
        z_mmod = (np.min(z_support) + np.max(z_support)) / 2 if len(z_support) > 0 else 0
        w_support = w_values[mu_w_agg == mu_max_w]
        w_mmod = (np.min(w_support) + np.max(w_support)) / 2 if len(w_support) > 0 else 0

        z_ca = ca_z_numer / ca_z_denom if ca_z_denom > 0 else 0
        w_ca = ca_w_numer / ca_w_denom if ca_w_denom > 0 else 0

        results.append((x, y, z_cog, w_cog, z_mom, w_mom, z_mmod, w_mmod, z_ca, w_ca))

# === 存入 DataFrame ===
df = pd.DataFrame(results, columns=[
    "x", "y",
    "z_COG", "w_COG",
    "z_MoM", "w_MoM",
    "z_ModMoM", "w_ModMoM",
    "z_CA", "w_CA"
])

# === 繪圖 ===
def plot_surface(x, y, z, title, zlabel, cmap='viridis'):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_trisurf(x, y, z, cmap=cmap, edgecolor='none')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    ax.set_title(title)
    ax.set_xlabel("Temperature (x)")
    ax.set_ylabel("Weight (y)")
    ax.set_zlabel(zlabel)
    plt.tight_layout()
    plt.show()

# 繪圖：不遮蔽但顯示所有值（包含 w < 600）
plot_surface(df["x"], df["y"], df["z_COG"], "COG - Time (z*)", "Time z*")
plot_surface(df["x"], df["y"], df["z_MoM"], "MoM - Time (z*)", "Time z*")
plot_surface(df["x"], df["y"], df["z_ModMoM"], "Modified MoM - Time (z*)", "Time z*")
plot_surface(df["x"], df["y"], df["z_CA"], "Center Average - Time (z*)", "Time z*")
plot_surface(df["x"], df["y"], df["w_COG"], "COG - Power (w*)", "Power w*", cmap='plasma')
plot_surface(df["x"], df["y"], df["w_MoM"], "MoM - Power (w*)", "Power w*", cmap='plasma')
plot_surface(df["x"], df["y"], df["w_ModMoM"], "Modified MoM - Power (w*)", "Power w*", cmap='plasma')
plot_surface(df["x"], df["y"], df["w_CA"], "Center Average - Power (w*)", "Power w*", cmap='plasma')
