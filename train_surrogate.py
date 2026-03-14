import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 设置画图的中文字体（防止乱码）
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

def extract_num(text):
    """从 'Max thickness 10.1% at...' 提取数字 10.1"""
    if pd.isna(text): return np.nan
    match = re.search(r"([\d\.]+)\%", str(text))
    return float(match.group(1)) if match else np.nan

def extract_ld(text):
    """从 '40.4 at α=4°' 提取升阻比 40.4"""
    if pd.isna(text): return np.nan
    match = re.search(r"([\d\.]+)\s*at", str(text))
    if match: return float(match.group(1))
    match2 = re.search(r"([\d\.]+)", str(text))
    return float(match2.group(1)) if match2 else np.nan

print("⏳ 正在读取并清洗 Excel 数据，请稍候...")
df = pd.read_excel("airfoiltools_geo_clcd.xlsx") # 确保文件名正确
df.columns = df.columns.str.strip()

# 1. 清洗特征 (X)
df['thickness_num'] = df['最大厚度信息'].apply(extract_num)
df['camber_num'] = df['最大弯度信息'].apply(extract_num)

# 2. 展开雷诺数数据 (把宽表变成长表，为了喂给 AI)
# 我们把 5万, 10万, 20万, 50万 的数据都堆叠起来
dataset =[]
for re_val, col_name in zip([50000, 100000, 200000, 500000],['最大升阻比_Re50000', '最大升阻比_Re100000', '最大升阻比_Re200000', '最大升阻比_Re500000']):
    temp_df = df[['UIUC翼型名', 'thickness_num', 'camber_num', col_name]].copy()
    temp_df['Re'] = re_val
    temp_df['L_D'] = temp_df[col_name].apply(extract_ld)
    dataset.append(temp_df[['UIUC翼型名', 'thickness_num', 'camber_num', 'Re', 'L_D']])

final_df = pd.concat(dataset).dropna() # 拼合数据并删掉缺失值

X = final_df[['thickness_num', 'camber_num', 'Re']]
y = final_df['L_D']

print(f"✅ 数据清洗完毕！有效数据条数：{len(final_df)}")

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 训练随机森林模型
print("🧠 正在训练随机森林代理模型...")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 5. 评估模型并画图！(这图直接放进论文里)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"🎯 模型训练完成！R2 得分: {r2:.4f} (越接近1越好)")

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2) # 画一条完美预测的对角线
plt.title(f'翼型最大升阻比预测模型 Parity Plot (R²={r2:.3f})')
plt.xlabel('CFD 真实升阻比 (来自 Excel)')
plt.ylabel('AI 预测升阻比')
plt.grid(True, linestyle='-- ฉาก', alpha=0.6)
plt.savefig('模型预测误差图.png', dpi=300)
print("📊 预测散点图已保存为 '模型预测误差图.png'，请在文件夹中查看！")

# 6. 保存模型
joblib.dump(model, 'airfoil_surrogate_model.pkl')
print("💾 模型已保存为 'airfoil_surrogate_model.pkl'，可以接入网页了！")
