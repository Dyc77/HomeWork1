import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression

# 設置標題
st.title("線性回歸互動應用")

# 參數設置
st.sidebar.header("參數設置")
num_points = st.sidebar.slider("數據點數量", min_value=10, max_value=200, value=100)
x_range = st.sidebar.slider("X 範圍", min_value=0.0, max_value=10.0, value=(0.0, 2.0))
noise_std = st.sidebar.slider("噪聲標準差", min_value=0.0, max_value=10.0, value=5.0)  # 噪聲標準差的滑塊

# 提取 x_min 和 x_max
x_min, x_max = x_range

# 隨機生成 X 和 y
X = np.random.uniform(x_min, x_max, (num_points, 1))
noise = np.random.normal(0, noise_std, (num_points, 1))  # 使用滑塊值作為噪聲的標準差
y = 4 + 3 * X + noise  # y = 4 + 3x + 噪聲

# 擬合線性回歸模型
model = LinearRegression()
model.fit(X, y)

# 預測
X_new = np.array([[x_min], [x_max]])
y_predict = model.predict(X_new)

# 繪製圖表
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='數據點')
plt.plot(X_new, y_predict, color='red', label='回歸線')
plt.axvline(x=x_min, color='green', linestyle='--', label='X 最小值')
plt.axvline(x=x_max, color='orange', linestyle='--', label='X 最大值')

# 添加 X 最小值和最大值的標籤
plt.text(x_min, 0, f'X={x_min:.2f}', horizontalalignment='center', verticalalignment='bottom', color='green')
plt.text(x_max, 0, f'X={x_max:.2f}', horizontalalignment='center', verticalalignment='bottom', color='orange')

plt.xlabel('X')
plt.ylabel('y')
plt.title('線性回歸')
plt.legend()

# 設置中文顯示
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 使用微軟正黑體
plt.rcParams['axes.unicode_minus'] = False  # 正常顯示負號

# 顯示圖表
st.pyplot(plt)

# 顯示模型係數
st.sidebar.subheader("模型係數")
st.sidebar.write(f"截距: {model.intercept_[0]:.2f}")
st.sidebar.write(f"斜率: {model.coef_[0][0]:.2f}")
