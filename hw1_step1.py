import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression

# 設置標題
st.title("線性回歸互動應用")

# 隨機生成數據
num_points = 100  # 固定數據點數量
np.random.seed(0)
X = 2 * np.random.rand(num_points, 1)
y = 4 + 3 * X + np.random.randn(num_points, 1)  # y = 4 + 3x + 噪聲

# 擬合線性回歸模型
model = LinearRegression()
model.fit(X, y)

# 預測
X_new = np.array([[0], [2]])
y_predict = model.predict(X_new)

# 繪製圖表
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='數據點')
plt.plot(X_new, y_predict, color='red', label='回歸線')

# 添加標籤和標題
plt.xlabel('X')
plt.ylabel('y')
plt.title('線性回歸')
plt.legend()

# 設置支持中文顯示的字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 或者使用其他支持中文的字體
plt.rcParams['axes.unicode_minus'] = False  # 正常顯示負號

# 顯示圖表
st.pyplot(plt)

# 顯示模型係數
st.sidebar.subheader("模型係數")
st.sidebar.write(f"截距: {model.intercept_[0]:.2f}")
st.sidebar.write(f"斜率: {model.coef_[0][0]:.2f}")
