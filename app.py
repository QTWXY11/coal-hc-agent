import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="煤基硬碳AI智能体", layout="wide")
st.title("🏭 煤基硬碳合成工艺AI智能体")

# 检查模型是否存在，不存在就训练
if not os.path.exists("model.pkl"):
    with st.spinner("首次运行，正在训练模型，请稍候..."):
        df = pd.read_csv("coal_hc_data.csv")
        features = ['ash_content', 'volatile_matter', 'carbonization_temp', 'carbonization_time', 'heating_rate']
        X = df[features].fillna(df[features].median())
        y = df['reversible_capacity']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
        model.fit(X_scaled, y)
        joblib.dump(model, "model.pkl")
        joblib.dump(scaler, "scaler.pkl")
        joblib.dump(features, "features.pkl")
    st.success("模型训练完成！")

# 加载模型
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

# 侧边栏：参数输入
st.sidebar.header("⚙️ 工艺参数输入")
coal_type = st.sidebar.selectbox("煤种", ["褐煤", "烟煤", "无烟煤"])
ash = st.sidebar.number_input("灰分 (%)", 0.0, 20.0, 8.0)
volatile = st.sidebar.number_input("挥发分 (%)", 0.0, 50.0, 35.0)
temp = st.sidebar.slider("碳化温度 (℃)", 800, 2000, 1300)
time = st.sidebar.slider("保温时间 (h)", 0.5, 5.0, 2.0)
rate = st.sidebar.slider("升温速率 (℃/min)", 1, 20, 5)

# 预测按钮
if st.sidebar.button("🔮 预测容量"):
    input_data = pd.DataFrame([[ash, volatile, temp, time, rate]], columns=features)
    input_scaled = scaler.transform(input_data)
    cap = model.predict(input_scaled)[0]
    st.sidebar.success(f"📊 预测可逆容量：**{cap:.1f} mAh/g**")

# 主区域：说明 + 数据展示
st.markdown("""
### 📖 项目说明
本AI智能体基于 **随机森林机器学习模型** 构建，输入煤基硬碳的关键工艺参数（灰分、挥发分、碳化温度、保温时间、升温速率），即可快速预测可逆容量（mAh/g）。

### 🔬 当前知识库数据
""")
df = pd.read_csv("coal_hc_data.csv")
st.dataframe(df, use_container_width=True)

st.markdown("""
---
**中国矿业大学 大学生创新训练计划项目** | 指导教师：朱荣涛、杨浩 | 负责人：任权
""")
