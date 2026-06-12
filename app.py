import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 页面配置
st.set_page_config(page_title="煤基硬碳AI工艺智能体", layout="wide")
st.title("🏭 煤基硬碳合成工艺AI智能体")
st.markdown("**左侧输入参数 → 右侧自动检索相似案例 + 生成工艺方案 + 预测性能**")

# 加载数据（缓存，避免重复加载）
@st.cache_resource
def load_data():
    df = pd.read_csv("data.csv")
    # 构建数值特征矩阵（用于相似度检索和预测）
    feature_cols = ['ash', 'volatile', 'carbon_temp', 'hold_time', 'heating_rate']
    X = df[feature_cols].fillna(df[feature_cols].median())
    # 训练容量预测模型（随机森林）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    model.fit(X_scaled, df['capacity'])
    # 训练最近邻检索器（基于数值特征）
    nn = NearestNeighbors(n_neighbors=3, metric='euclidean')
    nn.fit(X_scaled)
    return df, feature_cols, scaler, model, nn

df, feature_cols, scaler, model, nn = load_data()

# 左侧输入栏
st.sidebar.header("⚙️ 输入原料特性与目标")
coal_type = st.sidebar.selectbox("煤种", df['coal_type'].unique())
ash = st.sidebar.number_input("灰分 (%)", 0.0, 20.0, 8.0, step=0.1)
volatile = st.sidebar.number_input("挥发分 (%)", 0.0, 50.0, 35.0, step=0.1)
temp = st.sidebar.slider("碳化温度 (℃)", 800, 2000, 1300, step=10)
hold = st.sidebar.slider("保温时间 (h)", 0.5, 5.0, 2.0, step=0.5)
rate = st.sidebar.slider("升温速率 (℃/min)", 1, 20, 5, step=1)

# 可选：目标容量（用于案例筛选）
target_cap = st.sidebar.number_input("目标容量 (mAh/g) 可选", 200, 500, 300, step=10)

# 右侧主区域
if st.sidebar.button("🚀 生成工艺方案", use_container_width=True):
    with st.spinner("正在检索知识库并计算..."):
        # 1. 检索相似案例（基于数值特征）
        input_vec = np.array([[ash, volatile, temp, hold, rate]])
        input_scaled = scaler.transform(input_vec)
        distances, indices = nn.kneighbors(input_scaled)
        similar_df = df.iloc[indices[0]].copy()
        
        # 2. 预测容量
        pred_capacity = model.predict(input_scaled)[0]
        
        # 3. 基于相似案例的工艺描述生成（简单规则：取最相似案例的工艺，并根据参数微调）
        best_case = similar_df.iloc[0]
        base_process = best_case['process']
        # 动态生成工艺说明
        process_desc = f"""
        根据您输入的原料特性（煤种{coal_type}，灰分{ash}%，挥发分{volatile}%）和工艺参数（碳化温度{temp}℃，保温{hold}h，升温速率{rate}℃/min），系统检索到最相似的实验案例为：
        - **案例煤种**：{best_case['coal_type']}，容量{best_case['capacity']} mAh/g
        - **参考工艺**：{base_process}
        
        **综合推荐的工艺方案**：
        1. 预处理：{best_case['process'].split('→')[0] if '→' in best_case['process'] else '根据原料灰分建议酸洗（若灰分>5%）或碱洗（若挥发分>35%）'}
        2. 碳化：{temp}℃ 氩气气氛，保温{hold}小时，升温速率{rate}℃/min
        3. 后处理：碳化完成后自然冷却至室温（若追求更高倍率可考虑急冷）
        
        预期该工艺可获得可逆容量约 **{pred_capacity:.1f} mAh/g**。
        """
        
        # 4. 显示结果
        st.subheader("📋 生成的工艺方案与预测")
        st.markdown(process_desc)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📊 机器学习预测容量", f"{pred_capacity:.1f} mAh/g")
        with col2:
            st.metric("🎯 与目标容量差距", f"{pred_capacity - target_cap:.1f} mAh/g" if target_cap else "未设目标")
        
        st.subheader("🔍 知识库中最相似的3个案例")
        st.dataframe(similar_df[['coal_type', 'ash', 'volatile', 'carbon_temp', 'hold_time', 'heating_rate', 'capacity', 'process']])
        
        # 可选：显示检索过程（右侧输出处理逻辑）
        st.info("💡 处理逻辑：基于您输入的5个数值特征，使用最近邻算法（Euclidean距离）从知识库中匹配最相似的实验记录，并结合随机森林模型预测容量。工艺方案融合了最相似案例的预处理和后处理步骤，碳化参数直接采用您输入的数值。")
else:
    st.info("👈 请在左侧输入参数，然后点击「生成工艺方案」按钮。右侧将显示完整的工艺方案、容量预测以及参考案例。")

# 底部说明
st.markdown("---")
st.caption("本项目基于 Streamlit Community Cloud 部署 | 知识库包含历史实验数据 | 预测模型：随机森林 | 相似度检索：最近邻")
