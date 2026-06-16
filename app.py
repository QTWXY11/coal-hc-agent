import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import requests
import json

# ========== 页面配置 ==========
st.set_page_config(page_title="煤基硬碳AI工艺智能体", layout="wide")
st.title("🏭 煤基硬碳合成工艺AI智能体")
st.markdown("**左侧输入参数 → 右侧自动检索相似案例 + 容量预测 + 大模型生成智能工艺方案**")

# ========== 加载数据 ==========
@st.cache_resource
def load_data():
    df = pd.read_csv("data.csv")
    # 特征列（用于ML预测）
    feature_cols = ['ash', 'volatile', 'carbon_temp', 'hold_time', 'heating_rate', 
                    'activation_temp', 'activation_time', 'activator_ratio']
    # 构建文本描述（用于相似度检索）
    df['text_desc'] = df.apply(lambda row: 
        f"煤种{row['coal_type']} 灰分{row['ash']}% 挥发分{row['volatile']}% "
        f"碳化温度{row['carbon_temp']}℃ 保温{row['hold_time']}h 升温{row['heating_rate']}℃/min "
        f"活化温度{row['activation_temp']}℃ 活化时间{row['activation_time']}h "
        f"预处理{row['pretreatment']} 容量{row['capacity']}mAh/g", axis=1)
    
    # 训练ML模型
    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df['capacity']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_scaled, y)
    
    # 构建TF-IDF向量索引（用于相似检索）
    vectorizer = TfidfVectorizer()
    text_vectors = vectorizer.fit_transform(df['text_desc']).toarray().astype(np.float32)
    index = faiss.IndexFlatL2(text_vectors.shape[1])
    index.add(text_vectors)
    
    return df, feature_cols, scaler, model, vectorizer, index

df, feature_cols, scaler, model, vectorizer, index = load_data()

# ========== 调用大模型 ==========
def call_llm(prompt):
    api_key = st.secrets.get("API_KEY")
    base_url = st.secrets.get("BASE_URL", "https://api.siliconflow.cn/v1")
    model_name = st.secrets.get("MODEL", "deepseek-ai/DeepSeek-V3")  # 使用稳定模型
    
    if not api_key:
        return "⚠️ 未配置 API Key，请在 `.streamlit/secrets.toml` 中设置 `API_KEY`。"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 1500
    }
    try:
        resp = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=30)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        else:
            return f"❌ API 调用失败：{resp.text}"
    except Exception as e:
        return f"❌ 请求异常：{str(e)}"

# ========== 侧边栏输入 ==========
st.sidebar.header("⚙️ 输入原料特性与目标")
coal_type = st.sidebar.selectbox("煤种", df['coal_type'].unique())
ash = st.sidebar.number_input("灰分 (%)", 0.0, 20.0, 8.0, step=0.1)
volatile = st.sidebar.number_input("挥发分 (%)", 0.0, 50.0, 35.0, step=0.1)
carbon_temp = st.sidebar.slider("碳化温度 (℃)", 800, 2000, 1300, step=10)
hold_time = st.sidebar.slider("保温时间 (h)", 0.5, 5.0, 2.0, step=0.5)
heating_rate = st.sidebar.slider("升温速率 (℃/min)", 1, 20, 5, step=1)
activation_temp = st.sidebar.number_input("活化温度 (℃) 可选", 0, 1000, 0, step=10)
activation_time = st.sidebar.number_input("活化时间 (h) 可选", 0, 6, 0, step=0.5)
activator_ratio = st.sidebar.number_input("煤:活化剂质量比 可选", 0.0, 4.0, 0.0, step=0.5)
pretreatment = st.sidebar.selectbox("预处理方式", ["酸洗", "碱洗", "氧化活化", "酸洗+海藻酸钠", "水蒸气活化", "CO2活化", "KOH活化"])
target_cap = st.sidebar.number_input("目标容量 (mAh/g) 可选", 200, 500, 300, step=10)

# ========== 主区域 ==========
if st.sidebar.button("🚀 生成工艺方案", use_container_width=True):
    with st.spinner("正在检索知识库、预测容量，并召唤大模型思考..."):
        # 1. 构建用户输入向量并检索相似案例
        user_text = f"煤种{coal_type} 灰分{ash}% 挥发分{volatile}% 碳化温度{carbon_temp}℃ 保温{hold_time}h 升温{heating_rate}℃/min 活化温度{activation_temp}℃ 活化时间{activation_time}h 预处理{pretreatment}"
        user_vec = vectorizer.transform([user_text]).toarray().astype(np.float32)
        distances, indices = index.search(user_vec, k=3)
        similar_df = df.iloc[indices[0]].copy()
        
        # 2. 机器学习预测容量
        input_features = np.array([[ash, volatile, carbon_temp, hold_time, heating_rate,
                                    activation_temp, activation_time, activator_ratio]])
        input_scaled = scaler.transform(input_features)
        pred_capacity = model.predict(input_scaled)[0]
        
        # 3. 构建大模型提示词
        similar_cases_text = ""
        for _, row in similar_df.iterrows():
            similar_cases_text += f"- 煤种{row['coal_type']}，灰分{row['ash']}%，挥发分{row['volatile']}%，碳化温度{row['carbon_temp']}℃，保温{row['hold_time']}h，升温{row['heating_rate']}℃/min，容量{row['capacity']}mAh/g，ICE{row['ice']}%\n"
        
        prompt = f"""你是一名煤基硬碳负极材料合成专家。用户需要制备煤基硬碳，原料参数和工艺要求如下：
- 煤种：{coal_type}
- 灰分：{ash}%
- 挥发分：{volatile}%
- 碳化温度：{carbon_temp}℃
- 保温时间：{hold_time}h
- 升温速率：{heating_rate}℃/min
- 活化温度：{activation_temp}℃（0表示无活化）
- 活化时间：{activation_time}h
- 预处理方式：{pretreatment}
- 目标容量：{target_cap} mAh/g

知识库中与用户条件最相似的3条实验记录为：
{similar_cases_text}

请结合这些知识和你的专业经验，为用户提供一套**完整、具体、可执行**的煤基硬碳合成工艺方案。包括：
1. 原料预处理方法（酸洗/碱洗/氧化等，说明浓度、温度、时间）
2. 碳化工艺参数（温度、保温时间、升温速率、气氛）
3. 活化工艺（是否需要活化，活化剂、温度、时间）
4. 后处理（冷却方式等）
5. 预期电化学性能（可逆容量、首次库伦效率、循环稳定性）
6. 优化建议

请用清晰的条目输出，语言专业且简洁。
"""
        llm_response = call_llm(prompt)
        
        # 4. 显示结果
        st.subheader("📋 大模型生成的智能工艺方案")
        st.markdown(llm_response)
        
        st.subheader("📊 容量预测与对比")
        col1, col2 = st.columns(2)
        col1.metric("机器学习预测可逆容量", f"{pred_capacity:.1f} mAh/g")
        col2.metric("最相似案例实际容量", f"{similar_df.iloc[0]['capacity']} mAh/g")
        
        if target_cap:
            st.metric("与目标容量差距", f"{pred_capacity - target_cap:.1f} mAh/g")
        
        st.subheader("🔎 知识库中最相似的3个案例")
        st.dataframe(similar_df[['coal_type', 'ash', 'volatile', 'carbon_temp', 'hold_time', 'heating_rate', 'capacity', 'ice', 'pretreatment']])
        
        st.info("💡 处理逻辑：TF-IDF文本检索相似案例 → 随机森林预测容量 → 大模型（基于用户参数+相似案例）生成完整方案。")

else:
    st.info("👈 请在左侧输入参数，然后点击「生成工艺方案」。右侧将展示大模型生成的专业方案、容量预测及参考案例。")

st.markdown("---")
st.caption("本项目基于 Streamlit + GitHub 部署 | 知识库含30组文献参数 | 机器学习 + 大模型（SiliconFlow API）")