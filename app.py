import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import requests
import json
from sentence_transformers import SentenceTransformer
import graphviz
import warnings
warnings.filterwarnings('ignore')

# ========== 页面配置 ==========
st.set_page_config(page_title="煤基硬碳AI工艺智能体 v2.0", layout="wide")
st.title("🏭 煤基硬碳合成工艺AI智能体 v2.0")
st.markdown("**多模型集成预测 + RAG知识库检索 + 智能工艺优化 + 可视化分析 + 工艺流程图**")

# ========== 完整特征列 ==========
ALL_FEATURES = [
    'ash', 'volatile', 'fixed_carbon', 'carbon_content',
    'hydrogen_content', 'oxygen_content', 'vitrinite_content',
    'acid_concentration', 'acid_temp', 'acid_time',
    'preoxidation_temp', 'preoxidation_time',
    'carbon_temp', 'hold_time', 'heating_rate',
    'activation_temp', 'activation_time', 'activator_ratio',
    'd002', 'La', 'Lc', 'id_ig', 'ssa', 'micropore_volume'
]

# ========== 1. 数据加载与模型训练 ==========
@st.cache_resource
def load_data():
    df = pd.read_csv("data.csv")
    available_features = [f for f in ALL_FEATURES if f in df.columns]
    X = df[available_features].fillna(df[available_features].median())
    y = df['capacity']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    gbdt = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    rf.fit(X_scaled, y)
    gbdt.fit(X_scaled, y)
    svr.fit(X_scaled, y)
    models = {'RandomForest': rf, 'GBDT': gbdt, 'SVR': svr}
    df['text_desc'] = df.apply(lambda row:
        f"煤种{row['coal_type']} 灰分{row.get('ash','')}% 挥发分{row.get('volatile','')}% "
        f"碳化温度{row.get('carbon_temp','')}℃ 保温{row.get('hold_time','')}h 升温{row.get('heating_rate','')}℃/min "
        f"预处理{row.get('pretreatment','')} 容量{row.get('capacity','')}mAh/g", axis=1)
    vectorizer = TfidfVectorizer()
    text_vectors = vectorizer.fit_transform(df['text_desc']).toarray().astype(np.float32)
    index = faiss.IndexFlatL2(text_vectors.shape[1])
    index.add(text_vectors)
    return df, available_features, scaler, models, vectorizer, index

df, available_features, scaler, models, vectorizer, index = load_data()

# ========== 2. Sentence Embedding ==========
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

embedding_model = load_embedding_model()

# ========== 3. RAG 知识库 ==========
@st.cache_resource
def build_rag_knowledge_base():
    texts = df['text_desc'].tolist()
    embeddings = embedding_model.encode(texts, convert_to_numpy=True).astype(np.float32)
    rag_index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss.normalize_L2(embeddings)
    rag_index.add(embeddings)
    return rag_index, texts

rag_index, rag_texts = build_rag_knowledge_base()

# ========== 4. 大模型调用 ==========
def call_llm(prompt):
    api_key = st.secrets.get("API_KEY")
    base_url = st.secrets.get("BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model_name = st.secrets.get("MODEL", "qwen3.7-plus")
    if not api_key:
        return "⚠️ 未配置 API Key，请检查 Secrets 设置。"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 1500
    }
    try:
        resp = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=90)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        else:
            return f"❌ API 调用失败：{resp.text}"
    except requests.exceptions.Timeout:
        return "❌ 请求超时，请稍后重试。"
    except Exception as e:
        return f"❌ 请求异常：{str(e)}"

# ========== 5. 集成预测 ==========
def ensemble_predict(input_values, available_features):
    input_dict = {feat: input_values[i] for i, feat in enumerate(available_features)}
    input_df = pd.DataFrame([input_dict])[available_features]
    input_scaled = scaler.transform(input_df)
    predictions = {}
    for name, model in models.items():
        predictions[name] = model.predict(input_scaled)[0]
    return np.mean(list(predictions.values())), predictions

# ========== 6. RAG 检索 ==========
def rag_search(query, top_k=3):
    query_vec = embedding_model.encode([query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(query_vec)
    distances, indices = rag_index.search(query_vec, top_k)
    results = [df.iloc[idx] for idx in indices[0] if idx < len(df)]
    return results, distances

# ========== 7. 可视化函数 ==========
def plot_parameter_trend(param_name, param_range, fixed_values, available_features):
    results = []
    for val in param_range:
        input_dict = {feat: fixed_values.get(feat, df[feat].median()) for feat in available_features}
        input_dict[param_name] = val
        input_df = pd.DataFrame([input_dict])[available_features]
        input_scaled = scaler.transform(input_df)
        pred, _ = ensemble_predict(input_scaled[0], available_features)
        results.append({'param': val, 'capacity': pred})
    df_plot = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df_plot['param'], df_plot['capacity'], 'o-', linewidth=2, markersize=6, color='#2E86AB')
    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel('预测可逆容量 (mAh/g)', fontsize=12)
    ax.set_title(f'{param_name} 对容量的影响趋势', fontsize=14)
    ax.grid(True, alpha=0.3)
    return fig

# ========== 8. 侧边栏输入 ==========
st.sidebar.header("⚙️ 输入原料特性与目标")

coal_type = st.sidebar.selectbox("煤种", df['coal_type'].unique())
coal_rank = st.sidebar.selectbox("煤阶", ["低阶", "中阶", "高阶"])
ash = st.sidebar.number_input("灰分 (%)", 0.0, 30.0, 8.0, step=0.1)
volatile = st.sidebar.number_input("挥发分 (%)", 0.0, 55.0, 35.0, step=0.1)
fixed_carbon = st.sidebar.number_input("固定碳 (%)", 0.0, 100.0, 57.0, step=0.1)
carbon_content = st.sidebar.number_input("碳含量 (%)", 0.0, 100.0, 75.0, step=0.1)
hydrogen_content = st.sidebar.number_input("氢含量 (%)", 0.0, 10.0, 4.8, step=0.1)
oxygen_content = st.sidebar.number_input("氧含量 (%)", 0.0, 30.0, 15.0, step=0.1)
vitrinite_content = st.sidebar.number_input("镜质组含量 (%)", 0.0, 100.0, 60.0, step=0.1)

st.sidebar.markdown("---")
pretreatment = st.sidebar.selectbox("预处理方式", ["无", "酸洗", "碱洗", "氧化活化", "酸洗+海藻酸钠", "水蒸气活化", "CO2活化", "KOH活化"])
acid_concentration = st.sidebar.number_input("酸浓度 (mol/L) 可选", 0.0, 10.0, 4.0, step=0.1)
acid_temp = st.sidebar.number_input("酸洗温度 (℃) 可选", 0.0, 150.0, 80.0, step=1.0)
acid_time = st.sidebar.number_input("酸洗时间 (h) 可选", 0.0, 24.0, 12.0, step=0.5)
preoxidation_temp = st.sidebar.number_input("预氧化温度 (℃) 可选", 0.0, 500.0, 0.0, step=5.0)
preoxidation_time = st.sidebar.number_input("预氧化时间 (h) 可选", 0.0, 10.0, 0.0, step=0.5)

st.sidebar.markdown("---")
carbon_temp = st.sidebar.number_input("碳化终温 (℃)", 800.0, 2000.0, 1300.0, step=10.0)
hold_time = st.sidebar.number_input("保温时间 (h)", 0.5, 6.0, 2.0, step=0.5)
heating_rate = st.sidebar.number_input("升温速率 (℃/min)", 1.0, 20.0, 5.0, step=1.0)

st.sidebar.markdown("---")
activation_method = st.sidebar.selectbox("活化方式", ["无", "物理活化", "化学活化"])
activator = st.sidebar.selectbox("活化剂种类", ["无", "KOH", "CO2", "水蒸气", "海藻酸钠"])
activation_temp = st.sidebar.number_input("活化温度 (℃) 可选", 0.0, 1200.0, 0.0, step=10.0)
activation_time = st.sidebar.number_input("活化时间 (h) 可选", 0.0, 6.0, 0.0, step=0.5)
activator_ratio = st.sidebar.number_input("活化剂配比 (煤:活化剂) 可选", 0.0, 5.0, 0.0, step=0.5)

st.sidebar.markdown("---")
use_structure = st.sidebar.checkbox("输入微观结构数据（可选）", value=False)
if use_structure:
    d002 = st.sidebar.number_input("d002 层间距 (nm)", 0.35, 0.45, 0.375, step=0.001)
    La = st.sidebar.number_input("La 微晶尺寸 (nm)", 1.0, 5.0, 2.5, step=0.01)
    Lc = st.sidebar.number_input("Lc 微晶尺寸 (nm)", 0.5, 2.0, 0.85, step=0.01)
    id_ig = st.sidebar.number_input("ID/IG 缺陷比", 0.5, 2.0, 1.0, step=0.01)
    ssa = st.sidebar.number_input("比表面积 (m²/g)", 0.0, 500.0, 120.0, step=1.0)
    micropore_volume = st.sidebar.number_input("微孔孔容 (cm³/g)", 0.0, 1.0, 0.12, step=0.001)

st.sidebar.markdown("---")
target_cap = st.sidebar.number_input("目标容量 (mAh/g) 可选", 200.0, 500.0, 300.0, step=10.0)

# ========== 9. 构造输入向量 ==========
def build_input_vector():
    input_dict = {}
    for feat in available_features:
        if feat == 'ash': input_dict[feat] = ash
        elif feat == 'volatile': input_dict[feat] = volatile
        elif feat == 'fixed_carbon': input_dict[feat] = fixed_carbon
        elif feat == 'carbon_content': input_dict[feat] = carbon_content
        elif feat == 'hydrogen_content': input_dict[feat] = hydrogen_content
        elif feat == 'oxygen_content': input_dict[feat] = oxygen_content
        elif feat == 'vitrinite_content': input_dict[feat] = vitrinite_content
        elif feat == 'acid_concentration': input_dict[feat] = acid_concentration
        elif feat == 'acid_temp': input_dict[feat] = acid_temp
        elif feat == 'acid_time': input_dict[feat] = acid_time
        elif feat == 'preoxidation_temp': input_dict[feat] = preoxidation_temp
        elif feat == 'preoxidation_time': input_dict[feat] = preoxidation_time
        elif feat == 'carbon_temp': input_dict[feat] = carbon_temp
        elif feat == 'hold_time': input_dict[feat] = hold_time
        elif feat == 'heating_rate': input_dict[feat] = heating_rate
        elif feat == 'activation_temp': input_dict[feat] = activation_temp
        elif feat == 'activation_time': input_dict[feat] = activation_time
        elif feat == 'activator_ratio': input_dict[feat] = activator_ratio
        elif feat == 'd002': input_dict[feat] = d002 if use_structure else df[feat].median()
        elif feat == 'La': input_dict[feat] = La if use_structure else df[feat].median()
        elif feat == 'Lc': input_dict[feat] = Lc if use_structure else df[feat].median()
        elif feat == 'id_ig': input_dict[feat] = id_ig if use_structure else df[feat].median()
        elif feat == 'ssa': input_dict[feat] = ssa if use_structure else df[feat].median()
        elif feat == 'micropore_volume': input_dict[feat] = micropore_volume if use_structure else df[feat].median()
        else: input_dict[feat] = 0.0
    return [input_dict[f] for f in available_features]

# ========== 10. 生成工艺流程图 ==========
def generate_process_flowchart():
    dot = graphviz.Digraph(comment='Process Flow', format='svg')
    dot.attr(rankdir='LR', splines='ortho', nodesep='0.5', ranksep='0.5')
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='#E8F4FD', fontname='SimHei')
    
    # 节点名称（显示参数值）
    def fmt(val, unit=''):
        return f'{val:.1f}{unit}' if isinstance(val, float) else str(val)
    
    # 原料节点
    raw_label = f'原料煤\n{coal_type} ({coal_rank})\n灰分 {fmt(ash)}%\n挥发分 {fmt(volatile)}%'
    dot.node('raw', raw_label, fillcolor='#D4E9D4')
    
    # 预处理节点
    pre_label = f'预处理\n{pretreatment}'
    if pretreatment in ['酸洗', '酸洗+海藻酸钠'] and acid_concentration > 0:
        pre_label += f'\n酸 {fmt(acid_concentration)}mol/L {fmt(acid_temp)}℃ {fmt(acid_time)}h'
    if preoxidation_temp > 0:
        pre_label += f'\n预氧化 {fmt(preoxidation_temp)}℃ {fmt(preoxidation_time)}h'
    dot.node('pre', pre_label, fillcolor='#FFEAD6')
    
    # 碳化节点
    carbon_label = f'碳化\n{fmt(carbon_temp)}℃\n保温 {fmt(hold_time)}h\n升温 {fmt(heating_rate)}℃/min'
    dot.node('carbon', carbon_label, fillcolor='#FED6D6')
    
    # 活化节点
    if activation_method != '无' and activation_temp > 0:
        act_label = f'活化\n{activation_method}\n{activator}\n{fmt(activation_temp)}℃ {fmt(activation_time)}h\n配比 {fmt(activator_ratio)}'
        dot.node('activation', act_label, fillcolor='#E6D6F5')
    
    # 产物节点
    product_label = '煤基硬碳\n微观结构\n'
    if use_structure:
        product_label += f'd002 {fmt(d002)}nm\nLa {fmt(La)}nm\nID/IG {fmt(id_ig)}\nSSA {fmt(ssa)}m²/g'
    else:
        product_label += '(未输入结构参数)'
    dot.node('product', product_label, fillcolor='#D6E6F5')
    
    # 性能预测节点
    # 先获取预测值
    input_vec = build_input_vector()
    cap, _ = ensemble_predict(input_vec, available_features)
    cap_val = f'{cap:.1f}'
    perf_label = f'电化学性能\n可逆容量 {cap_val} mAh/g\nICE (需实测)'
    dot.node('perf', perf_label, fillcolor='#F5D6E6')
    
    # 连接关系
    dot.edge('raw', 'pre', label='')
    dot.edge('pre', 'carbon', label='')
    if activation_method != '无' and activation_temp > 0:
        dot.edge('carbon', 'activation', label='')
        dot.edge('activation', 'product', label='')
    else:
        dot.edge('carbon', 'product', label='')
    dot.edge('product', 'perf', label='')
    
    return dot

# ========== 11. 主区域 Tabs ==========
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 智能预测", "📚 RAG知识检索", "📈 可视化分析", "💡 工艺优化", "🗺️ 工艺流程图"])

# 其他 Tab 内容与之前相同（此处省略重复代码，实际部署时必须保留完整）
# 下面只给出 Tab5 的完整内容，前四个 Tab 请按之前版本保留。

# ---------- Tab 5: 工艺流程图 ----------
with tab5:
    st.subheader("🗺️ 基于当前参数的工艺流程图")
    st.caption("下图展示了从原料到产物的完整工艺路径，节点中显示了您当前输入的参数值。")
    if st.button("🔄 生成流程图", use_container_width=True):
        try:
            dot = generate_process_flowchart()
            st.graphviz_chart(dot)
        except Exception as e:
            st.error(f"生成流程图失败：{e}。请确保已安装 graphviz 系统工具（已创建 packages.txt）。")
    else:
        st.info("👈 点击「生成流程图」按钮，将根据您左侧输入的参数绘制工艺路径图。")