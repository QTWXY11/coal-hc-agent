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
st.markdown("**多模型集成预测 + RAG知识库检索 + 智能工艺优化 + 可视化分析 + 详细工艺流程图**")

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

# ========== 10. 生成详细工艺步骤流程图 ==========
def generate_process_flowchart():
    dot = graphviz.Digraph(comment='Process Flow', format='svg')
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.6', ranksep='0.7')
    dot.attr('node', shape='box', style='rounded,filled', fontname='SimHei', width='4.5')
    
    # 定义颜色
    colors = {'原料': '#D4E9D4', '预处理': '#FFEAD6', '碳化': '#FED6D6', 
              '活化': '#E6D6F5', '后处理': '#D6E6F5', '性能': '#F5D6E6'}
    
    # 步骤1: 原料准备
    raw_label = f'''【步骤1】原料准备\n
煤种：{coal_type}（{coal_rank}）\n
● 灰分 {ash:.1f}%，挥发分 {volatile:.1f}%\n
● 建议：若灰分>5%需脱灰预处理\n
● 目标：获得均质高碳前驱体'''
    dot.node('raw', raw_label, fillcolor=colors['原料'])
    
    # 步骤2: 预处理
    pre_label = f'''【步骤2】预处理\n
方式：{pretreatment}\n
'''
    if pretreatment == '酸洗' or pretreatment == '酸洗+海藻酸钠':
        if acid_concentration > 0:
            pre_label += f'''● 酸洗：{acid_concentration:.1f}mol/L，{acid_temp:.0f}℃，{acid_time:.1f}h\n'''
        else:
            pre_label += f'''● 酸洗：4-6mol/L，80-90℃，8-12h（参考）\n'''
    if pretreatment == '碱洗':
        pre_label += f'''● 碱洗：NaOH溶液，80℃，数小时\n'''
    if pretreatment == '氧化活化':
        if preoxidation_temp > 0:
            pre_label += f'''● 预氧化：{preoxidation_temp:.0f}℃，{preoxidation_time:.1f}h\n'''
        else:
            pre_label += f'''● 预氧化：200-300℃，空气气氛，2h\n'''
    if pretreatment == '酸洗+海藻酸钠':
        pre_label += f'''● 海藻酸钠添加：煤:SA=1:1~1:4\n'''
    pre_label += '''\n● 目的：脱灰/引入官能团/调控交联'''
    dot.node('pre', pre_label, fillcolor=colors['预处理'])
    
    # 步骤3: 碳化
    carbon_label = f'''【步骤3】碳化\n
● 终温：{carbon_temp:.0f}℃\n
● 保温：{hold_time:.1f}h\n
● 升温：{heating_rate:.1f}℃/min\n
● 气氛：高纯氩气或氮气\n
● 注意：升温速率影响缺陷密度与层间距\n
● 目标：形成无序碳骨架与初步微晶'''
    dot.node('carbon', carbon_label, fillcolor=colors['碳化'])
    
    # 步骤4: 活化（如果启用）
    act_nodes = []
    if activation_method != '无' and activation_temp > 0:
        act_label = f'''【步骤4】活化\n
方式：{activation_method}\n
活化剂：{activator}\n
温度：{activation_temp:.0f}℃\n
时间：{activation_time:.1f}h\n
配比：煤:活化剂 = {activator_ratio:.1f}\n
● 目的：造孔（微孔/介孔）增加储钠位点\n
● 注意：过量活化会降低首效'''
        dot.node('activation', act_label, fillcolor=colors['活化'])
        act_nodes.append('activation')
    
    # 步骤5: 后处理
    post_label = f'''【步骤5】后处理\n
● 冷却方式：自然冷却（>2h）\n
● 建议：若需提高倍率，可考虑急冷\n
● 后处理：可研磨、筛分至目标粒径\n
● 注意：避免吸潮与氧化'''
    dot.node('post', post_label, fillcolor=colors['后处理'])
    
    # 步骤6: 产物
    product_label = f'''【步骤6】硬碳产物\n
预期微观结构：
● d002：{d002:.3f} nm  (若输入)\n
● La：{La:.2f} nm\n
● ID/IG：{id_ig:.2f}\n
● 比表面积：{ssa:.1f} m²/g\n
● 闭孔：促进平台容量'''
    dot.node('product', product_label, fillcolor='#D6E6F5')
    
    # 步骤7: 性能预测
    input_vec = build_input_vector()
    cap, _ = ensemble_predict(input_vec, available_features)
    perf_label = f'''【步骤7】电化学性能\n
● 预测可逆容量：{cap:.1f} mAh/g\n
● 预期首次库伦效率：75-92%（参考）\n
● 应用场景：钠离子电池负极\n
● 建议：如需高倍率，优化活化条件'''
    dot.node('perf', perf_label, fillcolor=colors['性能'])
    
    # 连接线（带箭头标注）
    dot.edge('raw', 'pre', label='研磨/筛分')
    dot.edge('pre', 'carbon', label='装炉/气氛')
    if act_nodes:
        dot.edge('carbon', 'activation', label='转移')
        dot.edge('activation', 'post', label='冷却')
    else:
        dot.edge('carbon', 'post', label='冷却')
    dot.edge('post', 'product', label='后处理')
    dot.edge('product', 'perf', label='电化学测试')
    
    return dot

# ========== 11. 主区域 Tabs ==========
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 智能预测", "📚 RAG知识检索", "📈 可视化分析", "💡 工艺优化", "🗺️ 工艺流程图"])

# ---------- Tab 1: 智能预测 ----------
with tab1:
    # ⚠️ 按钮就在这里！位于 Tab1 内部
    if st.button("🚀 生成预测与工艺方案", use_container_width=True):
        with st.spinner("正在检索、预测并生成方案..."):
            # TF-IDF 相似度检索
            user_text = f"煤种{coal_type} 灰分{ash}% 挥发分{volatile}% 碳化温度{carbon_temp}℃ 保温{hold_time}h 升温{heating_rate}℃/min"
            user_vec = vectorizer.transform([user_text]).toarray().astype(np.float32)
            distances, indices = index.search(user_vec, k=3)
            similar_df = df.iloc[indices[0]].copy()
            
            # 多模型集成预测
            input_vec = build_input_vector()
            ensemble_pred, pred_dict = ensemble_predict(input_vec, available_features)
            
            # 显示预测结果
            st.subheader("📊 多模型集成预测结果")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🎯 集成预测容量", f"{ensemble_pred:.1f} mAh/g")
            col2.metric("🌲 随机森林", f"{pred_dict['RandomForest']:.1f} mAh/g")
            col3.metric("🌳 GBDT", f"{pred_dict['GBDT']:.1f} mAh/g")
            col4.metric("📈 SVR", f"{pred_dict['SVR']:.1f} mAh/g")
            
            # 相似案例
            st.subheader("🔎 TF-IDF 相似案例检索")
            st.dataframe(similar_df[['coal_type', 'ash', 'volatile', 'carbon_temp', 'hold_time', 'heating_rate', 'capacity', 'ice', 'pretreatment']])
            
            # 大模型生成
            similar_text = ""
            for _, row in similar_df.iterrows():
                similar_text += f"- {row['coal_type']}，灰分{row.get('ash','')}%，容量{row.get('capacity','')}mAh/g\n"
            
            prompt = f"""你是一位煤基硬碳专家。用户参数：
煤种{coal_type}，灰分{ash}%，挥发分{volatile}%，固定碳{fixed_carbon}%，碳含量{carbon_content}%
碳化温度{carbon_temp}℃，保温{hold_time}h，升温{heating_rate}℃/min
预处理{pretreatment}，活化方式{activation_method}，活化剂{activator}
相似案例：{similar_text}
集成模型预测容量：{ensemble_pred:.1f} mAh/g。

请提供：1)完整工艺方案  2)参数优化建议  3)预期性能（容量、ICE、循环）
用条目输出。"""
            llm_response = call_llm(prompt)
            
            st.subheader("📋 大模型生成工艺方案")
            st.markdown(llm_response)

# ---------- Tab 2: RAG 知识检索 ----------
with tab2:
    st.subheader("📚 语义检索增强（RAG）")
    st.caption("输入自然语言问题，系统将从文献知识库中语义检索相关内容，并由大模型生成答案。")
    rag_query = st.text_input("请输入您的问题", placeholder="例如：海藻酸钠造孔对煤基硬碳性能有什么影响？")
    if st.button("🔍 检索知识库", use_container_width=True):
        if rag_query:
            with st.spinner("正在检索知识库并生成回答..."):
                results, scores = rag_search(rag_query, top_k=3)
                st.subheader("📄 检索到的相关文献（Top 3）")
                for i, row in enumerate(results):
                    with st.expander(f"📖 文献 {i+1} (相似度: {scores[0][i]:.3f})"):
                        st.markdown(f"**煤种**: {row['coal_type']}")
                        st.markdown(f"**参数**: 灰分{row.get('ash','')}%, 挥发分{row.get('volatile','')}%, 碳化温度{row.get('carbon_temp','')}℃")
                        st.markdown(f"**性能**: 容量{row.get('capacity','')}mAh/g, ICE{row.get('ice','')}%")
                        st.markdown(f"**预处理**: {row.get('pretreatment','')}")
                context_text = "\n".join([f"文献{i+1}: {row['text_desc']}" for i, row in enumerate(results)])
                rag_prompt = f"""基于以下文献信息回答用户问题：
问题：{rag_query}

文献资料：
{context_text}

请基于上述文献内容回答，如果文献中没有相关信息，请明确说明。"""
                rag_answer = call_llm(rag_prompt)
                st.subheader("💡 智能回答")
                st.markdown(rag_answer)
        else:
            st.warning("请输入问题后再检索。")

# ---------- Tab 3: 可视化分析 ----------
with tab3:
    st.subheader("📈 关键参数对容量的影响趋势")
    col1, col2 = st.columns(2)
    with col1:
        visualize_param = st.selectbox("选择参数", available_features)
    with col2:
        param_range_steps = st.slider("参数变化步数", 5, 20, 10)
    if st.button("📊 生成趋势图", use_container_width=True):
        fixed_values = {feat: df[feat].median() for feat in available_features}
        current_input = build_input_vector()
        for i, feat in enumerate(available_features):
            fixed_values[feat] = current_input[i]
        min_val = df[visualize_param].min()
        max_val = df[visualize_param].max()
        param_range = np.linspace(min_val, max_val, param_range_steps)
        fig = plot_parameter_trend(visualize_param, param_range, fixed_values, available_features)
        st.pyplot(fig)
        
        st.subheader("📊 参数相关性热力图")
        corr_cols = [f for f in available_features if f in df.columns] + ['capacity']
        if 'ice' in df.columns:
            corr_cols.append('ice')
        corr_df = df[corr_cols].corr()
        fig2, ax2 = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='coolwarm', ax=ax2)
        st.pyplot(fig2)

# ---------- Tab 4: 工艺优化 ----------
with tab4:
    st.subheader("💡 基于大模型的工艺优化建议")
    st.caption("输入当前工艺参数，系统将给出针对性的优化建议。")
    if st.button("🔧 生成优化建议", use_container_width=True):
        with st.spinner("正在生成优化建议..."):
            input_vec = build_input_vector()
            ensemble_pred, _ = ensemble_predict(input_vec, available_features)
            
            opt_prompt = f"""你是一位煤基硬碳工艺优化专家。当前工艺参数：
- 煤种：{coal_type}，煤阶：{coal_rank}
- 灰分：{ash}%，挥发分：{volatile}%，固定碳：{fixed_carbon}%，碳含量：{carbon_content}%
- 预处理：{pretreatment}
- 碳化温度：{carbon_temp}℃，保温：{hold_time}h，升温速率：{heating_rate}℃/min
- 活化方式：{activation_method}，活化剂：{activator}，活化温度：{activation_temp}℃，活化时间：{activation_time}h
- 当前预测容量：{ensemble_pred:.1f} mAh/g
- 目标容量：{target_cap if target_cap > 0 else '未设定'} mAh/g

请给出具体、量化的优化建议。"""
            opt_response = call_llm(opt_prompt)
            st.markdown(opt_response)

# ---------- Tab 5: 工艺流程图 ----------
with tab5:
    st.subheader("🗺️ 详细工艺步骤指导流程图")
    st.caption("以下为根据当前输入参数生成的完整工艺指导流程，每步包含具体操作与建议。")
    if st.button("🔄 生成流程图", use_container_width=True):
        try:
            dot = generate_process_flowchart()
            st.graphviz_chart(dot)
        except Exception as e:
            st.error(f"生成流程图失败：{e}。请确保已正确安装 graphviz 系统工具。")
    else:
        st.info("👈 点击「生成流程图」按钮，系统将根据您输入的参数生成详细的工艺步骤指导图。")

# ========== 底部 ==========
st.markdown("---")
st.caption("煤基硬碳AI工艺智能体 v2.0 | 多模型集成 + RAG语义检索 + 可视化分析 + 详细工艺流程图 | Powered by Streamlit")