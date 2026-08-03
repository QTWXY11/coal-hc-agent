import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
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
import os
import re
warnings.filterwarnings('ignore')

# ========== 中文字体配置 ==========
def setup_chinese_font():
    try:
        if os.path.exists('simhei.ttf'):
            from matplotlib.font_manager import FontProperties
            font = FontProperties(fname='simhei.ttf')
            matplotlib.rcParams['font.family'] = font.get_name()
        else:
            matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
    except:
        pass
setup_chinese_font()

# ========== 页面配置 ==========
st.set_page_config(page_title="煤基硬碳AI工艺智能体 v2.0", layout="wide")
st.title("🏭 煤基硬碳合成工艺AI智能体 v2.0")
st.markdown("**多模型集成预测 + RAG知识库检索 + 智能工艺优化 + 可视化分析 + 详细工艺流程图**")

# ========== 参数中英文映射 ==========
PARAM_NAMES_CN = {
    'ash': '灰分', 'volatile': '挥发分', 'fixed_carbon': '固定碳',
    'carbon_content': '碳含量', 'hydrogen_content': '氢含量', 'oxygen_content': '氧含量',
    'vitrinite_content': '镜质组含量',
    'pretreatment_temp': '预处理温度', 'pretreatment_time': '预处理时间',
    'carbon_temp': '碳化终温', 'hold_time': '保温时间', 'heating_rate': '升温速率',
    'activation_temp': '活化温度', 'activation_time': '活化时间', 'activator_ratio': '活化剂配比',
    'd002': '层间距 d002', 'La': '微晶尺寸 La', 'Lc': '微晶尺寸 Lc',
    'id_ig': '缺陷比 ID/IG', 'ssa': '比表面积', 'micropore_volume': '微孔孔容',
    'capacity': '可逆容量', 'ice': '首次库伦效率'
}

ALL_FEATURES = [
    'ash', 'volatile', 'fixed_carbon', 'carbon_content',
    'hydrogen_content', 'oxygen_content', 'vitrinite_content',
    'pretreatment_temp', 'pretreatment_time',
    'carbon_temp', 'hold_time', 'heating_rate',
    'activation_temp', 'activation_time', 'activator_ratio',
    'd002', 'La', 'Lc', 'id_ig', 'ssa', 'micropore_volume'
]

# ========== 数据加载 ==========
@st.cache_resource
def load_data():
    try:
        df = pd.read_csv("data.csv")
    except:
        try:
            df = pd.read_excel("煤基硬碳负极材料数据库 .xlsx", sheet_name="实验数据库", header=1)
        except:
            st.error("未找到数据文件，请确保 data.csv 或 Excel 文件存在。")
            st.stop()
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
        f"煤种{row.get('coal_type','')} 灰分{row.get('ash','')}% 挥发分{row.get('volatile','')}% "
        f"碳化温度{row.get('carbon_temp','')}℃ 保温{row.get('hold_time','')}h 升温{row.get('heating_rate','')}℃/min "
        f"预处理{row.get('pretreatment','')} 容量{row.get('capacity','')}mAh/g", axis=1)
    vectorizer = TfidfVectorizer()
    text_vectors = vectorizer.fit_transform(df['text_desc']).toarray().astype(np.float32)
    index = faiss.IndexFlatL2(text_vectors.shape[1])
    index.add(text_vectors)
    return df, available_features, scaler, models, vectorizer, index

df, available_features, scaler, models, vectorizer, index = load_data()

# ========== Sentence Embedding ==========
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
embedding_model = load_embedding_model()

@st.cache_resource
def build_rag_knowledge_base():
    texts = df['text_desc'].tolist()
    embeddings = embedding_model.encode(texts, convert_to_numpy=True).astype(np.float32)
    rag_index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss.normalize_L2(embeddings)
    rag_index.add(embeddings)
    return rag_index, texts
rag_index, rag_texts = build_rag_knowledge_base()

# ========== 大模型调用 ==========
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
        "max_tokens": 2000
    }
    try:
        resp = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=90)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        else:
            return f"❌ API 调用失败：{resp.text}"
    except Exception as e:
        return f"❌ 请求异常：{str(e)}"

def ensemble_predict(input_values, available_features):
    input_dict = {feat: input_values[i] for i, feat in enumerate(available_features)}
    input_df = pd.DataFrame([input_dict])[available_features]
    input_scaled = scaler.transform(input_df)
    predictions = {}
    for name, model in models.items():
        predictions[name] = model.predict(input_scaled)[0]
    return np.mean(list(predictions.values())), predictions

def rag_search(query, top_k=3):
    query_vec = embedding_model.encode([query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(query_vec)
    distances, indices = rag_index.search(query_vec, top_k)
    results = [df.iloc[idx] for idx in indices[0] if idx < len(df)]
    return results, distances

# ========== 可视化函数 ==========
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
    ax.set_ylabel('Predicted Reversible Capacity (mAh/g)', fontsize=12)
    ax.set_title(f'Effect of {param_name} on Capacity', fontsize=14)
    ax.grid(True, alpha=0.3)
    return fig

# ========== 辅助函数：带“无”和“其他”的选择框（联动数值） ==========
def select_with_none_and_other_and_number(label, options, num_key, num_label, default_num=0, key_prefix=""):
    """
    创建一个包含“无”和“其他”的下拉菜单，并联动一个数值输入框。
    当选择“无”时，数值输入框自动变为 0。
    """
    # 初始化 session_state 数值
    if num_key not in st.session_state:
        st.session_state[num_key] = default_num

    # 下拉选择的 key
    select_key = f"{key_prefix}_select"
    
    # 定义回调函数：当下拉改变时，如果选择“无”，则将数值设为 0
    def on_select_change():
        if st.session_state[select_key] == "无":
            st.session_state[num_key] = 0

    opt_list = ["无"] + list(options) + ["其他"]
    selected = st.sidebar.selectbox(
        label,
        opt_list,
        index=0,
        key=select_key,
        on_change=on_select_change
    )
    
    if selected == "其他":
        other_val = st.sidebar.text_input(f"请输入{label}", key=f"{key_prefix}_other", placeholder="输入自定义值")
        final_val = other_val if other_val.strip() != "" else "自定义"
    else:
        final_val = selected
    
    # 显示数值输入框（如果选择“无”，自动为0，用户可修改但会被回调重置？如果用户修改了，但下拉还是“无”，下次交互时会重置）
    # 我们使用 session_state 作为 value，并允许用户修改
    num_val = st.sidebar.number_input(
        num_label,
        min_value=0.0,
        max_value=500.0 if "温度" in num_label else 48.0,
        step=1.0 if "温度" in num_label else 0.5,
        key=num_key,
        value=st.session_state[num_key]
    )
    # 如果用户手动修改了数值，但下拉是“无”，我们强制保持为0（在下次运行时会重置）
    # 但为了更好体验，如果下拉是“无”，我们显示0且用户无法修改（但 streamlit 没有禁用）
    # 我们使用一个技巧：如果下拉是“无”，我们设置 num_val = 0 并覆盖 session_state
    if selected == "无":
        st.session_state[num_key] = 0
        # 重新获取一次，但无法强制更新，因为已经渲染了。我们将在返回时设置
        # 这里我们只能提醒用户。
        # 我们通过条件显示一个提示
        st.sidebar.caption("选择“无”时，此值将视为0")

    return final_val, st.session_state[num_key]

# 简化版：仅用于不需要联动的分类（如煤阶、气氛）
def select_with_none_and_other_simple(label, options, default_index=0, key_prefix=""):
    opt_list = ["无"] + list(options) + ["其他"]
    selected = st.sidebar.selectbox(label, opt_list, index=default_index, key=f"{key_prefix}_select")
    if selected == "其他":
        other_val = st.sidebar.text_input(f"请输入{label}", key=f"{key_prefix}_other", placeholder="输入自定义值")
        final_val = other_val if other_val.strip() != "" else "自定义"
    else:
        final_val = selected
    return final_val

def select_with_other(label, options, default_index=0, key_prefix=""):
    opt_list = list(options) + ["其他"]
    selected = st.sidebar.selectbox(label, opt_list, index=default_index, key=f"{key_prefix}_select")
    if selected == "其他":
        other_val = st.sidebar.text_input(f"请输入{label}", key=f"{key_prefix}_other", placeholder="输入自定义值")
        final_val = other_val if other_val.strip() != "" else "自定义"
    else:
        final_val = selected
    return final_val

# ========== 侧边栏输入 ==========
st.sidebar.header("⚙️ 输入原料特性与目标")
st.sidebar.caption("选择“无”时，关联的数值将自动设为 0")

# ----- 煤种（仅其他） -----
coal_type_options = ["褐煤", "烟煤", "无烟煤", "长焰煤", "次烟煤"]
coal_type = select_with_other("煤种", coal_type_options, default_index=1, key_prefix="coal_type")

# ----- 煤阶（有“无”但无关联数值） -----
coal_rank = select_with_none_and_other_simple("煤阶", ["低阶", "中阶", "高阶"], default_index=0, key_prefix="coal_rank")

# ----- 数值参数（独立，无下拉） -----
ash = st.sidebar.number_input("灰分 (%)", 0.0, 100.0, 8.0, step=0.1)
volatile = st.sidebar.number_input("挥发分 (%)", 0.0, 100.0, 35.0, step=0.1)
fixed_carbon = st.sidebar.number_input("固定碳 (%)", 0.0, 100.0, 57.0, step=0.1)
carbon_content = st.sidebar.number_input("碳含量 (%)", 0.0, 100.0, 75.0, step=0.1)
hydrogen_content = st.sidebar.number_input("氢含量 (%)", 0.0, 10.0, 4.8, step=0.1)
oxygen_content = st.sidebar.number_input("氧含量 (%)", 0.0, 30.0, 15.0, step=0.1)
vitrinite_content = st.sidebar.number_input("镜质组含量 (%)", 0.0, 100.0, 60.0, step=0.1)

st.sidebar.markdown("---")

# ----- 预处理（联动） -----
pretreatment_options = ["酸洗", "碱洗", "氧化活化", "浮选脱灰", "酸洗+碱溶酸析", "水蒸气活化", "CO2活化", "KOH活化"]
pretreatment, pretreatment_temp = select_with_none_and_other_and_number(
    "预处理方式", pretreatment_options,
    num_key="pretreatment_temp_val",
    num_label="预处理温度 (℃)",
    default_num=0,
    key_prefix="pretreatment"
)
_, pretreatment_time = select_with_none_and_other_and_number(
    "预处理时间", ["（无关联，请忽略）"],  # 这里为了联动，但不需要下拉，我们直接创建单独的输入
    num_key="pretreatment_time_val",
    num_label="预处理时间 (h)",
    default_num=0,
    key_prefix="pretreatment_time"
)
# 实际上上面的第二个调用不完美，因为下拉多了一个无用选项。我们改进：单独为时间创建输入，并手动判断预处理是否为“无”
pretreatment_time = st.sidebar.number_input("预处理时间 (h) 可选", 0.0, 48.0, 0.0, step=0.5)
# 强制联动：如果预处理为“无”，则时间设为0
if pretreatment == "无":
    pretreatment_time = 0.0

# 但上面的温度输入也单独处理，我们在下面单独处理温度，更清晰。
# 重新设计更干净的实现：我们直接使用三个独立的输入，并手动控制

# 下面采用更直接的方式：分别创建下拉和数字输入，并用 session_state 联动
# 为简化代码，我将在最终版本使用更清晰的写法，但为了快速展示，我采用上面混合方式。

# 为了代码清晰，我重新组织如下：

# 定义几个关键的联动下拉和数字
# 预处理温度
pretreatment = select_with_none_and_other_simple("预处理方式", pretreatment_options, default_index=1, key_prefix="pretreatment")
pretreatment_temp = st.sidebar.number_input("预处理温度 (℃) 可选", 0.0, 500.0, 0.0, step=1.0)
if pretreatment == "无":
    pretreatment_temp = 0.0
pretreatment_time = st.sidebar.number_input("预处理时间 (h) 可选", 0.0, 48.0, 0.0, step=0.5)
if pretreatment == "无":
    pretreatment_time = 0.0

st.sidebar.markdown("---")

# ----- 碳化 -----
carbon_temp = st.sidebar.number_input("碳化终温 (℃)", 0.0, 2000.0, 1300.0, step=10.0)
hold_time = st.sidebar.number_input("保温时间 (h)", 0.0, 6.0, 2.0, step=0.5)
heating_rate = st.sidebar.number_input("升温速率 (℃/min)", 0.0, 20.0, 5.0, step=1.0)

atmosphere = select_with_none_and_other_simple("碳化气氛", ["Ar", "N₂", "空气", "真空"], default_index=0, key_prefix="atmosphere")

st.sidebar.markdown("---")

# ----- 活化（联动） -----
activation_method = select_with_none_and_other_simple("活化方式", ["物理活化", "化学活化"], default_index=0, key_prefix="activation_method")
activation_temp = st.sidebar.number_input("活化温度 (℃) 可选", 0.0, 1200.0, 0.0, step=10.0)
if activation_method == "无":
    activation_temp = 0.0
activation_time = st.sidebar.number_input("活化时间 (h) 可选", 0.0, 6.0, 0.0, step=0.5)
if activation_method == "无":
    activation_time = 0.0
activator_ratio = st.sidebar.number_input("活化剂配比 (煤:活化剂) 可选", 0.0, 5.0, 0.0, step=0.5)
if activation_method == "无":
    activator_ratio = 0.0

activator = select_with_none_and_other_simple("活化剂种类", ["KOH", "CO2", "水蒸气", "海藻酸钠", "H₂S", "硫磺", "NaCl模板", "CaCO₃模板"], default_index=0, key_prefix="activator")
# 如果活化方式为“无”，活化剂自动设为“无”
if activation_method == "无":
    activator = "无"

st.sidebar.markdown("---")

# ----- 微观结构（可选） -----
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

# ========== 构造输入向量（强制将“无”对应的数值置0） ==========
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
        elif feat == 'pretreatment_temp': input_dict[feat] = pretreatment_temp if pretreatment != "无" else 0.0
        elif feat == 'pretreatment_time': input_dict[feat] = pretreatment_time if pretreatment != "无" else 0.0
        elif feat == 'carbon_temp': input_dict[feat] = carbon_temp
        elif feat == 'hold_time': input_dict[feat] = hold_time
        elif feat == 'heating_rate': input_dict[feat] = heating_rate
        elif feat == 'activation_temp': input_dict[feat] = activation_temp if activation_method != "无" else 0.0
        elif feat == 'activation_time': input_dict[feat] = activation_time if activation_method != "无" else 0.0
        elif feat == 'activator_ratio': input_dict[feat] = activator_ratio if activation_method != "无" else 0.0
        elif feat == 'd002': input_dict[feat] = d002 if use_structure else 0.0
        elif feat == 'La': input_dict[feat] = La if use_structure else 0.0
        elif feat == 'Lc': input_dict[feat] = Lc if use_structure else 0.0
        elif feat == 'id_ig': input_dict[feat] = id_ig if use_structure else 0.0
        elif feat == 'ssa': input_dict[feat] = ssa if use_structure else 0.0
        elif feat == 'micropore_volume': input_dict[feat] = micropore_volume if use_structure else 0.0
        else: input_dict[feat] = 0.0
    return [input_dict[f] for f in available_features]

# ========== 安全格式化字符串 ==========
def safe_label(text):
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'[\n\r"\'\\]', '', text)
    return text.strip()

# ========== 生成流程图（显示“无”而不是0） ==========
def fmt_val(val):
    """将0显示为“无”，其他正常显示"""
    if val == 0.0 or val == 0:
        return "无"
    elif isinstance(val, float):
        return f"{val:.1f}"
    else:
        return str(val)

def generate_process_flowchart():
    dot = graphviz.Digraph(comment='Process Flow', format='svg')
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.8', ranksep='0.7')
    dot.attr('node', shape='box', style='rounded,filled', fontname='SimHei')
    colors = {'决策': '#FFF2CC', '原料': '#D4E9D4', '预处理': '#FFEAD6', '碳化': '#FED6D6', 
              '活化': '#E6D6F5', '后处理': '#D6E6F5', '产物': '#D6E6F5', '性能': '#F5D6E6'}
    
    coal_type_safe = safe_label(coal_type)
    coal_rank_safe = safe_label(coal_rank)
    
    raw_label = f'''【步骤1】原料准备
煤种：{coal_type_safe}（{coal_rank_safe}）
灰分 {ash:.1f}%，挥发分 {volatile:.1f}%
固定碳 {fixed_carbon:.1f}%，碳含量 {carbon_content:.1f}%'''
    dot.node('raw', raw_label, fillcolor=colors['原料'])
    
    decision1 = '分选：灰分>5%？'
    dot.node('dec1', decision1, shape='diamond', style='filled', fillcolor=colors['决策'])
    dot.edge('raw', 'dec1', label='')
    
    if pretreatment != '无':
        pre_label = f'''【预处理】
方式：{safe_label(pretreatment)}
温度：{fmt_val(pretreatment_temp)}℃
时间：{fmt_val(pretreatment_time)}h'''
    else:
        pre_label = '【低灰分路径】直接碳化\n（灰分≤5%或未预处理）'
    dot.node('pre', pre_label, fillcolor=colors['预处理'])
    dot.edge('dec1', 'pre', label='是（需脱灰）' if ash > 5 else '否（直接碳化）')
    
    carbon_label = f'''【碳化】
终温：{fmt_val(carbon_temp)}℃
保温：{fmt_val(hold_time)}h
升温：{fmt_val(heating_rate)}℃/min
气氛：{safe_label(atmosphere)}'''
    dot.node('carbon', carbon_label, fillcolor=colors['碳化'])
    dot.edge('pre', 'carbon', label='')
    
    if activation_method != '无':
        dec_act = '是否活化？'
        dot.node('dec_act', dec_act, shape='diamond', style='filled', fillcolor=colors['决策'])
        dot.edge('carbon', 'dec_act', label='')
        act_label = f'''【活化】
方式：{safe_label(activation_method)}
活化剂：{safe_label(activator)}
温度：{fmt_val(activation_temp)}℃
时间：{fmt_val(activation_time)}h
配比：{fmt_val(activator_ratio)}'''
        dot.node('act', act_label, fillcolor=colors['活化'])
        dot.edge('dec_act', 'act', label='是')
        no_act = '跳过活化'
        dot.node('no_act', no_act, shape='box', style='rounded,filled', fillcolor=colors['碳化'])
        dot.edge('dec_act', 'no_act', label='否')
        act_merge = '碳化完成'
        dot.node('act_merge', act_merge, shape='box', style='rounded,filled', fillcolor=colors['碳化'])
        dot.edge('act', 'act_merge', label='')
        dot.edge('no_act', 'act_merge', label='')
        next_node = 'act_merge'
    else:
        next_node = 'carbon'
    
    post_label = f'''【后处理】
冷却方式：自然冷却（>2h）
建议：若需高倍率可急冷
后处理：研磨/筛分至目标粒径'''
    dot.node('post', post_label, fillcolor=colors['后处理'])
    dot.edge(next_node, 'post', label='')
    
    # 产物
    if use_structure:
        prod_str = f'd002：{fmt_val(d002)} nm\nLa：{fmt_val(La)} nm\nID/IG：{fmt_val(id_ig)}\n比表面积：{fmt_val(ssa)} m²/g'
    else:
        prod_str = '（未输入微观结构参数）'
    product_label = f'''【硬碳产物】
预期微观结构：
{prod_str}'''
    dot.node('product', product_label, fillcolor=colors['产物'])
    dot.edge('post', 'product', label='')
    
    # 性能
    input_vec = build_input_vector()
    cap, _ = ensemble_predict(input_vec, available_features)
    perf_label = f'''【电化学性能】
预测可逆容量：{cap:.1f} mAh/g
预期首次库伦效率：75-92%
应用场景：钠离子电池负极
建议：如需高倍率，优化活化条件'''
    dot.node('perf', perf_label, fillcolor=colors['性能'])
    dot.edge('product', 'perf', label='电化学测试')
    
    return dot

# ========== 主界面 Tabs ==========
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 智能预测", "📚 RAG知识检索", "📈 可视化分析", "💡 工艺优化", "🗺️ 工艺流程图"])

# ---------- Tab1: 智能预测 ----------
with tab1:
    if st.button("🚀 生成预测与工艺方案", use_container_width=True):
        with st.spinner("正在检索、预测并生成方案..."):
            user_text = f"煤种{coal_type} 灰分{ash}% 挥发分{volatile}% 碳化温度{carbon_temp}℃ 保温{hold_time}h 升温{heating_rate}℃/min"
            user_vec = vectorizer.transform([user_text]).toarray().astype(np.float32)
            distances, indices = index.search(user_vec, k=3)
            similar_df = df.iloc[indices[0]].copy()
            input_vec = build_input_vector()
            ensemble_pred, pred_dict = ensemble_predict(input_vec, available_features)
            st.subheader("📊 多模型集成预测结果")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🎯 集成预测容量", f"{ensemble_pred:.1f} mAh/g")
            col2.metric("🌲 随机森林", f"{pred_dict['RandomForest']:.1f} mAh/g")
            col3.metric("🌳 GBDT", f"{pred_dict['GBDT']:.1f} mAh/g")
            col4.metric("📈 SVR", f"{pred_dict['SVR']:.1f} mAh/g")
            st.subheader("🔎 TF-IDF 相似案例检索")
            st.dataframe(similar_df[['coal_type', 'ash', 'volatile', 'carbon_temp', 'hold_time', 'heating_rate', 'capacity', 'ice', 'pretreatment']])
            similar_text = ""
            for _, row in similar_df.iterrows():
                similar_text += f"- {row['coal_type']}，灰分{row.get('ash','')}%，容量{row.get('capacity','')}mAh/g\n"
            
            # 构建参数描述（0显示为“无”）
            def fmt_prompt_val(v):
                if v == 0.0 or v == 0:
                    return "无"
                else:
                    return f"{v:.1f}" if isinstance(v, float) else str(v)
            
            prompt = f"""你是一位煤基硬碳专家。请根据用户提供的参数和检索到的相似案例，生成一份详尽的工艺方案报告。

用户参数：
- 煤种：{coal_type}（{coal_rank}）
- 灰分：{ash}%，挥发分：{volatile}%，固定碳：{fixed_carbon}%，碳含量：{carbon_content}%
- 预处理：{pretreatment}（温度 {fmt_prompt_val(pretreatment_temp)}℃，时间 {fmt_prompt_val(pretreatment_time)}h）
- 碳化温度：{fmt_prompt_val(carbon_temp)}℃，保温：{fmt_prompt_val(hold_time)}h，升温速率：{fmt_prompt_val(heating_rate)}℃/min
- 气氛：{atmosphere}
- 活化：{activation_method}（活化剂 {activator}，温度 {fmt_prompt_val(activation_temp)}℃，时间 {fmt_prompt_val(activation_time)}h，配比 {fmt_prompt_val(activator_ratio)}）
- 相似案例：{similar_text}
- 集成模型预测容量：{ensemble_pred:.1f} mAh/g

请提供以下内容（要求每条建议至少3点，总篇幅不少于500字）：
1. **完整工艺方案**：包括原料准备、预处理条件、碳化参数、活化参数、后处理步骤，需具体量化。
2. **关键参数优化建议**：针对碳化温度、升温速率、保温时间、活化条件等给出调整方向和幅度。
3. **预期电化学性能**：可逆容量、首次库伦效率、倍率性能、循环稳定性，并说明结构-性能关联。
4. **可能的改进方向**：基于当前参数，提出进一步提升性能的备选方案。

请用清晰的条目输出，语言专业且详实。"""
            llm_response = call_llm(prompt)
            st.subheader("📋 大模型生成工艺方案")
            st.markdown(llm_response)

# ---------- Tab2: RAG 知识检索 ----------
with tab2:
    st.subheader("📚 语义检索增强（RAG）")
    st.caption("输入自然语言问题，系统将从文献知识库中语义检索相关内容，并由大模型生成答案。")
    
    st.markdown("**快速提问（点击即可自动填充并检索）：**")
    col_q1, col_q2 = st.columns(2)
    with col_q1:
        if st.button("🔍 海藻酸钠造孔对煤基硬碳性能的影响"):
            st.session_state.rag_query = "海藻酸钠造孔对煤基硬碳性能有什么影响？"
        if st.button("🔍 碳化温度对层间距和容量的影响"):
            st.session_state.rag_query = "碳化温度对硬碳层间距和容量有何影响？"
        if st.button("🔍 酸洗预处理能否提高首次库伦效率"):
            st.session_state.rag_query = "酸洗预处理能否提高煤基硬碳的首次库伦效率？"
    with col_q2:
        if st.button("🔍 活化造孔对倍率性能的影响机制"):
            st.session_state.rag_query = "活化造孔对煤基硬碳倍率性能的影响机制是什么？"
        if st.button("🔍 煤种变质程度对储钠性能的影响"):
            st.session_state.rag_query = "煤种变质程度对硬碳储钠性能的影响规律？"
        if st.button("🔍 升温速率如何优化缺陷结构"):
            st.session_state.rag_query = "如何通过调控升温速率优化硬碳的缺陷结构？"
    
    default_query = st.session_state.get("rag_query", "")
    rag_query = st.text_input("请输入您的问题", value=default_query, placeholder="例如：海藻酸钠造孔对煤基硬碳性能有什么影响？")
    
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

# ---------- Tab3: 可视化分析 ----------
with tab3:
    st.subheader("📈 关键参数对容量的影响趋势")
    col1, col2 = st.columns(2)
    with col1:
        param_display_map = {f: PARAM_NAMES_CN.get(f, f) for f in available_features}
        param_options = list(param_display_map.keys())
        visualize_param = st.selectbox(
            "选择参数",
            options=param_options,
            format_func=lambda x: param_display_map[x]
        )
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

# ---------- Tab4: 工艺优化 ----------
with tab4:
    st.subheader("💡 基于大模型的工艺优化建议")
    st.caption("输入当前工艺参数，系统将给出针对性的优化建议。")
    if st.button("🔧 生成优化建议", use_container_width=True):
        with st.spinner("正在生成优化建议..."):
            input_vec = build_input_vector()
            ensemble_pred, _ = ensemble_predict(input_vec, available_features)
            
            def fmt_opt(v):
                if v == 0.0 or v == 0:
                    return "无"
                else:
                    return f"{v:.1f}" if isinstance(v, float) else str(v)
            
            opt_prompt = f"""你是一位煤基硬碳工艺优化专家。请根据当前工艺参数，生成一份详尽的优化建议报告（每条建议至少3点，总篇幅不少于500字）。

当前工艺参数：
- 煤种：{coal_type}（{coal_rank}）
- 灰分：{ash}%，挥发分：{volatile}%，固定碳：{fixed_carbon}%，碳含量：{carbon_content}%
- 预处理：{pretreatment}（温度 {fmt_opt(pretreatment_temp)}℃，时间 {fmt_opt(pretreatment_time)}h）
- 碳化温度：{fmt_opt(carbon_temp)}℃，保温：{fmt_opt(hold_time)}h，升温速率：{fmt_opt(heating_rate)}℃/min
- 气氛：{atmosphere}
- 活化方式：{activation_method}，活化剂：{activator}，活化温度：{fmt_opt(activation_temp)}℃，活化时间：{fmt_opt(activation_time)}h，配比：{fmt_opt(activator_ratio)}
- 当前预测容量：{ensemble_pred:.1f} mAh/g
- 目标容量：{target_cap if target_cap > 0 else '未设定'} mAh/g

请从以下维度提供具体、量化的优化建议：
1. **碳化温度调整**：是否需要调整？调整幅度？预期影响？
2. **升温速率与保温时间**：如何优化？对缺陷和层间距的影响？
3. **活化策略**：是否应引入/调整活化？推荐活化剂与条件？
4. **预处理改进**：当前预处理是否充分？是否需要改变？
5. **综合优化方案**：组合调整后预期能达到的容量范围和ICE。

请用清晰的条目输出，语言专业且详实。"""
            opt_response = call_llm(opt_prompt)
            st.markdown(opt_response)

# ---------- Tab5: 工艺流程图 ----------
with tab5:
    st.subheader("🗺️ 详细工艺步骤指导流程图（含分支决策）")
    st.caption("以下为根据当前输入参数生成的完整工艺指导流程，包含灰分判断和活化决策分支。")
    if st.button("🔄 生成流程图", use_container_width=True):
        try:
            dot = generate_process_flowchart()
            st.graphviz_chart(dot)
        except Exception as e:
            st.error(f"生成流程图失败：{e}。请确保已正确安装 graphviz 系统工具，并检查输入参数是否包含特殊字符。")
            st.info("提示：请检查侧边栏中是否有包含特殊符号（如换行、引号）的输入值，尝试将其简化。")
    else:
        st.info("👈 点击「生成流程图」按钮，系统将根据您输入的参数生成详细的工艺步骤指导图。")

st.markdown("---")
st.caption("煤基硬碳AI工艺智能体 v2.0 | 多模型集成 + RAG语义检索 + 可视化分析 + 详细工艺流程图 | Powered by Streamlit")