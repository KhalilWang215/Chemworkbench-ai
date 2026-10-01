import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D, SimilarityMaps
import requests
import json
import os
import re
from datetime import datetime

# 页面基础配置
st.set_page_config(page_title="AI化学分子多维工作台", layout="wide", page_icon="🧪")

CACHE_FILE = "chem_cache.json"

# --- 化学式专业下标化 (C8H9NO2 -> C₈H₉NO₂) ---
def to_subscript_formula(formula: str) -> str:
    sub_map = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return formula.translate(sub_map)

# --- 本地缓存机制 ---
def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_cache(cache_data):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)

cache = load_cache()

# --- 本地离线高频中文化学词典（秒级出图，免耗Token） ---
LOCAL_CHEMICAL_DB = {
    # 常用抗生素与药物
    "阿莫西林": {"smiles": "CC1(C(N2C(S1)C(C2=O)NC(=O)C(c3ccc(O)cc3)N)C(=O)O)C", "title": "阿莫西林 (Amoxicillin)", "iupac": "(2S,5R,6R)-6-[[(2R)-2-amino-2-(4-hydroxyphenyl)acetyl]amino]-3,3-dimethyl-7-oxo-4-thia-1-azabicyclo[3.2.0]heptane-2-carboxylic acid"},
    "利多卡因": {"smiles": "CCN(CC)CC(=O)Nc1c(C)cccc1C", "title": "利多卡因 (Lidocaine)", "iupac": "2-(diethylamino)-N-(2,6-dimethylphenyl)acetamide"},
    "奥美拉唑": {"smiles": "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1", "title": "奥美拉唑 (Omeprazole)", "iupac": "6-methoxy-2-[(4-methoxy-3,5-dimethylpyridin-2-yl)methylsulfinyl]-1H-benzimidazole"},
    "甲硝唑": {"smiles": "Cc1ncc([N+](=O)[O-])n1CCO", "title": "甲硝唑 / 灭滴灵 (Metronidazole)", "iupac": "2-(2-methyl-5-nitro-1H-imidazol-1-yl)ethanol"},
    "灭滴灵": {"smiles": "Cc1ncc([N+](=O)[O-])n1CCO", "title": "灭滴灵 / 甲硝唑 (Metronidazole)", "iupac": "2-(2-methyl-5-nitro-1H-imidazol-1-yl)ethanol"},
    "替硝唑": {"smiles": "CCS(=O)(=O)CCn1c(C)ncc1[N+](=O)[O-]", "title": "替硝唑 (Tinidazole)", "iupac": "1-(2-ethylsulfonylethyl)-2-methyl-5-nitro-imidazole"},
    "青蒿素": {"smiles": "CC1CCC2C(C(=O)OC3C24C1CCC(O3)(OO4)C)C", "title": "青蒿素 (Artemisinin)", "iupac": "(1R,4S,5R,8S,9R,12S,13R)-1,5,9-trimethyl-11,14,15,16-tetraoxatetracyclo[10.3.1.04,13.08,13]hexadecan-10-one"},
    "阿司匹林": {"smiles": "CC(=O)Oc1ccccc1C(=O)O", "title": "阿司匹林 / 乙酰水杨酸 (Aspirin)", "iupac": "2-acetyloxybenzoic acid"},
    "乙酰水杨酸": {"smiles": "CC(=O)Oc1ccccc1C(=O)O", "title": "乙酰水杨酸 / 阿司匹林 (Aspirin)", "iupac": "2-acetyloxybenzoic acid"},
    "扑热息痛": {"smiles": "CC(=O)Nc1ccc(O)cc1", "title": "对乙酰氨基酚 / 扑热息痛 (Paracetamol)", "iupac": "N-(4-hydroxyphenyl)acetamide"},
    "对乙酰氨基酚": {"smiles": "CC(=O)Nc1ccc(O)cc1", "title": "对乙酰氨基酚 / 扑热息痛 (Paracetamol)", "iupac": "N-(4-hydroxyphenyl)acetamide"},
    "布洛芬": {"smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "title": "布洛芬 (Ibuprofen)", "iupac": "2-[4-(2-methylpropyl)phenyl]propanoic acid"},
    "咖啡因": {"smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "title": "咖啡因 (Caffeine)", "iupac": "1,3,7-trimethylpurine-2,6-dione"},
    "青霉素": {"smiles": "CC1(C(N2C(S1)C(C2=O)NC(=O)Cc3ccccc3)C(=O)O)C", "title": "青霉素G (Penicillin G)", "iupac": "(2S,5R,6R)-3,3-dimethyl-7-oxo-6-[(2-phenylacetyl)amino]-4-thia-1-azabicyclo[3.2.0]heptane-2-carboxylic acid"},
    "香兰素": {"smiles": "O=Cc1ccc(O)c(OC)c1", "title": "香兰素 (Vanillin)", "iupac": "4-hydroxy-3-methoxybenzaldehyde"},
    "薄荷醇": {"smiles": "CC1CCC(C(C1)O)C(C)C", "title": "薄荷醇 (Menthol)", "iupac": "5-methyl-2-(propan-2-yl)cyclohexan-1-ol"},
    "多巴胺": {"smiles": "NCCc1ccc(O)c(O)c1", "title": "多巴胺 (Dopamine)", "iupac": "4-(2-aminoethyl)benzene-1,2-diol"},
    "褪黑素": {"smiles": "CC(=O)NCCC1=CNc2c1cc(OC)cc2", "title": "褪黑素 (Melatonin)", "iupac": "N-[2-(5-methoxy-1H-indol-3-yl)ethyl]acetamide"},
    "维生素c": {"smiles": "C1(=C(C(=O)OC1C(CO)O)O)O", "title": "维生素C / 抗坏血酸 (Vitamin C)", "iupac": "(5R)-[(1S)-1,2-dihydroxyethyl]-3,4-dihydroxyfuran-2(5H)-one"},
    "抗坏血酸": {"smiles": "C1(=C(C(=O)OC1C(CO)O)O)O", "title": "维生素C / 抗坏血酸 (Vitamin C)", "iupac": "(5R)-[(1S)-1,2-dihydroxyethyl]-3,4-dihydroxyfuran-2(5H)-one"},
    
    # 常用溶剂与基础分子
    "乙醇": {"smiles": "CCO", "title": "乙醇 / 酒精 (Ethanol)", "iupac": "ethanol"},
    "酒精": {"smiles": "CCO", "title": "乙醇 / 酒精 (Ethanol)", "iupac": "ethanol"},
    "甲醇": {"smiles": "CO", "title": "甲醇 (Methanol)", "iupac": "methanol"},
    "丙酮": {"smiles": "CC(=O)C", "title": "丙酮 (Acetone)", "iupac": "propan-2-one"},
    "乙酸": {"smiles": "CC(=O)O", "title": "乙酸 / 醋酸 (Acetic acid)", "iupac": "ethanoic acid"},
    "醋酸": {"smiles": "CC(=O)O", "title": "乙酸 / 醋酸 (Acetic acid)", "iupac": "ethanoic acid"},
    "乙酸乙酯": {"smiles": "CCOC(=O)C", "title": "乙酸乙酯 (Ethyl acetate)", "iupac": "ethyl acetate"},
    "乙醚": {"smiles": "CCOCC", "title": "乙醚 (Diethyl ether)", "iupac": "ethoxyethane"},
    "苯": {"smiles": "c1ccccc1", "title": "苯 (Benzene)", "iupac": "benzene"},
    "甲苯": {"smiles": "Cc1ccccc1", "title": "甲苯 (Toluene)", "iupac": "methylbenzene"},
    "苯酚": {"smiles": "Oc1ccccc1", "title": "苯酚 (Phenol)", "iupac": "phenol"},
    "柠檬酸": {"smiles": "OC(=O)CC(O)(CC(=O)O)C(=O)O", "title": "柠檬酸 (Citric acid)", "iupac": "2-hydroxypropane-1,2,3-tricarboxylic acid"},
    "水杨酸": {"smiles": "O=C(O)c1ccccc1O", "title": "水杨酸 (Salicylic acid)", "iupac": "2-hydroxybenzoic acid"},
    "葡萄糖": {"smiles": "OC[C@@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", "title": "葡萄糖 (Glucose)", "iupac": "D-glucopyranose"}
}

# 结构式反查索引
REVERSE_SMILES_INDEX = {}
for k, v in LOCAL_CHEMICAL_DB.items():
    try:
        can_smi = Chem.CanonSmiles(v["smiles"])
        if can_smi not in REVERSE_SMILES_INDEX:
            REVERSE_SMILES_INDEX[can_smi] = v
    except Exception:
        pass

# --- 基于官方规范的 requests 通信函数（初版稳定内核） ---
def call_deepseek_api(endpoint_url: str, api_key: str, model_name: str, prompt: str, timeout: int = 60):
    clean_key = str(api_key).strip(" []'\"`\n\r\t")
    clean_url = str(endpoint_url).strip(" []'\"`\n\r\t")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {clean_key}"
    }
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }
    try:
        resp = requests.post(clean_url, headers=headers, json=payload, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"], None
        else:
            return None, f"HTTP {resp.status_code}: {resp.text}"
    except Exception as e:
        return None, f"网络请求出错: {str(e)}"

# --- AI 生僻名称识别转换 ---
def resolve_name_with_ai(query: str, endpoint_url: str, key: str, model: str):
    prompt = f"""你是一名化学专家。请将用户输入的化学名称（中文/英文/俗名/商品名/CAS号）解析转换为标准规范 SMILES 字符串。
待解析名称: "{query}"

必须仅输出一个严格合法的 JSON 对象，不加任何 Markdown 代码块标签与多余文字：
{{"smiles": "标准SMILES", "title": "规范名称", "iupac": "IUPAC命名"}}
如果不是确定的化学分子，请输出: {{"error": "not_found"}}"""

    content, err = call_deepseek_api(endpoint_url, key, model, prompt, timeout=25)
    if err:
        return None, err
    try:
        text = content.strip()
        # 清除 markdown 标记
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
        data = json.loads(text)
        if "smiles" in data and data["smiles"]:
            test_mol = Chem.MolFromSmiles(data["smiles"])
            if test_mol:
                return data, None
        return None, "AI 返回的结构未能通过化学有效性校验"
    except Exception as ex:
        return None, f"解析异常: {ex}"

# --- 侧边栏 ---
with st.sidebar:
    st.header("📚 历史查询库")
    history_keys = list(cache.keys())
    selected_history = st.selectbox(
        "快速载入历史已解析分子：",
        options=["-- 请选择 --"] + history_keys,
        help="切换历史分子完全不消耗 Token。"
    )
    
    st.markdown("---")
    st.header("⚙️ AI 专家模型配置")
    
    # 锁定官方标准接口，杜绝填错 URL 造成 404
    provider = st.selectbox(
        "选择服务商",
        options=["DeepSeek 官方 (推荐)", "自定义服务商"]
    )
    
    if provider == "DeepSeek 官方 (推荐)":
        final_api_url = "https://api.deepseek.com/v1/chat/completions"
        model_choice = st.selectbox(
            "选择模型版本",
            options=["deepseek-flash (V4.1 Flash 最新极速)", "deepseek-chat (V3 通用旗舰)"],
            index=0
        )
        final_model_name = "deepseek-flash" if "flash" in model_choice else "deepseek-chat"
    else:
        final_api_url = st.text_input("API URL", value="https://api.deepseek.com/v1/chat/completions")
        final_model_name = st.text_input("模型名称", value="deepseek-flash")

    raw_key = st.text_input("API Key", type="password", placeholder="填入 sk-xxxx 密钥")
    final_api_key = raw_key.strip(" []'\"`\n\r\t")
    
    st.markdown("---")
    if st.button("🗑️ 清空本地历史缓存"):
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
            st.success("缓存已清空，请刷新页面。")

# --- 主界面 ---
st.title("🧪 AI 化学分子多维工作台")
st.caption("支持输入：**中英文俗名/商品名/学名**（如阿莫西林、青蒿素、利多卡因、柠檬酸） / **SMILES 结构式**")

# 输入框初值
default_input = "CC(=O)Nc1ccc(O)cc1"
if selected_history != "-- 请选择 --":
    default_input = selected_history

user_query = st.text_input("🔍 分子搜索或输入：", value=default_input)

# --- 分子解析流程 ---
mol = None
current_smiles = ""
meta_info = {}

if user_query:
    clean_q = user_query.strip()
    
    # 1. 离线高频词典匹配
    if clean_q.lower() in LOCAL_CHEMICAL_DB:
        entry = LOCAL_CHEMICAL_DB[clean_q.lower()]
        mol = Chem.MolFromSmiles(entry["smiles"])
        current_smiles = Chem.MolToSmiles(mol)
        meta_info = {"title": entry["title"], "iupac": entry["iupac"]}
    
    # 2. 直接作为 SMILES 识别
    if mol is None:
        mol_candidate = Chem.MolFromSmiles(clean_q)
        if mol_candidate:
            mol = mol_candidate
            current_smiles = Chem.MolToSmiles(mol)
            if current_smiles in REVERSE_SMILES_INDEX:
                matched = REVERSE_SMILES_INDEX[current_smiles]
                meta_info = {"title": matched["title"], "iupac": matched["iupac"]}
            else:
                meta_info = {"title": "自主输入分子骨架", "iupac": "由专家推导或系统自动分析"}
            
    # 3. 词典未命中时，调用 DeepSeek 4.1 Flash 智能转换
    if mol is None and final_api_key:
        with st.spinner(f"正在让 AI 智能识别“{clean_q}”的化学拓扑结构..."):
            ai_data, err_msg = resolve_name_with_ai(clean_q, final_api_url, final_api_key, final_model_name)
            if ai_data and "smiles" in ai_data:
                mol = Chem.MolFromSmiles(ai_data["smiles"])
                current_smiles = Chem.MolToSmiles(mol)
                meta_info = {
                    "title": ai_data.get("title", clean_q),
                    "iupac": ai_data.get("iupac", "由AI推导")
                }
            elif err_msg:
                st.warning(f"AI 联想识别提示: {err_msg}")

    # 4. 无法解析提示
    if mol is None:
        if not final_api_key:
            st.error(f"本地词典未收录“{clean_q}”。\n\n👉 **提示**：请在左侧侧边栏填入 API Key 开启 **AI 全量分子联想**；或直接输入其 SMILES 结构式。")
        else:
            st.error(f"未能成功解析“{clean_q}”，请检查名称拼写或输入标准 SMILES。")

# --- 正常展示区域 ---
if mol:
    raw_formula = rdMolDescriptors.CalcMolFormula(mol)
    subscript_formula = to_subscript_formula(raw_formula)  # 下标化化学式
    
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    bertz = Descriptors.BertzCT(mol)
    fsp3 = Descriptors.FractionCSP3(mol)
    heavy_atoms = mol.GetNumHeavyAtoms()

    # 读取本地缓存
    cached_entry = cache.get(current_smiles, {})
    ai_report_cached = cached_entry.get("ai_report", "")

    # --- 模块 1：常规基础性质（已去掉蓝色提示框） ---
    with st.expander("📖 1. 基础信息与通用物理化学性质（普适概览）", expanded=True):
        col_meta1, col_meta2 = st.columns([1, 1])
        with col_meta1:
            st.markdown(f"**分子通用名**：`{meta_info.get('title', user_query)}`")
            st.markdown(f"**化学式**：`{subscript_formula}`")
            st.markdown(f"**规范 SMILES**：`{current_smiles}`")
        with col_meta2:
            st.markdown(f"**IUPAC 系统命名**：*{meta_info.get('iupac', '标准有机命名')}*")
            state_text = "晶体/固体" if mw > 110 and tpsa > 30 else "低熔点液体/易挥发相"
            sol_text = "极易溶于水 (亲水)" if logp < 0.2 else ("中等脂水兼溶" if logp < 3.0 else "难溶于水 (强亲脂)")
            st.markdown(f"**常态预期**：{state_text}  |  **溶解性倾向**：{sol_text}")

    # --- 模块 2：结构式与归因热力图 ---
    col1, col2 = st.columns([1, 1])
    with col1:
        with st.expander("📌 2. 2D 平面分子拓扑结构", expanded=True):
            img = Draw.MolToImage(mol, size=(240, 240))
            sub_c1, sub_c2, sub_c3 = st.columns([1, 2, 1])
            with sub_c2:
                st.image(img, caption=f"{subscript_formula} 骨架图", width=200)

    with col2:
        with st.expander("🔬 3. 结构对性质影响归因（LogP 热力图）", expanded=True):
            st.caption("🟢 绿色：增加脂溶性（疏水基团） | 🔴 粉红：增加亲水性（极性基团）")
            try:
                contribs = rdMolDescriptors._CalcCrippenContribs(mol)
                weights = [c[0] for c in contribs]
                drawer = rdMolDraw2D.MolDraw2DSVG(240, 240)
                SimilarityMaps.GetSimilarityMapFromWeights(mol, weights, drawer)
                drawer.FinishDrawing()
                svg_code = drawer.GetDrawingText()
                st.markdown(f'<div style="display:flex; justify-content:center;">{svg_code}</div>', unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"热力图渲染提示: {e}")

    # --- 模块 3：专业化学计算参数 ---
    with st.expander("📊 4. 深度药化与拓扑量化参数", expanded=False):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("相对分子质量", f"{mw:.2f}", help="单位 g/mol")
        m2.metric("脂水分配 LogP", f"{logp:.2f}", help=">0倾向脂溶，<0倾向水溶")
        m3.metric("极性表面积 TPSA", f"{tpsa:.2f} Å²", help="与透膜性及细胞吸收相关")
        m4.metric("饱和碳比例 Fsp3", f"{fsp3:.2f}", help="sp3碳占比，反映三维空间柔性")
        
        m5, m6, m7, m8 = st.columns(4)
        m5.metric("氢键供体数 (HBD)", hbd)
        m6.metric("氢键受体数 (HBA)", hba)
        m7.metric("Bertz 拓扑复杂度", f"{bertz:.1f}", help="骨架及杂原子连接复杂度")
        m8.metric("非氢重原子数", heavy_atoms)

    # --- 模块 4：AI 专家研报推演 ---
    st.markdown("---")
    with st.expander("🤖 5. 专家机理推演与逆合成路径", expanded=True):
        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            run_ai = st.button("🚀 生成/重算研报", type="primary")
        with col_btn2:
            if ai_report_cached:
                st.success("✅ 该分子已在本地缓存，直接显示历史研报（未消耗 Token）。若需重新生成请点击左侧按钮。")

        if run_ai:
            if not final_api_key:
                st.warning("请在左侧侧边栏填入 API Key 才能调用 AI 模型生成。")
            else:
                with st.spinner("DeepSeek 正在解析反应切断位点与机理..."):
                    prompt = f"""
你是一名资深的有机化学与药物化学家。请基于以下经由化学信息学严密计算的数据，对目标分子做出一份详尽的专家推演研报：

【分子特征】
- 通用名称: {meta_info.get('title', user_query)}
- 化学式: {subscript_formula}
- SMILES: {current_smiles}
- 相对分子质量: {mw:.2f} g/mol
- 脂水分配系数 (LogP): {logp:.2f}
- 极性表面积 (TPSA): {tpsa:.2f} Å²
- 氢键供体/受体: {hbd} / {hba}
- 拓扑复杂度 (Bertz): {bertz:.1f}
- 饱和碳比例 (Fsp3): {fsp3:.2f}

请严格按以下维度展开推演（专业严谨，切中反应机理与具体官能团）：
### 一、 构效关系深度解析 (SAR & Electronic Effects)
1. 解释分子关键官能团对水溶性/脂溶性（LogP）的微观作用机理（结合电子效应、共轭或氢键网络）。
2. 分析空间骨架柔性对实际应用（或生物活性靶点匹配）的影响。

### 二、 逆合成路线推演 (Retrosynthesis)
1. 明确指出最适宜的【关键切断键 (Disconnection)】。
2. 给出具体的合成路线设计：列出商业可得的起始原料、经典人名反应（如酰化、重氮化、取代反应等）及推荐试剂/反应条件。

### 三、 工艺安全与反应难点控制
1. 指出合成过程中的副反应、区域/化学选择性控制难点。
2. 实验安全注意点（如放热控制、有毒中间体、产物纯化重结晶建议）。
"""
                    content_out, req_err = call_deepseek_api(final_api_url, final_api_key, final_model_name, prompt)
                    if not req_err and content_out:
                        ai_report_cached = content_out
                        cache[current_smiles] = {
                            "query_name": meta_info.get('title', user_query),
                            "formula": subscript_formula,
                            "mw": mw,
                            "logp": logp,
                            "ai_report": ai_report_cached,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        save_cache(cache)
                        st.rerun()
                    else:
                        st.error(f"API 请求失败: {req_err}")

        # 展示研报内容与导出功能
        if ai_report_cached:
            st.markdown(ai_report_cached)
            st.markdown("---")
            markdown_report = f"""# 化学分子多维分析报告：{meta_info.get('title', user_query)}
- **生成时间**：{cache.get(current_smiles, {}).get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}
- **化学式**：{subscript_formula}
- **SMILES**：`{current_smiles}`

## 一、 理化与拓扑数据对照表
| 参数指标 | 数值 | 物理/化学含义 |
| :--- | :--- | :--- |
| 分子量 (MolWt) | {mw:.2f} g/mol | 相对分子质量 |
| 脂水分配系数 (LogP) | {logp:.2f} | 衡量亲脂/亲水倾向 |
| 极性表面积 (TPSA) | {tpsa:.2f} Å² | 极性原子表面积和 |
| 氢键供体 / 受体 | {hbd} / {hba} | 分子间/内氢键作用能力 |
| Bertz 拓扑复杂度 | {bertz:.1f} | 骨架及杂原子连接复杂度 |
| 饱和碳比例 (Fsp3) | {fsp3:.2f} | 三维空间度与立体柔性 |

## 二、 专家机理推演与合成路线研报
{ai_report_cached}

---
*本研报由 AI 化学分子多维工作台自动计算与推演生成。*
"""
            st.download_button(
                label="📥 导出完整研报为 Markdown 文档 (.md)",
                data=markdown_report,
                file_name=f"{raw_formula}_{meta_info.get('title', 'molecule')}_Report.md",
                mime="text/markdown"
            )