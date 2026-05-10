import json
import streamlit as st
from openai import OpenAI
import pandas as pd
import os
import re

# ----------------- 1. 初始化设置 -----------------
API_KEY = st.secrets["DEEPSEEK_API_KEY"]
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com") 

st.set_page_config(page_title="智能机翼检索系统", page_icon="✈️", layout="wide")

if "search_done" not in st.session_state:
    st.session_state.search_done = False
    st.session_state.matched_airfoils =[]
    st.session_state.params = {}

# ----------------- 2. 加载与清洗真实数据库 -----------------
@st.cache_data
def load_database():
    excel_path = "airfoiltools_geo_clcd.xlsx" 
    df = pd.read_excel(excel_path)
    df.columns = df.columns.str.strip() 
    return df

def parse_ld_data(val):
    if pd.isna(val):
        return -1.0, "未知"
    val_str = str(val)
    match = re.search(r"([\d\.]+)\s*at\s*.*?=\s*([-\d\.]+)", val_str)
    if match:
        return float(match.group(1)), match.group(2)
    match_single = re.search(r"([\d\.]+)", val_str)
    if match_single:
        return float(match_single.group(1)), "未知"
    return -1.0, "未知"

def safe_float(val, default_val):
    if val in [None, "null", "None", "", "未知"]:
        return default_val
    try:
        return float(val)
    except ValueError:
        return default_val

# ----------------- 3. 调用大模型提取参数 -----------------
def extract_params(user_input):
    system_prompt = """
    你是一个资深的空气动力学专家和飞行器设计工程师。
    你的任务是从用户的模糊自然语言中，提取出机翼选型所需的参数。
    
    1. reynolds_number (雷诺数): 根据场景推算(如室内5万，航模10-20万，大型50万)。只输出纯数字，不要带单位。
    2. lift_to_drag_ratio (目标升阻比): 根据机型推算(如航模10-15，客机25+)。只输出纯数字。
    3. airfoil_family (期望翼型系列): 如果用户明确提到了某个特定的机翼系列（如 "NACA", "Clark", "Eppler", "Boeing" 等），请提取出该英文名称（忽略大小写）。如果没有指定，必须返回 "null"。

    请严格以 JSON 格式输出，包含以下四个字段：
    {
        "reasoning": "你的推理过程",
        "reynolds_number": "...",
        "lift_to_drag_ratio": "...",
        "airfoil_family": "..."
    }
    """
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            response_format={ "type": "json_object" }, 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.2 
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return None

# ----------------- 4. 核心逻辑：在 Excel 中精准比对 -----------------
def search_airfoils(df, params):
    target_re = safe_float(params.get("reynolds_number"), 100000)
    target_ld = safe_float(params.get("lift_to_drag_ratio"), None)
    
    # 获取用户指定的翼型家族（例如 "naca"），并全部转为小写方便比对
    target_family = params.get("airfoil_family")
    target_family = None if target_family in [None, "null", "", "未知"] else str(target_family).lower()
    
    re_columns = {50000: '最大升阻比_Re50000', 100000: '最大升阻比_Re100000', 200000: '最大升阻比_Re200000', 500000: '最大升阻比_Re500000'}
    closest_re = min(re_columns.keys(), key=lambda k: abs(k - target_re))
    target_col = re_columns[closest_re]
    
    results =[]
    for index, row in df.iterrows():
        airfoil_name = str(row.get('UIUC翼型名', '未知翼型'))
        max_ld, alpha = parse_ld_data(row[target_col])
        camber_info = row.get('最大弯度信息', '暂无')
        
        if max_ld <= 0:
            continue
        if "camber 0%" in str(camber_info).lower():
            continue
        
        if target_family and target_family not in airfoil_name.lower():
            continue

        if target_ld is None:
            results.append({
                "name": airfoil_name, "thickness": row.get('最大厚度信息', '暂无'),
                "camber": camber_info, "closest_re": closest_re,
                "max_ld_num": max_ld, "alpha": alpha
            })
        else:
            required_2d_ld = target_ld * 1.5 
            if max_ld >= required_2d_ld:
                results.append({
                    "name": airfoil_name, "thickness": row.get('最大厚度信息', '暂无'),
                    "camber": camber_info, "closest_re": closest_re,
                    "max_ld_num": max_ld, "alpha": alpha, "diff": abs(max_ld - required_2d_ld) 
                })
                
    if target_ld is None:
        results = sorted(results, key=lambda x: x["max_ld_num"], reverse=True)[:5]
    else:
        results = sorted(results, key=lambda x: x["diff"])[:5]
        
    return results

# ----------------- 5. 网页前端设计 -----------------
st.title("✈️ 智能机翼检索系统 (基于云端知识库)")

try:
    df = load_database()
    st.caption(f"✅ 成功连接云端数据库，已加载 {len(df)} 款机翼数据。")
except Exception as e:
    st.error(f"❌ 读取数据库失败，报错信息：{e}")
    st.stop()

user_input = st.text_area("请描述您的需求：", placeholder="例如：设计一款飞机，大概需要20的升阻比，希望能找个NACA系列的机翼。")

# 当点击按钮时，执行检索并存入 Session State
if st.button("🔍 开始智能检索", type="primary"):
    if not user_input.strip():
        st.warning("请输入需求！")
    else:
        with st.spinner("🧠 AI 正在深度分析您的需求..."):
            params = extract_params(user_input)
            
        if params:
            # 存入缓存
            st.session_state.params = params
            with st.spinner("🗄️ 正在本地知识库中为您精准匹配最合适的机翼..."):
                st.session_state.matched_airfoils = search_airfoils(df, params)
            st.session_state.search_done = True
        else:
            st.error("❌ AI 参数解析失败，请检查网络或 API Key 余额。")

# ================= 如果搜索完成，从 Session State 读取并展示结果 =================
if st.session_state.search_done:
    params = st.session_state.params
    matched_airfoils = st.session_state.matched_airfoils
    
    st.info(f"💡 **AI 专家思考过程**：\n{params.get('reasoning', '无')}")
    
    col1, col2, col3 = st.columns(3)
    disp_re = params.get("reynolds_number")
    disp_ld = params.get("lift_to_drag_ratio")
    disp_fam = params.get("airfoil_family")
    
    col1.metric("📌 目标雷诺数", disp_re if disp_re not in ["null", None, ""] else "100000 (默认)")
    col2.metric("📌 目标升阻比", disp_ld if disp_ld not in ["null", None, ""] else "未提供")
    col3.metric("📌 指定系列", disp_fam if disp_fam not in ["null", None, ""] else "不限")
    
    st.divider()
    
    if len(matched_airfoils) > 0:
        st.subheader(f"🎯 检索完毕，为您推荐以下 {len(matched_airfoils)} 款机翼：")
        
        for item in matched_airfoils:
            with st.container():
                st.markdown(f"### 🏆 翼型：`{item['name']}`")
                text_col, img_col = st.columns([1, 1])
                
                with text_col:
                    st.markdown(f"- **最大厚度**: {item['thickness']}")
                    st.markdown(f"- **最大弯度**: {item['camber']}")
                    st.success(
                        f"**在 Re={item['closest_re']} 工况下**：\n\n"
                        f"🌟 二维翼型最大升阻比为 **{item['max_ld_num']}**\n\n"
                        f"📐 对应最佳迎角为 **α={item['alpha']}°**"
                    )
                    
                    dat_path = os.path.join("uiuc_airfoil_dat", f"{item['name']}.dat")
                    if os.path.exists(dat_path):
                        with open(dat_path, "rb") as file:
                            st.download_button(
                                label=f"📥 一键下载 {item['name']} 坐标 (.dat)",
                                data=file,
                                file_name=f"{item['name']}.dat",
                                mime="text/plain",
                                key=f"dl_btn_{item['name']}"
                            )

                with img_col:
                    img_path = os.path.join("uiuc_airfoil_images", f"{item['name']}.gif")
                    if os.path.exists(img_path):
                        st.image(img_path, caption=f"{item['name']} 翼型轮廓", use_container_width=True)
                    else:
                        st.warning("🚫 暂无该机翼的预览动图")
                        
                st.markdown("---") 
    else:
        st.error("抱歉，当前本地数据库中没有符合您全部要求（包含指定翼型系列）的机翼。")
