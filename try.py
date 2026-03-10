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
# ----------------- 2. 加载与清洗真实数据库 -----------------
@st.cache_data
def load_database():
    excel_path = "airfoiltools_geo_clcd.xlsx" 
    
    df = pd.read_excel(excel_path)
    df.columns = df.columns.str.strip() 
    return df

# 【全新升级】解析升阻比和迎角
def parse_ld_data(val):
    """
    从 "40.4 at α=4°" 这样的字符串中，提取出 最大升阻比(40.4) 和 对应迎角(4)
    """
    if pd.isna(val):
        return -1.0, "未知"
        
    val_str = str(val)
    match = re.search(r"([\d\.]+)\s*at\s*.*?=\s*([-\d\.]+)", val_str)
    
    if match:
        max_ld = float(match.group(1))
        alpha = match.group(2) 
        return max_ld, alpha
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
    你的任务是从用户的模糊自然语言中，提取或【推理估算】出机翼选型所需的参数。
    
    【核心原则】：不要轻易返回 null！如果用户没有明确给出数值，你必须根据你的航空工程经验，结合用户的使用场景强行给出一个最合理的估算值！
    
    1. reynolds_number (雷诺数): 
       - 如果用户没给，请根据场景推算：
         * 室内微型飞行器/极低速 -> 50000
         * 常规航模/低速无人机/小飞机 (10-20m/s) -> 100000 或 200000
         * 稍大一点的无人机/高速航模 -> 500000
       - 必须只输出纯数字（不要带单位）。

    2. lift_to_drag_ratio (目标升阻比): 
       - 如果用户没给，请根据机型推算：
         * 简单的低速飞行器/航模 -> 10 到 15
         * 追求航程的巡航无人机 -> 18 到 20
         * 高性能滑翔机/客机 -> 25 到 30+
       - 必须只输出纯数字（例如 15 或 20）。

    请严格以 JSON 格式输出，包含以下三个字段：
    {
        "reasoning": "简要输出你的思考和推算过程（例如：用户提到低速飞行器和500m高度，通常这类无人机速度在15m/s左右，弦长约0.2m，推算雷诺数在10万到20万量级。常规低速飞行器升阻比需求大约在15左右。）",
        "reynolds_number": "...",
        "lift_to_drag_ratio": "..."
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
    
    re_columns = {
        50000: '最大升阻比_Re50000',
        100000: '最大升阻比_Re100000',
        200000: '最大升阻比_Re200000',
        500000: '最大升阻比_Re500000'
    }
    
    closest_re = min(re_columns.keys(), key=lambda k: abs(k - target_re))
    target_col = re_columns[closest_re]
    
    results =[]
    for index, row in df.iterrows():
        max_ld, alpha = parse_ld_data(row[target_col])
        camber_info = row.get('最大弯度信息', '暂无')
        if max_ld <= 0:
            continue
        if "camber 0%" in str(camber_info).lower():
            continue
        if target_ld is None:
            results.append({
                "name": row.get('UIUC翼型名', '未知翼型'),
                "thickness": row.get('最大厚度信息', '暂无'),
                "camber": camber_info,
                "closest_re": closest_re,
                "max_ld_num": max_ld,
                "alpha": alpha
            })
        else:
            required_2d_ld = target_ld * 1.5 
            if max_ld >= required_2d_ld:
                results.append({
                    "name": row.get('UIUC翼型名', '未知翼型'),
                    "thickness": row.get('最大厚度信息', '暂无'),
                    "camber": camber_info,
                    "closest_re": closest_re,
                    "max_ld_num": max_ld,
                    "alpha": alpha,
                    "diff": abs(max_ld - required_2d_ld) 
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

user_input = st.text_area("请描述您的需求：", placeholder="例如：设计一款飞机，大概需要20的升阻比，小飞机")

if st.button("🔍 开始智能检索", type="primary"):
    if not user_input.strip():
        st.warning("请输入需求！")
    else:
        with st.spinner("🧠 AI 正在深度分析您的需求..."):
            params = extract_params(user_input)
            
        if params:
            # ================= 1. 展示 AI 的思考链 (CoT) =================
            if "reasoning" in params:
                st.info(f"💡 **AI 专家思考过程**：\n{params['reasoning']}")
                
            # ================= 2. 展示提取与推算的核心参数 =================
            col1, col2 = st.columns(2)
            disp_re = params.get("reynolds_number")
            disp_ld = params.get("lift_to_drag_ratio")
            
            # 处理 null 或空值的情况，给予友好的默认提示
            col1.metric("📌 AI 提取/推算的雷诺数", disp_re if disp_re not in ["null", None, ""] else "未推算 (默认100000)")
            col2.metric("📌 AI 提取/推算的升阻比", disp_ld if disp_ld not in ["null", None, ""] else "未提供 (按最大升阻比推荐)")
            
            # ================= 3. 调用核心检索逻辑 =================
            with st.spinner("🗄️ 正在本地知识库中为您精准匹配最合适的机翼..."):
                matched_airfoils = search_airfoils(df, params)
            
            st.divider()
            
# ================= 4. 展示匹配结果 =================
            if len(matched_airfoils) > 0:
                st.subheader(f"🎯 检索完毕，为您推荐以下 {len(matched_airfoils)} 款机翼：")
                if disp_ld not in ["null", None, ""]:
                    required_2d = float(disp_ld) * 1.5
                    st.info(
                        f"📐 **系统工程提示**：您期望的飞行器【整机升阻比】约为 **{disp_ld}**。 "
                        f"考虑到真实三维机身带来的寄生阻力与诱导阻力，系统已启动智能补偿机制，"
                        f"自动将匹配下限提升至 1.5 倍，为您寻优【二维翼型理论升阻比】在 **{required_2d}** 以上、且具备有效弯度的实用型机翼。"
                    )
                
                for item in matched_airfoils:
                    with st.container():
                        st.markdown(f"### 🏆 翼型：`{item['name']}`")
                        text_col, img_col = st.columns([1, 1])
                        
                        with text_col:
                            st.markdown(f"- **最大厚度**: {item['thickness']}")
                            st.markdown(f"- **最大弯度**: {item['camber']}")
                            st.success(
                                f"**在 Re={item['closest_re']} 工况下**：\n\n"
                                f"🌟 二维翼型最大升阻比为 **{item['max_ld_num']}** (已满足整机设计余量)\n\n"
                                f"📐 对应最佳迎角为 **α={item['alpha']}°**"
                            )
                            dat_path = os.path.join("uiuc_airfoil_dat", f"{item['name']}.dat")
                            if os.path.exists(dat_path):
                                st.caption(f"📎 存在点位文件: `{dat_path}`")

                        with img_col:
                            img_path = os.path.join("uiuc_airfoil_images", f"{item['name']}.gif")
                            if os.path.exists(img_path):
                                st.image(img_path, caption=f"{item['name']} 翼型轮廓", use_container_width=True)
                            else:
                                st.warning("🚫 暂无该机翼的预览动图")
                                
                        st.markdown("---") 
            else:
                st.error("抱歉，当前本地数据库中没有能达到您升阻比要求的机翼。建议适当放宽升阻比要求。")
        else:
            st.error("❌ AI 参数解析失败，请检查网络或 API Key 余额。")
