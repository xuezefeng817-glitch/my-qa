import streamlit as st
import json
import os
from zhipuai import ZhipuAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

st.set_page_config(page_title="知识问答助手", page_icon="https://i.ibb.co/Xfc0gm63/sunline.png", layout="centered")
st.markdown("""
<style>
    /* 清爽商务风 */
    .stApp {
        background-color: #FFFFFF;
    }
    .st-bb {
        background-color: #FFFFFF;
        border-bottom: 1px solid #E0E0E0;
    }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        border: 1px solid #D0D7DE;
        border-radius: 8px;
    }
    .stButton>button {
        background-color: #4A90E2;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 8px 16px;
    }
    .stButton>button:hover {
        background-color: #357ABD;
    }
    .sidebar .sidebar-content {
        background-color: #FFFFFF;
        border-right: 1px solid #E0E0E0;
    }
    h1, h2, h3 {
        color: #2C3E50;
    }
    .stChatMessage {
        border-radius: 12px;
        padding: 12px;
    }
    .stExpander {
        border: 1px solid #E0E0E0;
        border-radius: 8px;
    }
    .stRadio > div {
        gap: 15px;
    }    
</style>
""", unsafe_allow_html=True)
ADMIN_PASSWORD = st.secrets["admin_password"]
DATA_FILE = "qa_database.json"

def load_qa_pairs():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_qa_pairs(pairs):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)

def search_relevant_pairs(query, pairs, top_k=3):
    """
    语义检索：使用智谱 Embedding-3 模型，理解问题“意思”
    """
    if not pairs:
        return []
    
    client = ZhipuAI(api_key=st.secrets["zhipuai_api_key"])
    
    # 1. 获取用户问题的向量
    query_response = client.embeddings.create(
        model="embedding-3",
        input=query
    )
    query_vec = np.array(query_response.data[0].embedding)
    
    # 2. 获取知识库所有问题的向量
    questions = [p["q"] for p in pairs]
    docs_response = client.embeddings.create(
        model="embedding-3",
        input=questions
    )
    
    # 3. 计算语义相似度
    sims = []
    for doc_data in docs_response.data:
        doc_vec = np.array(doc_data.embedding)
        # 余弦相似度
        sim = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
        sims.append(sim)
    
    sims = np.array(sims)
    # 按相似度从高到低排序，返回 top_k 条，且相似度 > 0.25
    top_indices = np.argsort(sims)[-top_k:][::-1]
    return [pairs[i] for i in top_indices if sims[i] > 0.25]

def generate_answer(user_query, relevant_pairs, chat_history=None):
    # ========== 计算器 Skill ==========
    import re, math
    calc_pattern = r'^[\d\s\+\-\*\/\(\)\.\,\^\=]+$'
    clean_query = user_query.strip().replace('?', '').replace('？', '').replace('=', '')
    if re.match(calc_pattern, clean_query):
        try:
            expr = clean_query
            safe_dict = {
                'abs': abs, 'round': round, 'min': min, 'max': max,
                'pow': pow, 'sqrt': math.sqrt, 'sin': math.sin, 'cos': math.cos,
                'tan': math.tan, 'pi': math.pi, 'e': math.e,
                'log': math.log, 'log10': math.log10
            }
            result = eval(expr, {"__builtins__": {}}, safe_dict)
            return f"计算结果：{result}"
        except:
            pass
    # ========== 计算器结束 ==========

    if not relevant_pairs:
        return "抱歉，我目前的知识库中还没有收录关于这个问题的答案。建议您联系管理员补充相关知识。"

    context_lines = []
    for i, pair in enumerate(relevant_pairs, 1):
        context_lines.append(f"知识{i}：问：{pair['q']} 答：{pair['a']}")
    context = "\n".join(context_lines)

    history_text = ""
    if chat_history:
        recent = chat_history[-4:]
        for h in recent:
            role = "用户" if h["role"] == "user" else "助手"
            history_text += f"{role}：{h['content']}\n"

    # 规则描述（system 消息）
    system_prompt = (
        "你是一个专业的知识问答助手，请基于用户提供的【背景知识】回答问题。\n"
        "要求：\n"
        "1. 先直接给出核心答案。\n"
        "2. 然后使用 Markdown 的列表格式（以 '-' 开头），逐条列出所有相关的细节、要求、说明或补充信息，"
        "每一项单独成行，项与项之间空一行。\n"
        "3. 如果有例外或特殊说明，请在最后以“补充说明：”开头单独说明，也使用列表形式。\n"
        "4. 回答要清晰、有条理，但必须严格基于背景知识，不添加外部信息。\n"
        "5. 若背景知识不足以回答，直接回复“抱歉，根据当前知识库，暂时无法回答这个问题。”"
    )

    # 用户消息：包含上下文、知识和痛点
    user_prompt = (
        f"【最近对话】\n{history_text}\n"
        f"【背景知识】\n{context}\n"
        f"【当前问题】\n{user_query}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    client = ZhipuAI(api_key=st.secrets["zhipuai_api_key"])
    response = client.chat.completions.create(
        model="glm-4-flash",
        messages=messages,
        temperature=0.1,
        max_tokens=300,
    )
    return response.choices[0].message.content

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_page" not in st.session_state:
    st.session_state.current_page = "问答"
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# 顶部导航栏
# 侧边栏导航
st.sidebar.title("📚 菜单")
page = st.sidebar.radio("选择页面", ["问答", "管理"])

if page != st.session_state.current_page:
    st.session_state.current_page = page
    st.rerun()

if st.session_state.current_page == "问答":
    st.markdown("<h1><img src='https://i.ibb.co/Xfc0gm63/sunline.png' width='40' style='vertical-align: middle; margin-right: 10px;'> Sunline Knowledge Q&A</h1>", unsafe_allow_html=True)
    st.caption("支持财务ERP报销相关问题的提问，AI 会根据知识库进行回答")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("请输入你的问题..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        all_pairs = load_qa_pairs()
        relevant = search_relevant_pairs(prompt, all_pairs, top_k=3)
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                try:
                    answer = generate_answer(prompt, relevant, st.session_state.chat_history)
                except Exception as e:
                    answer = f"⚠️ 回答生成失败：{str(e)}"
                st.markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

elif st.session_state.current_page == "管理":
    st.title("🔐 知识库管理")

    if "admin_authenticated" not in st.session_state:
        st.session_state.admin_authenticated = False

    if not st.session_state.admin_authenticated:
        pwd = st.text_input("请输入管理密码", type="password")
        if st.button("验证"):
            if pwd == ADMIN_PASSWORD:
                st.session_state.admin_authenticated = True
                st.rerun()
            else:
                st.error("密码错误")
    else:
        st.success("已认证为管理员")
        if st.button("退出管理"):
            st.session_state.admin_authenticated = False
            st.rerun()

        st.markdown("---")
        qa_pairs = load_qa_pairs()
        st.write(f"当前共 **{len(qa_pairs)}** 组问答")

        with st.expander("➕ 手动添加一条问答"):
            with st.form("add_single"):
                new_q = st.text_input("问题")
                new_a = st.text_area("答案")
                if st.form_submit_button("添加"):
                    if new_q.strip() and new_a.strip():
                        qa_pairs.append({"q": new_q.strip(), "a": new_a.strip()})
                        save_qa_pairs(qa_pairs)
                        st.success("添加成功")
                        st.rerun()
                    else:
                        st.warning("问题和答案不能为空")

        with st.expander("📁 批量上传文件"):
            st.markdown("文件格式：每两行为一组 Q: 问题和 A: 答案，空行和#注释行忽略")
            uploaded_file = st.file_uploader("选择 .txt 文件", type="txt", key=f"batch_uploader_{st.session_state.uploader_key}")
            if uploaded_file is not None:
                # 防止重复处理同一文件
                if "last_uploaded_file" in st.session_state and st.session_state.last_uploaded_file == uploaded_file.name:
                    st.info("文件已处理，如需重新上传请刷新页面或切换菜单")
                else:
                    content = uploaded_file.read().decode("utf-8")
                    lines = content.splitlines()
                    buffer_q, buffer_a = None, None
                    added = 0
                    for line in lines:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if line.startswith("Q:") or line.startswith("q:"):
                            buffer_q = line[2:].strip()
                        elif line.startswith("A:") or line.startswith("a:"):
                            buffer_a = line[2:].strip()
                            if buffer_q and buffer_a:
                                qa_pairs.append({"q": buffer_q, "a": buffer_a})
                                added += 1
                                buffer_q, buffer_a = None, None
                    if added:
                        save_qa_pairs(qa_pairs)
                        st.success(f"成功导入 {added} 组问答")
                        st.session_state.last_uploaded_file = uploaded_file.name
                        st.rerun()
                    else:
                        st.warning("未识别到有效问答对，请检查格式")

        with st.expander("📋 查看/删除现有条目"):
            if not qa_pairs:
                st.info("暂无内容")
            else:
                # 一键清空按钮
                if st.button("⚠️ 清空全部知识库", type="secondary"):
                    save_qa_pairs([])
                    # 重置上传状态，允许重新上传同名文件
                    st.session_state.pop("last_uploaded_file", None)
                    st.session_state.uploader_key += 1
                    st.success("已清空全部知识库，上传组件已重置")
                    st.rerun()

                st.markdown("---")
                for idx, pair in enumerate(qa_pairs):
                    col1, col2 = st.columns([0.8, 0.2])
                    with col1:
                        st.write(f"**Q{idx+1}:** {pair['q'][:50]}...")
                    with col2:
                        if st.button("🗑️", key=f"del_{idx}"):
                            qa_pairs.pop(idx)
                            save_qa_pairs(qa_pairs)
                            st.rerun()
                    with st.expander("展开查看详情 / 编辑"):
                        # 编辑表单
                        with st.form(key=f"edit_form_{idx}"):
                            new_q = st.text_input("问题", value=pair['q'], key=f"edit_q_{idx}")
                            new_a = st.text_area("答案", value=pair['a'], key=f"edit_a_{idx}")
                            if st.form_submit_button("💾 保存修改"):
                                if new_q.strip() and new_a.strip():
                                    qa_pairs[idx] = {"q": new_q.strip(), "a": new_a.strip()}
                                    save_qa_pairs(qa_pairs)
                                    st.success("修改已保存")
                                    st.rerun()
                                else:
                                    st.warning("问题和答案不能为空")
