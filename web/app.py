import streamlit as st
import requests
import time
import json
import os
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path



# 设置页面宽屏模式，看起来更像专业大模型
st.set_page_config(page_title="EasyTune Pro", layout="wide")


API_URL = "http://localhost:8000"
PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR = PROJECT_ROOT / "logs"

st.title("🚀 EasyTune - 大模型微调云平台")

# 初始化 session state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "task_id" not in st.session_state:
    st.session_state["task_id"] = ""


# ==================== 辅助函数 ====================
def load_training_loss(task_id):
    """从trainer_state.json读取训练loss数据"""
    trainer_state_path = os.path.join(LOG_DIR, task_id, "trainer_state.json")

    if not os.path.exists(trainer_state_path):
        return None

    try:
        with open(trainer_state_path, "r") as f:
            trainer_state = json.load(f)

        data_history = trainer_state.get("log_history", [])
        return data_history

    except Exception as e:
        st.warning(f"读取训练日志失败: {e}")
        return None

@st.fragment(run_every="2s")
def plot_loss_curve(task_id):
    # loss_data 是存字典的列表
    """绘制loss曲线和LR曲线（分图显示）"""
    loss_data = load_training_loss(task_id)
    if loss_data is None:
        st.warning("暂无训练日志数据，可能训练尚未开始或日志文件未生成。请稍后刷新重试。")
        return

    # 提取训练loss和验证loss
    steps = []
    train_loss = []
    learning_rates = []

    for log in loss_data:
        if "loss" in log:
            steps.append(log.get("step", len(steps)))
            train_loss.append(log["loss"])
            learning_rates.append(
                float(log.get("learning_rate", 0)) if log.get("learning_rate") is not None else 0.0
            )

    # --- 图表 1: Loss 曲线 ---
    fig_loss = go.Figure()
    if train_loss:
        fig_loss.add_trace(
            go.Scatter(
                x=steps,
                y=train_loss,
                mode="lines+markers",
                name="Training Loss",
                line=dict(color="#1f77b4", width=2),
                marker=dict(size=4),
            )
        )

    fig_loss.update_layout(
        title="训练 Loss 曲线",
        xaxis_title="步数 (Steps)",
        yaxis_title="Loss 值",
        hovermode="x unified",
        template="plotly_white",
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_loss, use_container_width=True)

    # --- 图表 2: Learning Rate 曲线 ---
    if any(lr is not None for lr in learning_rates):
        fig_lr = go.Figure()
        fig_lr.add_trace(
            go.Scatter(
                x=steps,
                y=learning_rates,
                mode="lines",
                name="Learning Rate",
                line=dict(color="#ff7f0e", width=2, dash="dash"),
            )
        )

        fig_lr.update_layout(
            title="学习率 (Learning Rate) 变化",
            xaxis_title="步数 (Steps)",
            yaxis_title="Learning Rate",
            hovermode="x unified",
            template="plotly_white",
            height=350,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_lr, use_container_width=True)


@st.fragment(run_every="2s")
def status_indicator():
    res = requests.get(f"{API_URL}/status/{st.session_state['task_id']}")
    status = res.json().get("status", "unknown")
    if status == "running":
        st.info("⏳ 正在全力训练中... (下方图表将自动刷新)")
    elif status == "success":
        st.success("🎉 训练完成！请切换到【模型对话对比】标签页进行测试。")
    elif status == "failed":
        st.error("❌ 训练失败，请检查后端日志。")

# 使用 Tabs 分隔功能
tab1, tab2 = st.tabs(["🏗️ 训练控制台", "💬 模型对话对比"])

# ==================== Tab 1: 训练控制台 ====================
with tab1:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("1. 上传数据")
        uploaded_file = st.file_uploader("上传JSON数据集", type="json")
        if uploaded_file is not None:
            if st.button("📤 上传并预处理", key="upload_btn"):
                uploaded_file.seek(0)
                files = {"file": uploaded_file}
                res = requests.post(f"{API_URL}/upload", files=files)
                if res.status_code == 200:
                    st.session_state["file_id"] = res.json()["file_id"]
                    st.success("✅ 上传成功！")

    with col2:
        st.subheader("2. 微调配置")

        # 模型选择
        st.markdown("**模型选择**")
        model_col1, model_col2 = st.columns(2)
        with model_col1:
            model_name = st.selectbox(
                "选择基座模型",
                options=[
                    "Qwen/Qwen2.5-0.5B-Instruct",
                    "Qwen/Qwen2.5-1.5B-Instruct",
                    "Qwen/Qwen2.5-3B-Instruct",
                ],
                index=0,
            )

        # LoRA参数配置
        st.markdown("**LoRA 参数**")
        lora_col1, lora_col2 = st.columns(2)
        with lora_col1:
            lora_r = st.number_input(
                "LoRA Rank (r)", min_value=4, max_value=128, value=8, step=1
            )
        with lora_col2:
            lora_alpha = st.number_input(
                "LoRA Alpha", min_value=4, max_value=128, value=16, step=1
            )

        # 训练超参数
        st.markdown("**训练超参数**")
        param_col1, param_col2, param_col3 = st.columns(3)
        with param_col1:
            epoch = st.number_input(
                "训练轮数 (Epochs)", min_value=1, max_value=50, value=1, step=1
            )
        with param_col2:
            learning_rate = st.number_input(
                "学习率 (Learning Rate)",
                value=1e-5,
                format="%.2e",
                min_value=1e-6,
                max_value=1e-2,
            )
        with param_col3:
            batch_size = st.number_input(
                "批次大小 (Batch Size)", min_value=1, max_value=64, value=2, step=1
            )

        # 梯度累计
        st.markdown("**高级参数**")
        gradient_accumulation_steps = st.number_input(
            "梯度累计步数 (Gradient Accumulation Steps)",
            min_value=1,
            max_value=32,
            value=1,
            step=1,
        )

        if st.button("🚀 开始微调任务", type="primary"):
            if "file_id" not in st.session_state:
                st.error("请先上传数据！")
            else:
                payload = {
                    "file_id": st.session_state["file_id"],
                    "args": {
                        "model_name": model_name,
                        "lora_r": int(lora_r),
                        "lora_alpha": int(lora_alpha),
                        "epoch": int(epoch),
                        "learning_rate": float(learning_rate),
                        "batch_size": int(batch_size),
                        "gradient_accumulation_steps": int(gradient_accumulation_steps),
                    },
                }

                try:
                    res = requests.post(f"{API_URL}/train", json=payload)
                    res.raise_for_status()  # 检查HTTP状态码
                    response_data = res.json()
                    st.session_state["task_id"] = response_data.get("task_id")
                    st.success(f"✅ 任务已提交，ID: {st.session_state['task_id']}")

                except requests.exceptions.RequestException as e:
                    st.error(f"❌ 请求失败: {str(e)}")
                except ValueError as e:
                    st.error(f"❌ 响应格式错误: {str(e)}")

        # # 状态监控
        # if st.session_state["task_id"]:
        #     st.divider()
        #     st.write(f"当前任务 ID: `{st.session_state['task_id']}`")

        #     # 修改：移除 while True 阻塞循环，改用简单的状态显示
        #     # 真正的实时监控逻辑移到下方的“实时监控”区域
        #     try:
        #         status_indicator()
                
        #     except:
        #         st.warning("无法连接后端获取状态")

    # ==================== 3. 实时监控 ====================
    st.divider()
    st.subheader("3. 📊 实时训练监控")

    monitor_task_id = st.text_input(
        "输入任务 ID 以查看训练曲线",
        value=st.session_state["task_id"]
    )

    if monitor_task_id:
        # 创建标签页用于Loss曲线
        st.subheader("📈 训练曲线")
        plot_loss_curve(monitor_task_id)


# ==================== Tab 2: 模型对话对比 ====================
with tab2:
    st.subheader("🤖 微调效果验证")

    # 侧边栏控制
    with st.sidebar:
        st.header("🎮 对话控制")
        use_lora = st.toggle("🔥 启用微调模型 (LoRA)", value=False)

        current_task = st.text_input(
            "任务 ID (自动填入)", value=st.session_state["task_id"]
        )
        if use_lora and not current_task:
            st.warning("⚠️ 请先在训练台完成训练，或手动输入Task ID")

        if st.button("🧹 清空对话历史"):
            st.session_state["chat_history"] = []

    # 聊天记录显示
    for role, text in st.session_state["chat_history"]:
        with st.chat_message(role):
            st.write(text)

    # 输入框
    if prompt := st.chat_input("输入你的问题（试试问'你是谁'）..."):
        # 1. 显示用户输入
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state["chat_history"].append(("user", prompt))

        # 2. 调用后端
        with st.chat_message("assistant"):
            with st.spinner("模型正在思考... (首次切换模型可能需要几秒加载)"):
                try:
                    payload = {
                        "query": prompt,
                        "task_id": current_task if current_task else None,
                        "use_lora": use_lora,
                    }
                    res = requests.post(f"{API_URL}/chat", json=payload)
                    response_text = res.json()["response"]

                    # 这里的 UI 优化：显示当前用的是什么模型
                    model_tag = "【🔥微调版】" if use_lora else "【🧊基座版】"
                    final_text = f"{model_tag} {response_text}"

                    st.write(final_text)
                    st.session_state["chat_history"].append(("assistant", final_text))

                except Exception as e:
                    st.error(f"连接失败: {e}")
