from smolagents import (
    CodeAgent,
    FinalAnswerTool,
    LiteLLMModel,
    DuckDuckGoSearchTool,
    VisitWebpageTool,
    tool
)

from smolagents.agent_types import AgentImage
from huggingface_hub import InferenceClient

import gradio as gr
import tempfile
import datetime
import pytz
import yaml
import os

from dotenv import load_dotenv

# =====================================================
# LOAD ENV
# =====================================================

load_dotenv()

print("OPENAI_API_KEY =", os.getenv("OPENAI_API_KEY"))
print("HF_TOKEN_ =", os.getenv("HF_TOKEN_"))

HF_TOKEN = os.getenv("HF_TOKEN_")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# =====================================================
# TIME TOOL
# =====================================================

@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """
    Get current time in a timezone.

    Args:
        timezone: Valid timezone such as Asia/Kolkata
    """
    try:
        tz = pytz.timezone(timezone)
        current_time = datetime.datetime.now(tz)
        return current_time.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        return str(e)

# =====================================================
# IMAGE TOOL
# =====================================================

@tool
def generate_image(prompt: str) -> AgentImage:
    """
    Generate an image from a text prompt.

    Args:
        prompt: Detailed image description.
    """
    client = InferenceClient(token=HF_TOKEN)
    image = client.text_to_image(
        prompt=prompt,
        model="black-forest-labs/FLUX.1-schnell"
    )
    return AgentImage(image)

# =====================================================
# FINAL ANSWER TOOL
# =====================================================

final_answer = FinalAnswerTool()

# =====================================================
# MODEL
# =====================================================

model = LiteLLMModel(
    model_id="gpt-4o",
    api_key=OPENAI_API_KEY,
    max_tokens=4096,
    temperature=0.3,
)

# =====================================================
# PROMPTS
# =====================================================

try:
    with open("prompts.yaml", "r", encoding="utf-8") as f:
        prompt_templates = yaml.safe_load(f)
except FileNotFoundError:
    prompt_templates = None

# =====================================================
# AGENT
# =====================================================

agent = CodeAgent(
    model=model,
    tools=[
        DuckDuckGoSearchTool(),
        VisitWebpageTool(),
        generate_image,
        get_current_time_in_timezone,
        final_answer,
    ],
    max_steps=6,
    verbosity_level=1,
    prompt_templates=prompt_templates,
)

# =====================================================
# CHAT LOGIC
# =====================================================

def run_agent(user_message, history):
    if not user_message.strip():
        yield history, ""
        return
    history = history + [{"role": "user", "content": user_message}]
    history = history + [{"role": "assistant", "content": "⏳ Searching and thinking..."}]
    yield history, ""
    try:
        response = agent.run(user_message)
        if isinstance(response, AgentImage):
            pil_img = response.to_raw()
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            pil_img.save(tmp.name)
            history[-1] = {"role": "assistant", "content": {"path": tmp.name, "mime_type": "image/png"}}
        else:
            history[-1] = {"role": "assistant", "content": str(response)}
    except Exception as e:
        history[-1] = {"role": "assistant", "content": f"⚠️ Error: {str(e)}"}
    yield history, ""

# =====================================================
# CUSTOM CSS
# =====================================================

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body { min-height: 100vh; background: #080810 !important; font-family: 'Inter','Segoe UI',sans-serif !important; }
.gradio-container { background: #080810 !important; min-height: 100vh !important; padding: 0 !important; max-width: 100% !important; }

/* PAGE SHELL */
.page-shell { width: min(1360px, 100%); margin: 0 auto; padding: 0 36px 48px; }

/* HEADER */
.header-wrap {
    display: flex; align-items: center; justify-content: space-between;
    padding: 26px 0 22px; border-bottom: 1px solid rgba(255,255,255,0.1);
    margin-bottom: 28px; gap: 20px; flex-wrap: wrap;
}
.header-left { display: flex; align-items: center; gap: 14px; }
.app-logo {
    width: 48px; height: 48px; background: linear-gradient(135deg,#6366f1,#a855f7);
    border-radius: 13px; display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem; box-shadow: 0 8px 24px rgba(99,102,241,0.5); flex-shrink: 0;
}
.app-title {
    font-size: 1.5rem; font-weight: 800;
    background: linear-gradient(135deg,#ffffff,#c4b5fd,#a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; letter-spacing: -0.4px;
}
.app-subtitle { color: #94a3b8; font-size: 0.875rem; margin-top: 3px; font-weight: 400; }
.tool-badges { display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; }
.badge {
    padding: 6px 14px; border-radius: 999px; font-size: 0.78rem; font-weight: 600;
    background: rgba(99,102,241,0.15); color: #c4b5fd; border: 1px solid rgba(139,92,246,0.35); white-space: nowrap;
    letter-spacing: 0.01em;
}

/* TWO-COLUMN GRID */
.content-grid { display: grid; grid-template-columns: 1fr 320px; gap: 24px; align-items: start; }

/* CHAT CARD */
.chat-card {
    background: #12121f !important; border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 20px !important; box-shadow: 0 24px 64px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.06) !important;
    overflow: hidden !important;
}
.chatbot-box { background: transparent !important; border: none !important; border-radius: 0 !important; }
.chatbot-box .message { font-size: 0.975rem !important; line-height: 1.7 !important; padding: 13px 17px !important; }
.chatbot-box .message.user {
    background: linear-gradient(135deg,#4f46e5,#7c3aed) !important; color: #fff !important;
    border-radius: 18px 18px 5px 18px !important; box-shadow: 0 4px 16px rgba(79,70,229,0.4) !important;
}
.chatbot-box .message.bot {
    background: #1e1e30 !important; color: #e2e8f0 !important;
    border: 1px solid rgba(255,255,255,0.1) !important; border-radius: 18px 18px 18px 5px !important;
}
.chat-divider { height: 1px; background: rgba(255,255,255,0.08); margin: 0 20px; }
.input-area { padding: 16px 20px 12px !important; background: transparent !important; }
.input-box textarea {
    background: #1a1a2e !important; border: 1.5px solid rgba(255,255,255,0.12) !important;
    border-radius: 14px !important; color: #f1f5f9 !important; font-size: 0.975rem !important;
    font-family: 'Inter',sans-serif !important; padding: 14px 18px !important; resize: none !important;
    line-height: 1.6 !important; transition: border-color 0.2s, box-shadow 0.2s, background 0.2s !important;
}
.input-box textarea:focus {
    border-color: rgba(139,92,246,0.7) !important; background: #1e1e35 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.18) !important; outline: none !important;
}
.input-box textarea::placeholder { color: #64748b !important; }
.send-btn {
    background: linear-gradient(135deg,#4f46e5,#7c3aed) !important; border: none !important;
    border-radius: 13px !important; color: #fff !important; font-weight: 700 !important;
    font-size: 0.925rem !important; padding: 14px 26px !important;
    box-shadow: 0 4px 16px rgba(79,70,229,0.5) !important; transition: all 0.2s !important;
    white-space: nowrap !important; min-width: 90px !important;
}
.send-btn:hover { background: linear-gradient(135deg,#4338ca,#6d28d9) !important; transform: translateY(-2px) !important; box-shadow: 0 8px 24px rgba(79,70,229,0.6) !important; }
.send-btn:active { transform: translateY(0) !important; }
.bottom-bar { display: flex; justify-content: flex-end; padding: 6px 20px 16px !important; background: transparent !important; }
.clear-btn {
    background: transparent !important; border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 9px !important; color: #94a3b8 !important;
    font-size: 0.825rem !important; padding: 6px 14px !important; transition: all 0.15s !important;
}
.clear-btn:hover { background: rgba(255,255,255,0.07) !important; color: #cbd5e1 !important; border-color: rgba(255,255,255,0.2) !important; }

/* SIDEBAR */
.sidebar { display: flex; flex-direction: column; gap: 18px; position: sticky; top: 24px; }
.sidebar-card {
    background: #12121f !important; border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 18px !important; overflow: hidden !important;
}
.sidebar-title {
    font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em;
    color: #94a3b8; padding: 16px 18px 12px; border-bottom: 1px solid rgba(255,255,255,0.08);
}
.examples-section { padding: 14px 16px 16px !important; background: transparent !important; }
.examples-section .examples table,
.examples-section .examples table tbody,
.examples-section .examples table tr {
    display: flex !important; flex-direction: column !important; gap: 9px !important;
    border: none !important; background: transparent !important; width: 100% !important;
}
.examples-section .examples table td {
    display: block !important; width: 100% !important; padding: 11px 15px !important;
    background: rgba(99,102,241,0.1) !important; border: 1px solid rgba(139,92,246,0.25) !important;
    border-radius: 11px !important; font-size: 0.85rem !important; font-weight: 500 !important;
    color: #c4b5fd !important; cursor: pointer !important; transition: all 0.15s !important;
    white-space: normal !important; line-height: 1.45 !important;
}
.examples-section .examples table td:hover {
    background: rgba(99,102,241,0.22) !important; border-color: rgba(139,92,246,0.55) !important;
    color: #e9d5ff !important; transform: translateX(4px) !important;
}

/* INFO CARD */
.info-card { background: #12121f; border: 1px solid rgba(255,255,255,0.1); border-radius: 18px; overflow: hidden; }
.info-card-title {
    font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em;
    color: #94a3b8; padding: 16px 18px 12px; border-bottom: 1px solid rgba(255,255,255,0.08);
}
.capability-item { display: flex; align-items: flex-start; gap: 12px; padding: 12px 18px; border-bottom: 1px solid rgba(255,255,255,0.06); }
.capability-item:last-child { border-bottom: none; }
.cap-icon { font-size: 1.05rem; width: 32px; height: 32px; background: rgba(99,102,241,0.15); border-radius: 9px; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
.cap-text strong { display: block; font-size: 0.85rem; font-weight: 700; color: #e2e8f0; margin-bottom: 2px; }
.cap-text span { font-size: 0.775rem; color: #94a3b8; line-height: 1.45; }

/* FOOTER */
.footer { text-align: center; color: #475569; font-size: 0.775rem; padding: 28px 0 8px; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.4); border-radius: 99px; }

/* TABLET */
@media (max-width: 900px) {
    .page-shell { padding: 0 22px 32px; }
    .content-grid { grid-template-columns: 1fr; }
    .sidebar { position: static; flex-direction: row; flex-wrap: wrap; gap: 16px; }
    .sidebar-card, .info-card { flex: 1; min-width: 260px; }
    .examples-section .examples table, .examples-section .examples table tbody, .examples-section .examples table tr { flex-direction: row !important; flex-wrap: wrap !important; }
    .examples-section .examples table td { width: auto !important; white-space: nowrap !important; }
}

/* MOBILE */
@media (max-width: 600px) {
    .page-shell { padding: 0 14px 24px; }
    .header-wrap { padding: 18px 0 16px; flex-direction: column; align-items: flex-start; gap: 12px; }
    .app-title { font-size: 1.25rem; }
    .app-subtitle { font-size: 0.8rem; }
    .tool-badges { justify-content: flex-start; }
    .badge { font-size: 0.72rem; padding: 5px 11px; }
    .chat-card { border-radius: 16px !important; }
    .input-area { padding: 13px 13px 10px !important; }
    .input-box textarea { font-size: 0.9rem !important; }
    .send-btn { padding: 14px 16px !important; min-width: 58px !important; font-size: 0.875rem !important; }
    .bottom-bar { padding: 4px 13px 14px !important; }
    .sidebar { flex-direction: column; }
    .sidebar-card, .info-card { min-width: unset !important; width: 100%; }
    .examples-section .examples table td { font-size: 0.82rem !important; padding: 10px 13px !important; }
    .cap-text strong { font-size: 0.82rem; }
    .cap-text span { font-size: 0.75rem; }
}
"""

# =====================================================
# GRADIO UI
# =====================================================

EXAMPLE_PROMPTS = [
    "What are the latest news about AI today?",
    "What time is it in Tokyo right now?",
    "Generate an image of a futuristic city at night with neon lights",
    "Search the web and summarize: what is the current price of Bitcoin?",
    "What happened in the world this week?",
    "Create an image of a robot reading books in a cozy library",
]

with gr.Blocks(title="✨ AI Agent") as demo:

    gr.HTML(f"<style>{CSS}</style>")

    with gr.Column(elem_classes="page-shell"):

        # HEADER
        gr.HTML("""
        <div class="header-wrap">
            <div class="header-left">
                <div class="app-logo">✨</div>
                <div class="header-text">
                    <div class="app-title">AI Agent</div>
                    <div class="app-subtitle">Powered by GPT-4o &nbsp;·&nbsp; Web Search &nbsp;·&nbsp; Image Generation</div>
                </div>
            </div>
            <div class="tool-badges">
                <span class="badge">🔍 Web Search</span>
                <span class="badge">🎨 Image Gen</span>
                <span class="badge">🕐 Timezone</span>
                <span class="badge">🌐 Browse</span>
                <span class="badge">⚡ Code</span>
            </div>
        </div>
        """)

        # TWO-COLUMN GRID
        with gr.Row(elem_classes="content-grid"):

            # LEFT: Chat panel
            with gr.Column(elem_classes="chat-card"):
                chatbot = gr.Chatbot(
                    value=[],
                    height=520,
                    show_label=False,
                    elem_classes="chatbot-box",
                    avatar_images=(None, "https://huggingface.co/front/assets/huggingface_logo-noborder.svg"),
                )
                gr.HTML('<div class="chat-divider"></div>')
                with gr.Group(elem_classes="input-area"):
                    with gr.Row():
                        user_input = gr.Textbox(
                            placeholder="Ask me anything — search the web, generate images, check timezones...",
                            show_label=False,
                            lines=1,
                            max_lines=4,
                            scale=7,
                            elem_classes="input-box",
                        )
                        send_btn = gr.Button("Send ➤", scale=1, elem_classes="send-btn")
                with gr.Row(elem_classes="bottom-bar"):
                    clear_btn = gr.Button("🗑 Clear chat", elem_classes="clear-btn")

            # RIGHT: Sidebar
            with gr.Column(elem_classes="sidebar"):
                with gr.Group(elem_classes="sidebar-card"):
                    gr.HTML('<div class="sidebar-title">✦ Try an example</div>')
                    with gr.Group(elem_classes="examples-section"):
                        gr.Examples(examples=EXAMPLE_PROMPTS, inputs=user_input, label="")

                gr.HTML("""
                <div class="info-card">
                    <div class="info-card-title">⚡ Capabilities</div>
                    <div class="capability-item">
                        <div class="cap-icon">🔍</div>
                        <div class="cap-text"><strong>Web Search</strong><span>Real-time DuckDuckGo search results</span></div>
                    </div>
                    <div class="capability-item">
                        <div class="cap-icon">🎨</div>
                        <div class="cap-text"><strong>Image Generation</strong><span>FLUX model via HuggingFace</span></div>
                    </div>
                    <div class="capability-item">
                        <div class="cap-icon">🌐</div>
                        <div class="cap-text"><strong>Web Browsing</strong><span>Visit and summarize any webpage</span></div>
                    </div>
                    <div class="capability-item">
                        <div class="cap-icon">🕐</div>
                        <div class="cap-text"><strong>Timezone Tool</strong><span>Current time in any timezone</span></div>
                    </div>
                    <div class="capability-item">
                        <div class="cap-icon">⚡</div>
                        <div class="cap-text"><strong>Code Execution</strong><span>Python reasoning &amp; computation</span></div>
                    </div>
                </div>
                """)

        gr.HTML('<div class="footer">Built with 🤗 smolagents &nbsp;·&nbsp; Gradio &nbsp;·&nbsp; GPT-4o &nbsp;·&nbsp; FLUX</div>')

    send_btn.click(fn=run_agent, inputs=[user_input, chatbot], outputs=[chatbot, user_input])
    user_input.submit(fn=run_agent, inputs=[user_input, chatbot], outputs=[chatbot, user_input])
    clear_btn.click(fn=lambda: ([], ""), outputs=[chatbot, user_input])

# =====================================================
# LAUNCH
# =====================================================

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7861)), theme=gr.themes.Default(font=gr.themes.GoogleFont("Inter")))
