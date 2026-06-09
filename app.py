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

html, body { min-height: 100vh; background: #0b0b14 !important; font-family: 'Inter','Segoe UI',sans-serif !important; }
.gradio-container { background: #0b0b14 !important; min-height: 100vh !important; padding: 0 !important; max-width: 100% !important; }

/* PAGE SHELL */
.page-shell { width: min(1360px, 100%); margin: 0 auto; padding: 0 32px 40px; }

/* HEADER */
.header-wrap {
    display: flex; align-items: center; justify-content: space-between;
    padding: 22px 0 18px; border-bottom: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 22px; gap: 16px; flex-wrap: wrap;
}
.header-left { display: flex; align-items: center; gap: 12px; }
.app-logo {
    width: 42px; height: 42px; background: linear-gradient(135deg,#6366f1,#a855f7);
    border-radius: 11px; display: flex; align-items: center; justify-content: center;
    font-size: 1.3rem; box-shadow: 0 6px 18px rgba(99,102,241,0.4); flex-shrink: 0;
}
.app-title {
    font-size: 1.25rem; font-weight: 800;
    background: linear-gradient(135deg,#e2e8ff,#a5b4fc,#c084fc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; letter-spacing: -0.3px;
}
.app-subtitle { color: rgba(148,163,184,0.6); font-size: 0.72rem; margin-top: 2px; }
.tool-badges { display: flex; flex-wrap: wrap; gap: 6px; justify-content: flex-end; }
.badge {
    padding: 4px 11px; border-radius: 999px; font-size: 0.66rem; font-weight: 600;
    background: rgba(99,102,241,0.1); color: #a5b4fc; border: 1px solid rgba(99,102,241,0.2); white-space: nowrap;
}

/* TWO-COLUMN GRID */
.content-grid { display: grid; grid-template-columns: 1fr 290px; gap: 20px; align-items: start; }

/* CHAT CARD */
.chat-card {
    background: rgba(255,255,255,0.025) !important; border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 20px !important; box-shadow: 0 24px 60px rgba(0,0,0,0.5),inset 0 1px 0 rgba(255,255,255,0.04) !important;
    overflow: hidden !important; backdrop-filter: blur(16px) !important;
}
.chatbot-box { background: transparent !important; border: none !important; border-radius: 0 !important; }
.chatbot-box .message { font-size: 0.91rem !important; line-height: 1.65 !important; padding: 11px 15px !important; }
.chatbot-box .message.user {
    background: linear-gradient(135deg,#4f46e5,#7c3aed) !important; color: #fff !important;
    border-radius: 18px 18px 5px 18px !important; box-shadow: 0 4px 14px rgba(79,70,229,0.3) !important;
}
.chatbot-box .message.bot {
    background: rgba(255,255,255,0.055) !important; color: #dde4f0 !important;
    border: 1px solid rgba(255,255,255,0.08) !important; border-radius: 18px 18px 18px 5px !important;
}
.chat-divider { height: 1px; background: rgba(255,255,255,0.055); margin: 0 18px; }
.input-area { padding: 14px 18px 10px !important; background: transparent !important; }
.input-box textarea {
    background: rgba(255,255,255,0.05) !important; border: 1.5px solid rgba(255,255,255,0.09) !important;
    border-radius: 14px !important; color: #f1f5f9 !important; font-size: 0.92rem !important;
    font-family: 'Inter',sans-serif !important; padding: 13px 16px !important; resize: none !important;
    line-height: 1.55 !important; transition: border-color 0.2s, box-shadow 0.2s, background 0.2s !important;
}
.input-box textarea:focus {
    border-color: rgba(99,102,241,0.55) !important; background: rgba(255,255,255,0.075) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.13) !important; outline: none !important;
}
.input-box textarea::placeholder { color: rgba(148,163,184,0.45) !important; }
.send-btn {
    background: linear-gradient(135deg,#4f46e5,#7c3aed) !important; border: none !important;
    border-radius: 12px !important; color: #fff !important; font-weight: 700 !important;
    font-size: 0.86rem !important; padding: 13px 22px !important;
    box-shadow: 0 4px 14px rgba(79,70,229,0.4) !important; transition: all 0.2s !important;
    white-space: nowrap !important; min-width: 84px !important;
}
.send-btn:hover { background: linear-gradient(135deg,#4338ca,#6d28d9) !important; transform: translateY(-2px) !important; box-shadow: 0 8px 22px rgba(79,70,229,0.5) !important; }
.send-btn:active { transform: translateY(0) !important; }
.bottom-bar { display: flex; justify-content: flex-end; padding: 4px 18px 14px !important; background: transparent !important; }
.clear-btn {
    background: transparent !important; border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 8px !important; color: rgba(148,163,184,0.5) !important;
    font-size: 0.73rem !important; padding: 5px 12px !important; transition: all 0.15s !important;
}
.clear-btn:hover { background: rgba(255,255,255,0.06) !important; color: #94a3b8 !important; border-color: rgba(255,255,255,0.12) !important; }

/* SIDEBAR */
.sidebar { display: flex; flex-direction: column; gap: 16px; position: sticky; top: 24px; }
.sidebar-card {
    background: rgba(255,255,255,0.025) !important; border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 16px !important; overflow: hidden !important;
}
.sidebar-title {
    font-size: 0.62rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em;
    color: rgba(148,163,184,0.4); padding: 13px 16px 10px; border-bottom: 1px solid rgba(255,255,255,0.05);
}
.examples-section { padding: 12px 14px 14px !important; background: transparent !important; }
.examples-section .examples table,
.examples-section .examples table tbody,
.examples-section .examples table tr {
    display: flex !important; flex-direction: column !important; gap: 7px !important;
    border: none !important; background: transparent !important; width: 100% !important;
}
.examples-section .examples table td {
    display: block !important; width: 100% !important; padding: 9px 13px !important;
    background: rgba(99,102,241,0.07) !important; border: 1px solid rgba(99,102,241,0.18) !important;
    border-radius: 10px !important; font-size: 0.75rem !important; font-weight: 500 !important;
    color: #a5b4fc !important; cursor: pointer !important; transition: all 0.15s !important;
    white-space: normal !important; line-height: 1.4 !important;
}
.examples-section .examples table td:hover {
    background: rgba(99,102,241,0.17) !important; border-color: rgba(99,102,241,0.4) !important;
    color: #c7d2fe !important; transform: translateX(3px) !important;
}

/* INFO CARD */
.info-card { background: rgba(255,255,255,0.025); border: 1px solid rgba(255,255,255,0.07); border-radius: 16px; overflow: hidden; }
.info-card-title {
    font-size: 0.62rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em;
    color: rgba(148,163,184,0.4); padding: 13px 16px 10px; border-bottom: 1px solid rgba(255,255,255,0.05);
}
.capability-item { display: flex; align-items: flex-start; gap: 10px; padding: 10px 16px; border-bottom: 1px solid rgba(255,255,255,0.04); }
.capability-item:last-child { border-bottom: none; }
.cap-icon { font-size: 0.95rem; width: 28px; height: 28px; background: rgba(99,102,241,0.1); border-radius: 8px; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
.cap-text strong { display: block; font-size: 0.74rem; font-weight: 600; color: #cbd5e1; }
.cap-text span { font-size: 0.67rem; color: rgba(148,163,184,0.5); line-height: 1.4; }

/* FOOTER */
.footer { text-align: center; color: rgba(148,163,184,0.28); font-size: 0.67rem; padding: 22px 0 6px; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.3); border-radius: 99px; }

/* TABLET */
@media (max-width: 900px) {
    .page-shell { padding: 0 20px 28px; }
    .content-grid { grid-template-columns: 1fr; }
    .sidebar { position: static; flex-direction: row; flex-wrap: wrap; gap: 14px; }
    .sidebar-card, .info-card { flex: 1; min-width: 240px; }
    .examples-section .examples table, .examples-section .examples table tbody, .examples-section .examples table tr { flex-direction: row !important; flex-wrap: wrap !important; }
    .examples-section .examples table td { width: auto !important; white-space: nowrap !important; }
}

/* MOBILE */
@media (max-width: 600px) {
    .page-shell { padding: 0 12px 20px; }
    .header-wrap { padding: 16px 0 14px; flex-direction: column; align-items: flex-start; gap: 10px; }
    .tool-badges { justify-content: flex-start; }
    .chat-card { border-radius: 16px !important; }
    .input-area { padding: 12px 12px 8px !important; }
    .send-btn { padding: 13px 14px !important; min-width: 56px !important; }
    .sidebar { flex-direction: column; }
    .sidebar-card, .info-card { min-width: unset !important; width: 100%; }
    .examples-section .examples table td { font-size: 0.73rem !important; }
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
