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

/* ── Page background ── */
html, body {
    min-height: 100vh;
    background: #0f0f1a !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}

.gradio-container {
    background: #0f0f1a !important;
    min-height: 100vh !important;
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── Outer wrapper: full-width layout ── */
.app-outer {
    display: flex;
    flex-direction: column;
    min-height: 100vh;
    width: 100%;
    max-width: 1100px;
    margin: 0 auto;
    padding: 0 24px 32px;
}

/* ── Header ── */
.header-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 40px 0 24px;
    gap: 10px;
}

.app-logo {
    width: 52px; height: 52px;
    background: linear-gradient(135deg, #6366f1, #a855f7);
    border-radius: 16px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.6rem;
    box-shadow: 0 8px 24px rgba(99,102,241,0.4);
    margin-bottom: 4px;
}

.app-title {
    font-size: clamp(1.6rem, 4vw, 2.4rem);
    font-weight: 800;
    background: linear-gradient(135deg, #e2e8ff 0%, #a5b4fc 50%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.5px;
    line-height: 1.2;
}

.app-subtitle {
    color: rgba(148,163,184,0.9);
    font-size: 0.9rem;
    font-weight: 400;
    letter-spacing: 0.01em;
}

.tool-badges {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 8px;
    margin-top: 4px;
}

.badge {
    padding: 5px 14px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    background: rgba(99,102,241,0.12);
    color: #a5b4fc;
    border: 1px solid rgba(99,102,241,0.25);
}

/* ── Main chat card ── */
.chat-card {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 24px !important;
    box-shadow: 0 32px 80px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.05) !important;
    overflow: hidden !important;
    backdrop-filter: blur(20px) !important;
    flex: 1;
}

/* ── Chatbot area ── */
.chatbot-box {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 8px 0 !important;
}

/* message bubbles */
.chatbot-box .message {
    font-size: 0.92rem !important;
    line-height: 1.65 !important;
    padding: 12px 16px !important;
}

.chatbot-box .message.user {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: #fff !important;
    border-radius: 20px 20px 6px 20px !important;
    box-shadow: 0 4px 16px rgba(79,70,229,0.35) !important;
}

.chatbot-box .message.bot {
    background: rgba(255,255,255,0.06) !important;
    color: #e2e8f0 !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 20px 20px 20px 6px !important;
}

/* divider between chatbot and input */
.chat-divider {
    height: 1px;
    background: rgba(255,255,255,0.06);
    margin: 0 20px;
}

/* ── Input section ── */
.input-area {
    padding: 16px 20px 12px !important;
    background: transparent !important;
}

.input-box textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1.5px solid rgba(255,255,255,0.1) !important;
    border-radius: 16px !important;
    color: #f1f5f9 !important;
    font-size: 0.94rem !important;
    font-family: 'Inter', sans-serif !important;
    padding: 14px 18px !important;
    resize: none !important;
    line-height: 1.55 !important;
    transition: border-color 0.2s, box-shadow 0.2s, background 0.2s !important;
}

.input-box textarea:focus {
    border-color: rgba(99,102,241,0.6) !important;
    background: rgba(255,255,255,0.08) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
    outline: none !important;
}

.input-box textarea::placeholder { color: rgba(148,163,184,0.5) !important; }

.send-btn {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    border: none !important;
    border-radius: 14px !important;
    color: #fff !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.02em !important;
    padding: 14px 26px !important;
    box-shadow: 0 4px 16px rgba(79,70,229,0.45) !important;
    transition: all 0.2s ease !important;
    white-space: nowrap !important;
    min-width: 90px !important;
    height: 48px !important;
}

.send-btn:hover {
    background: linear-gradient(135deg, #4338ca 0%, #6d28d9 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(79,70,229,0.55) !important;
}

.send-btn:active { transform: translateY(0px) !important; }

/* ── Bottom bar ── */
.bottom-bar {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    padding: 0 20px 16px !important;
    background: transparent !important;
}

.clear-btn {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    color: rgba(148,163,184,0.7) !important;
    font-size: 0.78rem !important;
    padding: 6px 14px !important;
    transition: all 0.15s !important;
    cursor: pointer !important;
}

.clear-btn:hover {
    background: rgba(255,255,255,0.09) !important;
    color: #94a3b8 !important;
    border-color: rgba(255,255,255,0.14) !important;
}

/* ── Examples section ── */
.examples-section {
    padding: 0 20px 20px !important;
    background: transparent !important;
}

.examples-label {
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: rgba(148,163,184,0.5);
    margin-bottom: 10px;
}

.examples-section .examples table,
.examples-section .examples table tbody,
.examples-section .examples table tr {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 8px !important;
    border: none !important;
    background: transparent !important;
}

.examples-section .examples table td {
    display: inline-flex !important;
    align-items: center !important;
    padding: 7px 15px !important;
    background: rgba(99,102,241,0.08) !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 999px !important;
    font-size: 0.76rem !important;
    font-weight: 500 !important;
    color: #a5b4fc !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
    white-space: nowrap !important;
    line-height: 1 !important;
}

.examples-section .examples table td:hover {
    background: rgba(99,102,241,0.2) !important;
    border-color: rgba(99,102,241,0.45) !important;
    color: #c7d2fe !important;
    transform: translateY(-1px) !important;
}

/* ── Footer ── */
.footer {
    text-align: center;
    color: rgba(148,163,184,0.35);
    font-size: 0.7rem;
    padding: 20px 0 8px;
    letter-spacing: 0.03em;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.3); border-radius: 99px; }

/* ── Tablet ── */
@media (max-width: 900px) {
    .app-outer { max-width: 100%; padding: 0 16px 24px; }
    .header-wrap { padding: 28px 0 18px; }
}

/* ── Mobile ── */
@media (max-width: 600px) {
    .app-outer { padding: 0 10px 20px; }
    .header-wrap { padding: 22px 0 14px; gap: 7px; }
    .app-logo { width: 42px; height: 42px; font-size: 1.3rem; border-radius: 12px; }
    .app-title { font-size: 1.5rem; }
    .app-subtitle { font-size: 0.78rem; }
    .badge { font-size: 0.65rem; padding: 4px 10px; }
    .chat-card { border-radius: 18px !important; }
    .input-area { padding: 12px 14px 10px !important; }
    .send-btn { padding: 14px 16px !important; min-width: 60px !important; font-size: 0.82rem !important; }
    .bottom-bar { padding: 0 14px 12px !important; }
    .examples-section { padding: 0 14px 16px !important; }
    .examples-section .examples table td { font-size: 0.72rem !important; padding: 6px 12px !important; }
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

    with gr.Column(elem_classes="app-outer"):

        gr.HTML("""
        <div class="header-wrap">
            <div class="app-logo">✨</div>
            <h1 class="app-title">AI Agent</h1>
            <p class="app-subtitle">Powered by GPT-4o &nbsp;·&nbsp; Web Search &nbsp;·&nbsp; Image Generation &nbsp;·&nbsp; HuggingFace</p>
            <div class="tool-badges">
                <span class="badge">🔍 Web Search</span>
                <span class="badge">🎨 Image Gen</span>
                <span class="badge">🕐 Timezone</span>
                <span class="badge">🌐 Browse</span>
                <span class="badge">⚡ Code</span>
            </div>
        </div>
        """)

        with gr.Group(elem_classes="chat-card"):

            chatbot = gr.Chatbot(
                value=[],
                height=500,
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

            with gr.Group(elem_classes="examples-section"):
                gr.HTML('<div class="examples-label">✦ Try an example</div>')
                gr.Examples(examples=EXAMPLE_PROMPTS, inputs=user_input, label="")

        gr.HTML('<div class="footer">Built with 🤗 smolagents &nbsp;·&nbsp; Gradio &nbsp;·&nbsp; GPT-4o &nbsp;·&nbsp; FLUX</div>')

    send_btn.click(fn=run_agent, inputs=[user_input, chatbot], outputs=[chatbot, user_input])
    user_input.submit(fn=run_agent, inputs=[user_input, chatbot], outputs=[chatbot, user_input])
    clear_btn.click(fn=lambda: ([], ""), outputs=[chatbot, user_input])

# =====================================================
# LAUNCH
# =====================================================

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7861)), theme=gr.themes.Default(font=gr.themes.GoogleFont("Inter")))
