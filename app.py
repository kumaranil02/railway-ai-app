from smolagents import (
    CodeAgent,
    FinalAnswerTool,
    LiteLLMModel,
    DuckDuckGoSearchTool,
    VisitWebpageTool,
    tool,
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

HF_TOKEN = os.getenv("HF_TOKEN_", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

print("OPENAI_API_KEY present:", bool(OPENAI_API_KEY))
print("HF_TOKEN_ present:", bool(HF_TOKEN))

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing from the environment")

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
        model="black-forest-labs/FLUX.1-schnell",
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
    if not user_message or not user_message.strip():
        yield history, ""
        return

    if history is None:
        history = []

    # Gradio Chatbot in messages mode expects dicts with role/content.
    history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": "⏳ Thinking..."},
    ]
    yield history, ""

    try:
        response = agent.run(user_message)

        if isinstance(response, AgentImage):
            pil_img = response.to_raw()
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            pil_img.save(tmp.name)

            history[-1] = {
                "role": "assistant",
                "content": {
                    "path": tmp.name,
                    "mime_type": "image/png",
                },
            }
        else:
            history[-1] = {
                "role": "assistant",
                "content": str(response),
            }

    except Exception as e:
        history[-1] = {
            "role": "assistant",
            "content": f"⚠️ Error: {str(e)}",
        }

    yield history, ""


# =====================================================
# CUSTOM CSS - NATURE LANDSCAPE THEME
# =====================================================

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root {
  --sky-1: #dff6ff;
  --sky-2: #bfe7ff;
  --sky-3: #8fd2ff;
  --forest-1: #104233;
  --forest-2: #17664d;
  --forest-3: #2d8a5e;
  --leaf-1: #61c98f;
  --leaf-2: #9ae7b3;
  --card: rgba(245, 249, 244, 0.76);
  --card-strong: rgba(237, 245, 238, 0.90);
  --line: rgba(23, 102, 77, 0.18);
  --line-strong: rgba(23, 102, 77, 0.28);
  --text: #163024;
  --text-strong: #10271e;
  --text-muted: #496255;
  --accent: #2d8a5e;
  --accent-2: #1f6f52;
  --shadow: 0 24px 70px rgba(14, 40, 29, 0.22);
}

* { box-sizing: border-box; }
html, body { min-height: 100%; }
body {
  margin: 0;
  font-family: 'Inter', 'Segoe UI', sans-serif !important;
  color: var(--text);
  background:
    linear-gradient(180deg, #dff6ff 0%, #c8eff8 18%, #f6fbf4 18%, #ebf8ee 100%) !important;
}

/* scenic landscape layers */
body::before {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  background:
    radial-gradient(circle at 15% 14%, rgba(255, 255, 255, 0.92) 0 6%, transparent 7%),
    radial-gradient(circle at 20% 12%, rgba(255, 248, 220, 0.40) 0 10%, transparent 11%),
    linear-gradient(180deg, rgba(255,255,255,0.35) 0%, rgba(255,255,255,0.00) 18%),
    linear-gradient(160deg, transparent 0 58%, rgba(34, 120, 83, 0.12) 58% 66%, rgba(22, 76, 55, 0.18) 66% 75%, transparent 75%),
    linear-gradient(175deg, transparent 0 62%, rgba(79, 173, 120, 0.16) 62% 73%, rgba(34, 120, 83, 0.20) 73% 82%, transparent 82%),
    linear-gradient(180deg, transparent 0 72%, rgba(26, 96, 68, 0.10) 72% 100%);
  opacity: 0.95;
  z-index: 0;
}

.gradio-container {
  max-width: 100% !important;
  min-height: 100vh !important;
  padding: 0 !important;
  background: transparent !important;
  position: relative;
  z-index: 1;
}

.page-shell {
  width: min(1500px, calc(100vw - 32px));
  margin: 0 auto;
  padding: 22px 0 28px;
}

/* HEADER */
.header-wrap {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 18px;
  padding: 8px 4px 18px;
  margin-bottom: 18px;
  border-bottom: 1px solid rgba(20, 60, 45, 0.12);
}

.header-left {
  display: flex;
  align-items: center;
  gap: 14px;
}

.app-logo {
  width: 54px;
  height: 54px;
  border-radius: 16px;
  display: grid;
  place-items: center;
  font-size: 1.35rem;
  background: linear-gradient(135deg, #2d8a5e 0%, #61c98f 45%, #b4f0c7 100%);
  box-shadow: 0 10px 24px rgba(45, 138, 94, 0.22), inset 0 1px 0 rgba(255, 255, 255, 0.36);
  color: #effbf3;
  flex-shrink: 0;
}

.header-text { display: flex; flex-direction: column; gap: 4px; }
.app-title {
  font-size: 2rem;
  line-height: 1;
  font-weight: 900;
  letter-spacing: -0.04em;
  color: var(--text-strong);
}

.app-subtitle {
  font-size: 0.95rem;
  color: var(--text-muted);
}

.tool-badges {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 10px;
}

.badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 8px 13px;
  border-radius: 999px;
  border: 1px solid rgba(23, 102, 77, 0.16);
  background: rgba(255, 255, 255, 0.70);
  color: var(--text-strong);
  font-size: 0.84rem;
  font-weight: 700;
  box-shadow: 0 8px 24px rgba(17, 60, 43, 0.06);
  white-space: nowrap;
}

/* GRID */
.content-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.35fr) minmax(340px, 0.85fr);
  gap: 20px;
  align-items: start;
}

/* CHAT CARD */
.chat-card {
  background: var(--card) !important;
  border: 1px solid rgba(23, 102, 77, 0.12) !important;
  border-radius: 26px !important;
  overflow: hidden !important;
  box-shadow: var(--shadow), inset 0 1px 0 rgba(255,255,255,0.45) !important;
  backdrop-filter: blur(18px) saturate(1.03) !important;
  min-height: 760px;
}

.chatbot-box {
  background: transparent !important;
  border: none !important;
  border-radius: 0 !important;
}

.chatbot-box .message {
  font-size: 1rem !important;
  line-height: 1.68 !important;
}

.chatbot-box .message.user {
  background: linear-gradient(135deg, #2d8a5e 0%, #61c98f 100%) !important;
  color: #f3fff5 !important;
  border-radius: 18px 18px 6px 18px !important;
  font-weight: 600 !important;
  box-shadow: 0 10px 18px rgba(45, 138, 94, 0.18) !important;
}

.chatbot-box .message.bot {
  background: rgba(255, 255, 255, 0.82) !important;
  color: var(--text-strong) !important;
  border: 1px solid rgba(23, 102, 77, 0.10) !important;
  border-radius: 18px 18px 18px 6px !important;
}

.chat-divider {
  height: 1px;
  margin: 0 20px;
  background: linear-gradient(90deg, transparent, rgba(23, 102, 77, 0.10), transparent);
}

.input-area {
  padding: 16px 20px 12px !important;
  background: transparent !important;
}

.composer-row {
  align-items: stretch !important;
  gap: 12px !important;
  padding: 12px !important;
  margin-top: 2px !important;
  border-radius: 22px !important;
  background: rgba(255, 255, 255, 0.62) !important;
  border: 1px solid rgba(23, 102, 77, 0.14) !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.70) !important;
}

.composer-row > div {
  min-width: 0 !important;
}

.input-box textarea {
  min-height: 58px !important;
  background: rgba(255, 255, 255, 0.92) !important;
  border: 1px solid rgba(23, 102, 77, 0.18) !important;
  border-radius: 16px !important;
  color: #153024 !important;
  font-size: 1rem !important;
  font-weight: 600 !important;
  padding: 16px 18px !important;
  resize: none !important;
  line-height: 1.5 !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.85) !important;
}

.input-box textarea:focus {
  outline: none !important;
  border-color: rgba(45, 138, 94, 0.45) !important;
  box-shadow: 0 0 0 3px rgba(45, 138, 94, 0.10) !important;
}

.input-box textarea::placeholder {
  color: rgba(76, 101, 89, 0.62) !important;
}

.send-btn {
  min-width: 118px !important;
  padding: 15px 18px !important;
  border: none !important;
  border-radius: 16px !important;
  background: linear-gradient(135deg, #2d8a5e 0%, #61c98f 100%) !important;
  color: #f7fff8 !important;
  font-weight: 800 !important;
  font-size: 0.96rem !important;
  box-shadow: 0 14px 24px rgba(45, 138, 94, 0.20) !important;
  transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease !important;
}

.send-btn:hover {
  transform: translateY(-1px) !important;
  filter: brightness(1.03) !important;
  box-shadow: 0 18px 30px rgba(45, 138, 94, 0.26) !important;
}

.send-btn:active { transform: translateY(0) !important; }

.bottom-bar {
  display: flex;
  justify-content: flex-end;
  padding: 0 20px 16px !important;
  background: transparent !important;
}

.clear-btn {
  background: transparent !important;
  border: 1px solid rgba(23, 102, 77, 0.12) !important;
  color: rgba(16, 39, 30, 0.72) !important;
  border-radius: 12px !important;
  padding: 8px 14px !important;
  font-size: 0.84rem !important;
  transition: all 0.15s ease !important;
}

.clear-btn:hover {
  background: rgba(255,255,255,0.68) !important;
  color: var(--text-strong) !important;
  border-color: rgba(45, 138, 94, 0.20) !important;
}

/* SIDEBAR */
.sidebar {
  display: flex;
  flex-direction: column;
  gap: 16px;
  position: sticky;
  top: 20px;
}

.sidebar-card,
.info-card {
  background: var(--card-strong) !important;
  border: 1px solid rgba(23, 102, 77, 0.12) !important;
  border-radius: 22px !important;
  box-shadow: var(--shadow), inset 0 1px 0 rgba(255,255,255,0.45) !important;
  overflow: hidden !important;
}

.panel-title {
  padding: 16px 18px 12px;
  border-bottom: 1px solid rgba(23, 102, 77, 0.08);
  font-size: 0.84rem;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: rgba(16, 39, 30, 0.86);
}

.panel-title span {
  color: var(--accent);
}

.examples-section {
  padding: 14px 16px 16px !important;
  background: transparent !important;
}

.examples-section .examples {
  margin-top: 2px !important;
}

.examples-section .examples table,
.examples-section .examples tbody,
.examples-section .examples tr {
  display: flex !important;
  flex-direction: column !important;
  gap: 10px !important;
  background: transparent !important;
  width: 100% !important;
  border: none !important;
}

.examples-section .examples td {
  display: block !important;
  width: 100% !important;
  padding: 13px 14px !important;
  border-radius: 14px !important;
  border: 1px solid rgba(45, 138, 94, 0.14) !important;
  background: linear-gradient(180deg, rgba(255,255,255,0.78), rgba(255,255,255,0.58)) !important;
  color: var(--text-strong) !important;
  font-size: 0.93rem !important;
  font-weight: 600 !important;
  line-height: 1.45 !important;
  white-space: normal !important;
  cursor: pointer !important;
  transition: transform 0.15s ease, border-color 0.15s ease, background 0.15s ease !important;
}

.examples-section .examples td:hover {
  transform: translateY(-1px) !important;
  border-color: rgba(45, 138, 94, 0.26) !important;
  background: linear-gradient(180deg, rgba(97,201,143,0.18), rgba(255,255,255,0.70)) !important;
}

.capability-list { padding: 6px 0 4px; }

.capability-item {
  display: flex;
  gap: 12px;
  align-items: flex-start;
  padding: 14px 18px;
  border-top: 1px solid rgba(23, 102, 77, 0.06);
}

.capability-item:first-child { border-top: none; }

.cap-icon {
  width: 34px;
  height: 34px;
  border-radius: 12px;
  display: grid;
  place-items: center;
  background: rgba(97, 201, 143, 0.12);
  border: 1px solid rgba(97, 201, 143, 0.12);
  flex-shrink: 0;
}

.cap-text strong {
  display: block;
  font-size: 0.98rem;
  color: var(--text-strong);
  font-weight: 700;
  margin-bottom: 3px;
}

.cap-text span {
  font-size: 0.88rem;
  color: var(--text-muted);
  line-height: 1.45;
}

.footer {
  text-align: center;
  font-size: 0.80rem;
  color: rgba(16, 39, 30, 0.42);
  padding: 22px 0 8px;
}

/* SCROLLBAR */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(45, 138, 94, 0.28); border-radius: 999px; }

/* TABLET */
@media (max-width: 1100px) {
  .page-shell { width: min(100vw - 24px, 1500px); }
  .content-grid { grid-template-columns: 1fr; }
  .sidebar { position: static; }
}

/* MOBILE */
@media (max-width: 640px) {
  .page-shell { width: calc(100vw - 16px); padding: 12px 0 20px; }
  .header-wrap {
    flex-direction: column;
    align-items: flex-start;
    padding: 10px 2px 16px;
  }
  .app-title { font-size: 1.55rem; }
  .app-subtitle { font-size: 0.88rem; }
  .tool-badges { justify-content: flex-start; gap: 8px; }
  .badge { font-size: 0.76rem; padding: 7px 11px; }
  .chat-card { border-radius: 20px !important; min-height: 640px; }
  .input-area { padding: 14px 14px 10px !important; }
  .bottom-bar { padding: 0 14px 14px !important; }
  .send-btn { min-width: 92px !important; padding: 14px 14px !important; }
  .examples-section td { font-size: 0.88rem !important; }
  .cap-text strong { font-size: 0.92rem; }
  .cap-text span { font-size: 0.84rem; }
}
"""

# =====================================================
# GRADIO UI
# =====================================================

EXAMPLE_PROMPTS = [
    "What are the latest news about AI today?",
    "What time is it in Tokyo right now?",
    "Generate an image of a lush forest waterfall at sunrise",
    "Search the web and summarize: what is the current price of Bitcoin?",
    "What happened in the world this week?",
    "Create an image of a fox sitting in a peaceful meadow",
]

with gr.Blocks(title="🌿 Nature AI Agent") as demo:
    gr.HTML(f"<style>{CSS}</style>")

    with gr.Column(elem_classes="page-shell"):
        gr.HTML(
            """
            <div class="header-wrap">
                <div class="header-left">
                    <div class="app-logo">🌿</div>
                    <div class="header-text">
                        <div class="app-title">Nature AI Agent</div>
                        <div class="app-subtitle">A calm, modern assistant for search, images, web browsing, and reasoning</div>
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
            """
        )

        with gr.Row(elem_classes="content-grid"):
            # LEFT: CHAT
            with gr.Column(elem_classes="chat-card"):
                chatbot = gr.Chatbot(
                    value=[],
                    height=640,
                    show_label=False,
                    elem_classes="chatbot-box",
                    avatar_images=(
                        None,
                        "https://huggingface.co/front/assets/huggingface_logo-noborder.svg",
                    ),
                    type="messages",
                )
                gr.HTML('<div class="chat-divider"></div>')
                with gr.Group(elem_classes="input-area"):
                    with gr.Row(elem_classes="composer-row"):
                        user_input = gr.Textbox(
                            placeholder="Ask anything — search the web, generate nature scenes, check timezones...",
                            show_label=False,
                            lines=1,
                            max_lines=4,
                            scale=7,
                            elem_classes="input-box",
                        )
                        send_btn = gr.Button("Send ➜", scale=0, elem_classes="send-btn")
                with gr.Row(elem_classes="bottom-bar"):
                    clear_btn = gr.Button("Clear chat", elem_classes="clear-btn")

            # RIGHT: SIDEBAR
            with gr.Column(elem_classes="sidebar"):
                with gr.Group(elem_classes="sidebar-card"):
                    gr.HTML('<div class="panel-title"><span>✦</span> Try an example</div>')
                    with gr.Group(elem_classes="examples-section"):
                        gr.Examples(
                            examples=EXAMPLE_PROMPTS,
                            inputs=user_input,
                            label="",
                        )

                with gr.Group(elem_classes="info-card"):
                    gr.HTML('<div class="panel-title"><span>✦</span> Capabilities</div>')
                    gr.HTML(
                        """
                        <div class="capability-list">
                            <div class="capability-item">
                                <div class="cap-icon">🔍</div>
                                <div class="cap-text"><strong>Web Search</strong><span>Real-time DuckDuckGo results for current topics</span></div>
                            </div>
                            <div class="capability-item">
                                <div class="cap-icon">🎨</div>
                                <div class="cap-text"><strong>Image Generation</strong><span>Create scenic visuals using FLUX on HuggingFace</span></div>
                            </div>
                            <div class="capability-item">
                                <div class="cap-icon">🌐</div>
                                <div class="cap-text"><strong>Web Browsing</strong><span>Visit, read, and summarize webpages</span></div>
                            </div>
                            <div class="capability-item">
                                <div class="cap-icon">🕐</div>
                                <div class="cap-text"><strong>Timezone Tool</strong><span>Check the current time anywhere in the world</span></div>
                            </div>
                            <div class="capability-item">
                                <div class="cap-icon">⚡</div>
                                <div class="cap-text"><strong>Code Execution</strong><span>Reason with Python for fast computations</span></div>
                            </div>
                        </div>
                        """
                    )

        gr.HTML('<div class="footer">Built with 🤗 smolagents · Gradio · GPT-4o · FLUX</div>')

    send_btn.click(fn=run_agent, inputs=[user_input, chatbot], outputs=[chatbot, user_input])
    user_input.submit(fn=run_agent, inputs=[user_input, chatbot], outputs=[chatbot, user_input])
    clear_btn.click(fn=lambda: ([], ""), outputs=[chatbot, user_input])

# =====================================================
# LAUNCH
# =====================================================

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7861)),
    theme=gr.themes.Default(font=gr.themes.GoogleFont("Inter")),
)