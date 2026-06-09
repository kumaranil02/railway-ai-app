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

    history = history + [{"role": "user", "content": user_message}]
    history = history + [{"role": "assistant", "content": "⏳ Thinking..."}]
    yield history, ""

    try:
        response = agent.run(user_message)
        if isinstance(response, AgentImage):
            pil_img = response.to_raw()
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            pil_img.save(tmp.name)
            history[-1] = {
                "role": "assistant",
                "content": {"path": tmp.name, "mime_type": "image/png"},
            }
        else:
            history[-1] = {"role": "assistant", "content": str(response)}
    except Exception as e:
        history[-1] = {"role": "assistant", "content": f"⚠️ Error: {str(e)}"}

    yield history, ""


# =====================================================
# CUSTOM CSS - NATURE THEME
# =====================================================

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root {
  --bg-0: #04110b;
  --bg-1: #071a12;
  --bg-2: #0b2418;
  --card: rgba(10, 24, 18, 0.74);
  --card-soft: rgba(16, 34, 25, 0.86);
  --line: rgba(165, 243, 188, 0.12);
  --line-strong: rgba(165, 243, 188, 0.20);
  --text: #f5fff8;
  --text-soft: rgba(233, 255, 239, 0.78);
  --text-muted: rgba(226, 255, 235, 0.58);
  --accent: #5ee38a;
  --accent-2: #35c98c;
  --accent-3: #76b7ff;
  --shadow: 0 28px 80px rgba(0, 0, 0, 0.52);
}

* { box-sizing: border-box; }
html, body { min-height: 100%; }
body {
  margin: 0;
  font-family: 'Inter', 'Segoe UI', sans-serif !important;
  color: var(--text);
  background:
    radial-gradient(circle at 15% 15%, rgba(94, 227, 138, 0.16), transparent 24%),
    radial-gradient(circle at 85% 10%, rgba(54, 193, 140, 0.12), transparent 20%),
    radial-gradient(circle at 75% 92%, rgba(118, 183, 255, 0.10), transparent 22%),
    linear-gradient(160deg, #030d08 0%, #07150f 42%, #06110c 100%) !important;
}

.gradio-container {
  max-width: 100% !important;
  min-height: 100vh !important;
  padding: 0 !important;
  background: transparent !important;
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
  border-bottom: 1px solid rgba(255, 255, 255, 0.06);
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
  background: linear-gradient(135deg, #1fa96e 0%, #5ee38a 45%, #8ce8bd 100%);
  box-shadow: 0 10px 24px rgba(94, 227, 138, 0.26), inset 0 1px 0 rgba(255, 255, 255, 0.28);
  color: #052012;
  flex-shrink: 0;
}

.header-text { display: flex; flex-direction: column; gap: 4px; }
.app-title {
  font-size: 2rem;
  line-height: 1;
  font-weight: 900;
  letter-spacing: -0.04em;
  color: var(--text);
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
  border: 1px solid rgba(94, 227, 138, 0.20);
  background: rgba(6, 20, 14, 0.55);
  color: rgba(245, 255, 248, 0.90);
  font-size: 0.84rem;
  font-weight: 600;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
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
  border: 1px solid var(--line) !important;
  border-radius: 26px !important;
  overflow: hidden !important;
  box-shadow: var(--shadow), inset 0 1px 0 rgba(255,255,255,0.05) !important;
  backdrop-filter: blur(20px) saturate(1.05) !important;
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
  background: linear-gradient(135deg, #1e9d69 0%, #32c688 45%, #58d9a0 100%) !important;
  color: #052012 !important;
  border-radius: 18px 18px 6px 18px !important;
  font-weight: 600 !important;
  box-shadow: 0 10px 18px rgba(94, 227, 138, 0.14) !important;
}

.chatbot-box .message.bot {
  background: rgba(255, 255, 255, 0.055) !important;
  color: var(--text) !important;
  border: 1px solid rgba(255, 255, 255, 0.07) !important;
  border-radius: 18px 18px 18px 6px !important;
}

.chat-divider {
  height: 1px;
  margin: 0 20px;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
}

.input-area {
  padding: 16px 20px 12px !important;
  background: transparent !important;
}

.input-box textarea {
  min-height: 58px !important;
  background: rgba(255, 255, 255, 0.07) !important;
  border: 1px solid rgba(255, 255, 255, 0.12) !important;
  border-radius: 16px !important;
  color: var(--text) !important;
  font-size: 1rem !important;
  font-weight: 500 !important;
  padding: 16px 16px !important;
  resize: none !important;
  line-height: 1.5 !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.04) !important;
}

.input-box textarea:focus {
  outline: none !important;
  border-color: rgba(94, 227, 138, 0.45) !important;
  box-shadow: 0 0 0 3px rgba(94, 227, 138, 0.10) !important;
}

.input-box textarea::placeholder {
  color: rgba(240, 255, 245, 0.42) !important;
}

.send-btn {
  min-width: 118px !important;
  padding: 15px 18px !important;
  border: none !important;
  border-radius: 16px !important;
  background: linear-gradient(135deg, #1fa96e 0%, #58d9a0 100%) !important;
  color: #052012 !important;
  font-weight: 800 !important;
  font-size: 0.96rem !important;
  box-shadow: 0 14px 24px rgba(94, 227, 138, 0.20) !important;
  transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease !important;
}

.send-btn:hover {
  transform: translateY(-1px) !important;
  filter: brightness(1.03) !important;
  box-shadow: 0 18px 30px rgba(94, 227, 138, 0.28) !important;
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
  border: 1px solid rgba(255,255,255,0.10) !important;
  color: rgba(245, 255, 248, 0.72) !important;
  border-radius: 12px !important;
  padding: 8px 14px !important;
  font-size: 0.84rem !important;
  transition: all 0.15s ease !important;
}

.clear-btn:hover {
  background: rgba(255,255,255,0.06) !important;
  color: var(--text) !important;
  border-color: rgba(94, 227, 138, 0.18) !important;
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
  background: var(--card-soft) !important;
  border: 1px solid var(--line) !important;
  border-radius: 22px !important;
  box-shadow: var(--shadow), inset 0 1px 0 rgba(255,255,255,0.04) !important;
  overflow: hidden !important;
}

.panel-title {
  padding: 16px 18px 12px;
  border-bottom: 1px solid rgba(255,255,255,0.05);
  font-size: 0.84rem;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: rgba(245,255,248,0.82);
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
  border: 1px solid rgba(94, 227, 138, 0.14) !important;
  background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03)) !important;
  color: var(--text) !important;
  font-size: 0.93rem !important;
  font-weight: 600 !important;
  line-height: 1.45 !important;
  white-space: normal !important;
  cursor: pointer !important;
  transition: transform 0.15s ease, border-color 0.15s ease, background 0.15s ease !important;
}

.examples-section .examples td:hover {
  transform: translateY(-1px) !important;
  border-color: rgba(94, 227, 138, 0.32) !important;
  background: linear-gradient(180deg, rgba(94,227,138,0.12), rgba(255,255,255,0.05)) !important;
}

.capability-list { padding: 6px 0 4px; }

.capability-item {
  display: flex;
  gap: 12px;
  align-items: flex-start;
  padding: 14px 18px;
  border-top: 1px solid rgba(255,255,255,0.05);
}

.capability-item:first-child { border-top: none; }

.cap-icon {
  width: 34px;
  height: 34px;
  border-radius: 12px;
  display: grid;
  place-items: center;
  background: rgba(94, 227, 138, 0.12);
  border: 1px solid rgba(94, 227, 138, 0.12);
  flex-shrink: 0;
}

.cap-text strong {
  display: block;
  font-size: 0.98rem;
  color: var(--text);
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
  color: rgba(245,255,248,0.28);
  padding: 22px 0 8px;
}

/* SCROLLBAR */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(94, 227, 138, 0.28); border-radius: 999px; }

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
                    with gr.Row():
                        user_input = gr.Textbox(
                            placeholder="Ask anything — search the web, generate nature scenes, check timezones...",
                            show_label=False,
                            lines=1,
                            max_lines=4,
                            scale=7,
                            elem_classes="input-box",
                        )
                        send_btn = gr.Button("Send ➤", scale=1, elem_classes="send-btn")
                with gr.Row(elem_classes="bottom-bar"):
                    clear_btn = gr.Button("🗑 Clear chat", elem_classes="clear-btn")

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
)