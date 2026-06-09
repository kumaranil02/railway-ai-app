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

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    min-height: 100vh;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
    margin: 0 !important;
    padding: 0 !important;
}

.gradio-container > .main {
    max-width: 780px !important;
    margin: 0 auto !important;
    padding: 16px !important;
}

.header-wrap {
    text-align: center;
    padding: 28px 20px 16px;
}

.app-title {
    font-size: clamp(1.6rem, 5vw, 2.2rem);
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.5px;
    margin: 0 0 6px 0;
    text-shadow: 0 2px 12px rgba(0,0,0,0.15);
}

.app-subtitle {
    color: rgba(255,255,255,0.75);
    font-size: 0.88rem;
    margin: 0 0 14px 0;
    font-weight: 400;
}

.tool-badges {
    display: flex;
    justify-content: center;
    gap: 8px;
    flex-wrap: wrap;
}

.badge {
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    background: rgba(255,255,255,0.18);
    color: #ffffff;
    border: 1px solid rgba(255,255,255,0.3);
    backdrop-filter: blur(4px);
}

.chat-card {
    background: #ffffff !important;
    border-radius: 20px !important;
    box-shadow: 0 20px 60px rgba(0,0,0,0.2) !important;
    overflow: hidden !important;
    border: none !important;
    margin-top: 4px;
}

.chatbot-box {
    background: #f8fafc !important;
    border: none !important;
    border-radius: 0 !important;
    border-bottom: 1px solid #f1f5f9 !important;
}

.chatbot-box .message.user {
    background: linear-gradient(135deg, #6366f1, #3b82f6) !important;
    color: #fff !important;
    border-radius: 18px 18px 4px 18px !important;
    max-width: 80% !important;
    margin-left: auto !important;
}

.chatbot-box .message.bot {
    background: #ffffff !important;
    color: #1e293b !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 18px 18px 18px 4px !important;
    max-width: 85% !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
}

.input-area {
    padding: 14px 16px !important;
    background: #ffffff !important;
}

.input-row {
    display: flex;
    gap: 10px;
    align-items: flex-end;
}

.input-box textarea {
    background: #f8fafc !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 14px !important;
    color: #1e293b !important;
    font-size: 0.93rem !important;
    padding: 12px 16px !important;
    resize: none !important;
    line-height: 1.5 !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}

.input-box textarea:focus {
    border-color: #6366f1 !important;
    background: #ffffff !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
    outline: none !important;
}

.input-box textarea::placeholder { color: #94a3b8 !important; }

.send-btn {
    background: linear-gradient(135deg, #6366f1, #3b82f6) !important;
    border: none !important;
    border-radius: 14px !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    padding: 12px 22px !important;
    box-shadow: 0 4px 12px rgba(99,102,241,0.35) !important;
    transition: all 0.2s !important;
    white-space: nowrap !important;
    min-width: 80px !important;
}

.send-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(99,102,241,0.5) !important;
}

.bottom-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 16px 12px !important;
    background: #ffffff !important;
}

.clear-btn {
    background: transparent !important;
    border: none !important;
    color: #94a3b8 !important;
    font-size: 0.78rem !important;
    padding: 4px 8px !important;
    cursor: pointer !important;
}

.clear-btn:hover { color: #64748b !important; }

.examples-label {
    color: #94a3b8;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 10px 16px 4px;
    background: #ffffff;
}

.examples-wrap {
    background: #ffffff !important;
    padding: 0 12px 14px !important;
    border-radius: 0 0 20px 20px !important;
}

.examples-wrap .examples table { display: flex; flex-wrap: wrap; gap: 8px; border: none !important; }
.examples-wrap .examples table tbody { display: contents; }
.examples-wrap .examples table tr { display: contents; }
.examples-wrap .examples table td {
    display: inline-block !important;
    padding: 6px 14px !important;
    background: #f1f5f9 !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 999px !important;
    font-size: 0.78rem !important;
    color: #475569 !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
    white-space: nowrap !important;
}
.examples-wrap .examples table td:hover {
    background: #ede9fe !important;
    border-color: #c4b5fd !important;
    color: #6366f1 !important;
}

.footer {
    text-align: center;
    color: rgba(255,255,255,0.45);
    font-size: 0.72rem;
    padding: 14px 0 20px;
}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #c7d2fe; border-radius: 99px; }

@media (max-width: 600px) {
    .gradio-container > .main { padding: 8px !important; }
    .header-wrap { padding: 20px 12px 12px; }
    .app-title { font-size: 1.5rem; }
    .tool-badges { gap: 6px; }
    .badge { font-size: 0.68rem; padding: 3px 10px; }
    .send-btn { padding: 12px 14px !important; min-width: 60px !important; }
    .chatbot-box .message.user, .chatbot-box .message.bot { max-width: 95% !important; }
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

    gr.HTML("""
    <div class="header-wrap">
        <h1 class="app-title">✨ AI Agent</h1>
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
            height=420,
            show_label=False,
            elem_classes="chatbot-box",
            avatar_images=(None, "https://huggingface.co/front/assets/huggingface_logo-noborder.svg"),
        )

        with gr.Group(elem_classes="input-area"):
            with gr.Row(elem_classes="input-row"):
                user_input = gr.Textbox(
                    placeholder="Ask me anything — search, generate images, check time...",
                    show_label=False,
                    lines=1,
                    max_lines=3,
                    scale=6,
                    elem_classes="input-box",
                )
                send_btn = gr.Button("Send ➤", scale=1, elem_classes="send-btn")

        with gr.Row(elem_classes="bottom-row"):
            clear_btn = gr.Button("🗑 Clear", elem_classes="clear-btn", scale=1)

        gr.HTML('<div class="examples-label">✦ Try an example</div>')
        with gr.Group(elem_classes="examples-wrap"):
            gr.Examples(examples=EXAMPLE_PROMPTS, inputs=user_input, label="")

    gr.HTML('<div class="footer">Built with 🤗 smolagents &nbsp;·&nbsp; Gradio &nbsp;·&nbsp; GPT-4o &nbsp;·&nbsp; FLUX</div>')

    send_btn.click(fn=run_agent, inputs=[user_input, chatbot], outputs=[chatbot, user_input])
    user_input.submit(fn=run_agent, inputs=[user_input, chatbot], outputs=[chatbot, user_input])
    clear_btn.click(fn=lambda: ([], ""), outputs=[chatbot, user_input])

# =====================================================
# LAUNCH
# =====================================================

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7861)), theme=gr.themes.Default(font=gr.themes.GoogleFont("Inter")))
