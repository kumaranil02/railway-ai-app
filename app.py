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
body, .gradio-container {
    background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #0f3443, #1a1a2e);
    background-size: 400% 400%;
    animation: gradientShift 12s ease infinite;
    min-height: 100vh;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.main-panel {
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 24px !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4) !important;
    padding: 24px !important;
}

.app-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1px;
    margin: 0;
}

.app-subtitle { color: rgba(255,255,255,0.5); font-size: 0.95rem; margin-top: 6px; }

.tool-badges { display: flex; justify-content: center; gap: 10px; flex-wrap: wrap; margin: 14px 0; }

.badge { padding: 5px 14px; border-radius: 999px; font-size: 0.78rem; font-weight: 600; }

.badge-purple { background: rgba(167,139,250,0.15); color: #a78bfa; border: 1px solid rgba(167,139,250,0.3); }
.badge-blue   { background: rgba(96,165,250,0.15);  color: #60a5fa; border: 1px solid rgba(96,165,250,0.3); }
.badge-green  { background: rgba(52,211,153,0.15);  color: #34d399; border: 1px solid rgba(52,211,153,0.3); }
.badge-orange { background: rgba(251,146,60,0.15);  color: #fb923c; border: 1px solid rgba(251,146,60,0.3); }
.badge-pink   { background: rgba(244,114,182,0.15); color: #f472b6; border: 1px solid rgba(244,114,182,0.3); }

.chatbot-box {
    background: rgba(0,0,0,0.25) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 16px !important;
}

.input-box textarea {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 14px !important;
    color: white !important;
    font-size: 0.95rem !important;
    padding: 12px 16px !important;
    resize: none !important;
}

.input-box textarea:focus {
    border-color: rgba(167,139,250,0.5) !important;
    box-shadow: 0 0 0 3px rgba(167,139,250,0.1) !important;
    outline: none !important;
}

.input-box textarea::placeholder { color: rgba(255,255,255,0.3) !important; }

.send-btn {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    border: none !important;
    border-radius: 14px !important;
    color: white !important;
    font-weight: 700 !important;
    padding: 12px 28px !important;
    box-shadow: 0 4px 15px rgba(124,58,237,0.4) !important;
    min-width: 100px !important;
}

.send-btn:hover { transform: translateY(-2px) !important; box-shadow: 0 6px 20px rgba(124,58,237,0.6) !important; }

.clear-btn {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 14px !important;
    color: rgba(255,255,255,0.6) !important;
}

.footer { text-align: center; color: rgba(255,255,255,0.25); font-size: 0.75rem; margin-top: 16px; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: rgba(167,139,250,0.3); border-radius: 99px; }
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
    <div style="text-align:center;padding:20px 0 10px 0">
        <h1 class="app-title">✨ AI Agent</h1>
        <p class="app-subtitle">Powered by GPT-4o · Web Search · Image Generation · HuggingFace</p>
        <div class="tool-badges">
            <span class="badge badge-purple">🔍 Web Search</span>
            <span class="badge badge-blue">🎨 Image Generation</span>
            <span class="badge badge-green">🕐 Timezone</span>
            <span class="badge badge-orange">🌐 Browse Web</span>
            <span class="badge badge-pink">⚡ Code Execution</span>
        </div>
    </div>
    """)

    with gr.Group(elem_classes="main-panel"):

        chatbot = gr.Chatbot(
            value=[],
            height=480,
            show_label=False,
            elem_classes="chatbot-box",
            avatar_images=(None, "https://huggingface.co/front/assets/huggingface_logo-noborder.svg"),
        )

        with gr.Row():
            user_input = gr.Textbox(
                placeholder="Ask anything — search the web, generate images, check timezones...",
                show_label=False,
                lines=1,
                max_lines=4,
                scale=5,
                elem_classes="input-box",
            )
            send_btn = gr.Button("Send ➤", scale=1, elem_classes="send-btn")

        with gr.Row():
            clear_btn = gr.Button("🗑 Clear conversation", elem_classes="clear-btn")

        gr.HTML('<p style="color:rgba(255,255,255,0.4);font-size:0.78rem;text-transform:uppercase;letter-spacing:1px;margin:16px 0 8px 2px">✦ Try an example</p>')
        gr.Examples(examples=EXAMPLE_PROMPTS, inputs=user_input, label="")

    gr.HTML('<div class="footer">Built with 🤗 smolagents · Gradio · GPT-4o · FLUX</div>')

    send_btn.click(fn=run_agent, inputs=[user_input, chatbot], outputs=[chatbot, user_input])
    user_input.submit(fn=run_agent, inputs=[user_input, chatbot], outputs=[chatbot, user_input])
    clear_btn.click(fn=lambda: ([], ""), outputs=[chatbot, user_input])

# =====================================================
# LAUNCH
# =====================================================

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7861)))
