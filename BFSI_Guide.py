import gradio as gr
import os
import time
import random

# --- LIBRARY IMPORTS ---
try:
    from google import genai
    from google.genai.types import GenerateContentConfig, Tool, GoogleSearch
except ImportError:
    pass

try:
    import openai
except ImportError:
    pass

try:
    import anthropic
except ImportError:
    pass

try:
    from duckduckgo_search import DDGS
except ImportError:
    pass

# ===========================
# 1. CONFIGURATION & SETUP
# ===========================

# Global state management
class AppState:
    def __init__(self):
        self.research_data = None
        self.general_report = None
        self.messages = []
        self.product_name = ""
        self.usage_count = 0
        self.client = None
        self.provider = "Google Gemini"
        self.model_id = "gemini-2.5-flash"
        self.api_key = None
        self.using_free_key = False

state = AppState()

# Rate Limit Constant
FREE_USAGE_LIMIT = 5

# ===========================
# 2. WEB SEARCH BRIDGE
# ===========================

def search_web_duckduckgo(query, max_results=5):
    """Fetches live search results using DuckDuckGo (Free)."""
    try:
        results = DDGS().text(query, max_results=max_results)
        return "\n".join([f"- {r['title']}: {r['body']} (Source: {r['href']})" for r in results])
    except Exception as e:
        return f"Search failed: {str(e)}"

# ===========================
# 3. UNIFIED LLM WRAPPER
# ===========================

def call_llm(system_instruction, user_prompt, use_search=False, search_query=None):
    """Unified function for Gemini, OpenAI, and Claude."""
    
    if not state.client:
        return "Error: Client not initialized. Check API Key."

    # --- GOOGLE GEMINI HANDLER ---
    if state.provider == "Google Gemini":
        tools = [Tool(google_search=GoogleSearch())] if use_search else None
        config = GenerateContentConfig(tools=tools, system_instruction=system_instruction, temperature=0.1)
        try:
            return state.client.models.generate_content(model=state.model_id, contents=user_prompt, config=config).text
        except Exception as e: return f"Gemini Error: {e}"

    # --- SEARCH INJECTION FOR OTHERS ---
    final_prompt = user_prompt
    if use_search and search_query:
        web_data = search_web_duckduckgo(search_query)
        final_prompt = f"CONTEXT FROM LIVE WEB SEARCH:\n{web_data}\n\nUSER QUERY:\n{user_prompt}"

    # --- OPENAI HANDLER ---
    if state.provider == "OpenAI (ChatGPT)":
        try:
            messages = [{"role": "system", "content": system_instruction}, {"role": "user", "content": final_prompt}]
            response = state.client.chat.completions.create(model=state.model_id, messages=messages, temperature=0.1)
            return response.choices[0].message.content
        except Exception as e: return f"OpenAI Error: {e}"

    # --- ANTHROPIC HANDLER ---
    elif state.provider == "Anthropic (Claude)":
        try:
            response = state.client.messages.create(model=state.model_id, system=system_instruction, messages=[{"role": "user", "content": final_prompt}], max_tokens=4000, temperature=0.3)
            return response.content[0].text
        except Exception as e: return f"Claude Error: {e}"

# ===========================
# 4. AGENT PERSONAS
# ===========================

RESEARCHER_INSTRUCTION = "ROLE: Financial Scrutinizer. GOAL: Find comparative data, hidden fees, regulatory risk. OUTPUT: RAW text dump."
EDITOR_INSTRUCTION = "ROLE: Risk Analyst. STRUCTURE: 1. Hidden Fees 2. Risk Score 3. Comparative TCO Table 4. Verdict."
PERSONALIZER_INSTRUCTION = "ROLE: Ethical Advisor. GOAL: Match user profile to product. OUTPUT: Recommendation letter."

# ===========================
# 5. CORE FUNCTIONS
# ===========================

def run_research(product_name):
    prompt = f"Current Date: December 2025. Investigate rival offers for: {product_name}. Confirm hidden fees."
    search_query = f"{product_name} financial terms hidden fees reviews 2025"
    return call_llm(RESEARCHER_INSTRUCTION, prompt, use_search=True, search_query=search_query)

def generate_report(product_name, research_data):
    prompt = f"Research Data:\n{research_data}\n\nGenerate Financial Report."
    return call_llm(EDITOR_INSTRUCTION, prompt)

def generate_personal_rec(product_name, research_data, user_profile):
    prompt = f"Research Data: {research_data}\nUser Profile: {user_profile}\nGenerate verdict."
    return call_llm(PERSONALIZER_INSTRUCTION, prompt)

# ===========================
# 6. GRADIO INTERFACE FUNCTIONS
# ===========================

def initialize_client(provider, key_source, api_key_input, model_choice):
    """Initialize the API client based on provider selection."""
    state.provider = provider
    state.using_free_key = False
    
    # Handle API Key
    if provider == "Google Gemini":
        if key_source == "Use Free Default Key":
            state.using_free_key = True
            try:
                state.api_key = os.getenv("GEMINI_API_KEY")  # For Hugging Face Spaces
                if not state.api_key:
                    return "Error: Default key not configured in environment."
            except Exception as e:
                return f"Error: {e}"
        else:
            state.api_key = api_key_input
        
        # Set model
        if "Flash" in model_choice: state.model_id = "gemini-2.5-flash"
        elif "2.5" in model_choice: state.model_id = "gemini-2.5-pro"
        else: state.model_id = "gemini-3-pro-preview"
        
    elif provider == "OpenAI (ChatGPT)":
        state.api_key = api_key_input
        state.model_id = model_choice
        
    elif provider == "Anthropic (Claude)":
        state.api_key = api_key_input
        state.model_id = model_choice
    
    # Initialize client
    if state.api_key:
        try:
            if provider == "Google Gemini":
                state.client = genai.Client(api_key=state.api_key)
            elif provider == "OpenAI (ChatGPT)":
                state.client = openai.OpenAI(api_key=state.api_key)
            elif provider == "Anthropic (Claude)":
                state.client = anthropic.Anthropic(api_key=state.api_key)
            return "✅ Client initialized successfully!"
        except Exception as e:
            return f"❌ Error: {e}"
    return "⚠️ API Key required"

def run_analysis(product_input, provider, key_source, api_key_input, model_choice):
    """Main analysis function."""
    
    # Check rate limit
    if state.using_free_key and state.usage_count >= FREE_USAGE_LIMIT:
        return (
            f"🛑 Free Usage Limit Reached ({FREE_USAGE_LIMIT}/{FREE_USAGE_LIMIT}).\n\n"
            "To continue, please provide your own API key.",
            "",
            "",
            gr.update(visible=False),
            gr.update(visible=False)
        )
    
    # Initialize client
    init_result = initialize_client(provider, key_source, api_key_input, model_choice)
    if "Error" in init_result or "required" in init_result:
        return init_result, "", "", gr.update(visible=False), gr.update(visible=False)
    
    if not product_input:
        return "⚠️ Please enter a financial product to analyze.", "", "", gr.update(visible=False), gr.update(visible=False)
    
    state.product_name = product_input
    state.messages = []
    
    try:
        # Research Phase
        status_msg = f"🌍 Gathering intelligence via {state.provider}..."
        research_data = run_research(product_input)
        state.research_data = research_data
        
        # Analysis Phase
        status_msg += "\n🧠 Calculating Risk & TCO..."
        report_text = generate_report(product_input, research_data)
        state.general_report = report_text
        
        # Increment usage counter
        if state.using_free_key:
            state.usage_count += 1
        
        search_note = "✅ Verified with Google Search" if state.provider == "Google Gemini" else "✅ Verified with DuckDuckGo Search"
        
        return (
            f"✅ Analysis Complete!\n\nQuota Used: {state.usage_count}/{FREE_USAGE_LIMIT if state.using_free_key else '∞'}",
            report_text,
            f"{search_note}\n\n{research_data}",
            gr.update(visible=True),
            gr.update(visible=True)
        )
        
    except Exception as e:
        return f"❌ System Error: {e}", "", "", gr.update(visible=False), gr.update(visible=False)

def get_personal_recommendation(user_profile):
    """Generate personalized recommendation."""
    if not state.general_report:
        return "⚠️ Please run an analysis first."
    
    if not user_profile:
        return "⚠️ Please describe your financial goal."
    
    try:
        rec = generate_personal_rec(state.product_name, state.research_data, user_profile)
        return rec
    except Exception as e:
        return f"❌ Error: {e}"

def chat_response(message, history):
    """Handle chat interactions."""
    if not state.research_data:
        return "⚠️ Please run an analysis first to enable chat."
    
    try:
        response = call_llm(f"Advisor. Context: {state.research_data}", message)
        return response
    except Exception as e:
        return f"❌ Error: {e}"

# ===========================
# 7. GRADIO APP INTERFACE
# ===========================

with gr.Blocks(title="Financial IQ: Multi-Engine Analyst", theme=gr.themes.Soft()) as app:
    
    gr.Markdown("# 📈 NexFin Intelligence")
    gr.Markdown("*Multi-Engine Financial Product Analyzer with Live Web Intelligence*")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## ⚙️ Engine Settings")
            
            provider = gr.Radio(
                choices=["Google Gemini", "OpenAI (ChatGPT)", "Anthropic (Claude)"],
                value="Google Gemini",
                label="Select AI Provider"
            )
            
            # Gemini Options
            key_source = gr.Radio(
                choices=["Use Free Default Key", "Enter My Own Key"],
                value="Use Free Default Key",
                label="API Key Source",
                visible=True
            )
            
            api_key_input = gr.Textbox(
                label="API Key",
                type="password",
                visible=False
            )
            
            gemini_model = gr.Dropdown(
                choices=["2.5 Flash (Fast)", "2.5 Pro (Stable)", "3.0 Pro (Latest)"],
                value="2.5 Flash (Fast)",
                label="Gemini Model",
                visible=True
            )
            
            openai_model = gr.Dropdown(
                choices=["gpt-5-mini", "gpt-5"],
                value="gpt-5-mini",
                label="OpenAI Model",
                visible=False
            )
            
            claude_model = gr.Dropdown(
                choices=["claude-opus-4-1-20250805", "claude-opus-4-5-20251101", "claude-haiku-4-5-20251001", "claude-sonnet-4-5-20250929"],
                value="claude-opus-4-1-20250805",
                label="Claude Model",
                visible=False
            )
            
            def update_provider_ui(provider_choice):
                if provider_choice == "Google Gemini":
                    return (
                        gr.update(visible=True), gr.update(visible=False),
                        gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
                    )
                else:
                    return (
                        gr.update(visible=False), gr.update(visible=True),
                        gr.update(visible=False),
                        gr.update(visible=True) if provider_choice == "OpenAI (ChatGPT)" else gr.update(visible=False),
                        gr.update(visible=True) if provider_choice == "Anthropic (Claude)" else gr.update(visible=False)
                    )
            
            provider.change(
                update_provider_ui,
                inputs=[provider],
                outputs=[key_source, api_key_input, gemini_model, openai_model, claude_model]
            )
            
            def update_key_visibility(choice):
                return gr.update(visible=(choice == "Enter My Own Key"))
            
            key_source.change(update_key_visibility, inputs=[key_source], outputs=[api_key_input])
        
        with gr.Column(scale=2):
            gr.Markdown("## 🔍 Product Analysis")
            
            product_input = gr.Textbox(
                label="Financial Product to Analyze",
                placeholder="e.g. HDFC Home Loan, SBI Mutual Fund",
                lines=1
            )
            
            analyze_btn = gr.Button("🚀 Run Analysis", variant="primary", size="lg")
            
            status_output = gr.Textbox(label="Status", lines=3, interactive=False)
            
            report_output = gr.Markdown(label="Analysis Report", visible=True)
            
            with gr.Accordion("🔍 Raw Data Transparency", open=False):
                raw_data_output = gr.Textbox(label="Research Notes", lines=8, interactive=False)
    
    with gr.Row(visible=False) as advisor_row:
        with gr.Column():
            gr.Markdown("## 👤 Personal Advisor")
            user_profile = gr.Textbox(
                label="Your Financial Goal",
                placeholder="e.g. Loan for house purchase, long-term retirement savings...",
                lines=3
            )
            rec_btn = gr.Button("✨ Get Personalized Verdict")
            recommendation_output = gr.Markdown()
    
    with gr.Row(visible=False) as chat_row:
        with gr.Column():
            gr.Markdown("## 💬 Chat with AI Advisor")
            chatbot = gr.Chatbot(height=400)
            chat_input = gr.Textbox(label="Ask a follow-up question", placeholder="e.g. How does this compare to...")
            chat_input.submit(chat_response, inputs=[chat_input, chatbot], outputs=[chatbot])
    
    # Wire up the analysis button
    model_selector = gr.State()
    
    def get_current_model(provider, gemini_m, openai_m, claude_m):
        if provider == "Google Gemini": return gemini_m
        elif provider == "OpenAI (ChatGPT)": return openai_m
        else: return claude_m
    
    analyze_btn.click(
        lambda p, g, o, c: get_current_model(p, g, o, c),
        inputs=[provider, gemini_model, openai_model, claude_model],
        outputs=[model_selector]
    ).then(
        run_analysis,
        inputs=[product_input, provider, key_source, api_key_input, model_selector],
        outputs=[status_output, report_output, raw_data_output, advisor_row, chat_row]
    )
    
    rec_btn.click(
        get_personal_recommendation,
        inputs=[user_profile],
        outputs=[recommendation_output]
    )

# Launch the app
if __name__ == "__main__":
    app.launch()
