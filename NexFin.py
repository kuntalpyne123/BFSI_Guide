import streamlit as st
import os
import time
import random
from streamlit.errors import StreamlitAPIException

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

st.set_page_config(page_title="Financial IQ: Multi-Engine Analyst", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .badge {
        display: inline-block;
        padding: 0.25em 0.6em;
        font-size: 0.85em;
        font-weight: 700;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.25rem;
        margin-right: 5px;
        margin-bottom: 5px;
    }
    .report-box { border: 1px solid #ddd; padding: 20px; border-radius: 10px; background-color: #f9f9f9; }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if "research_data" not in st.session_state: st.session_state.research_data = None
if "general_report" not in st.session_state: st.session_state.general_report = None
if "messages" not in st.session_state: st.session_state.messages = []
if "product_name" not in st.session_state: st.session_state.product_name = ""
if "usage_count" not in st.session_state: st.session_state.usage_count = 0
if "client" not in st.session_state: st.session_state.client = None

# Rate Limit Constant
FREE_USAGE_LIMIT = 5 

# ===========================
# 2. SIDEBAR CONFIGURATION
# ===========================

with st.sidebar:
    st.header("⚙️ Engine Settings")

    # --- A. PROVIDER SELECTION ---
    provider = st.radio(
        "Select AI Provider:",
        ("Google Gemini", "OpenAI (ChatGPT)", "Anthropic (Claude)"),
        index=0
    )

    api_key = None
    model_id = None
    using_free_key = False 
    
    # --- B. KEY MANAGEMENT ---
    if provider == "Google Gemini":
        st.info("⚡ Native Search Grounding (Most Accurate)")
        
        key_source = st.radio(
            "API Key Source:", 
            ("Use Free Default Key", "Enter My Own Key"),
            help="Default key is limited to 5 requests per session."
        )

        if key_source == "Use Free Default Key":
            using_free_key = True 
            usage_left = FREE_USAGE_LIMIT - st.session_state.usage_count
            st.progress(min(st.session_state.usage_count / FREE_USAGE_LIMIT, 1.0), 
                        text=f"Free Quota: {st.session_state.usage_count}/{FREE_USAGE_LIMIT} used")
            
            if usage_left <= 0:
                st.error("🚫 Session Quota Exceeded. Please enter your own API Key.")
            
            try:
                if "GEMINI_API_KEY" in st.secrets:
                    api_key = st.secrets["GEMINI_API_KEY"]
                else:
                    st.error("🚨 Default key not found in secrets!")
            except StreamlitAPIException:
                st.error("Secrets not available locally.")
        else:
            api_key = st.text_input("Enter Gemini API Key", type="password")
        
        model_choice = st.selectbox(
            "Select Gemini Model:",
            ("2.5 Flash", "2.5 Pro", "3 Flash", "3 Pro")
        )
        if "2.5 Flash" in model_choice: model_id = "gemini-2.5-flash"
        elif "2.5 Pro" in model_choice: model_id = "gemini-2.5-pro"
        elif "3 Flash" in model_choice: model_id = "gemini-3-flash-preview" 
        else: model_id = "gemini-3-pro-preview"

    elif provider == "OpenAI (ChatGPT)":
        st.info("🌐 Web Search enabled via DuckDuckGo")
        api_key = st.text_input("Enter OpenAI API Key", type="password")
        model_id = st.selectbox("Select Model:", ("gpt-4-turbo","gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"))

    elif provider == "Anthropic (Claude)":
        st.info("🌐 Web Search enabled via DuckDuckGo")
        api_key = st.text_input("Enter Anthropic API Key", type="password")
        
        anthropic_models = {
            "Sonnet 3.5": "claude-3-5-sonnet-20240620",
            "Haiku 3": "claude-3-haiku-20240307",
            "Opus 3": "claude-3-opus-20240229"
        }
        selected_display_name = st.selectbox("Select Model:", list(anthropic_models.keys()))
        model_id = anthropic_models[selected_display_name]

    # --- C. INITIALIZATION ---
    if api_key:
        api_key = api_key.strip() 
        if provider == "Google Gemini":
            try: st.session_state.client = genai.Client(api_key=api_key)
            except Exception as e: st.error(f"Gemini Error: {e}")
        elif provider == "OpenAI (ChatGPT)":
            try: st.session_state.client = openai.OpenAI(api_key=api_key)
            except Exception as e: st.error(f"OpenAI Error: {e}")
        elif provider == "Anthropic (Claude)":
            try: st.session_state.client = anthropic.Anthropic(api_key=api_key)
            except Exception as e: st.error(f"Anthropic Error: {e}")

# ===========================
# 3. WEB SEARCH BRIDGE
# ===========================

def search_web_duckduckgo(query, max_results=5):
    try:
        results = DDGS().text(query, max_results=max_results)
        return "\n".join([f"- {r['title']}: {r['body']} (Source: {r['href']})" for r in results])
    except Exception as e:
        return f"Search failed: {str(e)}"

# ===========================
# 4. UNIFIED LLM WRAPPER
# ===========================

def call_llm(system_instruction, user_prompt, use_search=False, search_query=None):
    client = st.session_state.get("client")
    if not client: return "Error: Client not initialized. Check API Key."

    if provider == "Google Gemini":
        tools = [Tool(google_search=GoogleSearch())] if use_search else None
        config = GenerateContentConfig(tools=tools, system_instruction=system_instruction, temperature=0.1)
        try:
            return client.models.generate_content(model=model_id, contents=user_prompt, config=config).text
        except Exception as e: return f"Gemini Error: {e}"

    final_prompt = user_prompt
    if use_search and search_query:
        with st.spinner(f"🕵️ Bridging to live web via DuckDuckGo for {provider}..."):
            web_data = search_web_duckduckgo(search_query)
            final_prompt = f"CONTEXT FROM LIVE WEB SEARCH:\n{web_data}\n\nUSER QUERY:\n{user_prompt}"

    if provider == "OpenAI (ChatGPT)":
        try:
            messages = [{"role": "system", "content": system_instruction}, {"role": "user", "content": final_prompt}]
            response = client.chat.completions.create(model=model_id, messages=messages, temperature=0.1)
            return response.choices[0].message.content
        except Exception as e: return f"OpenAI Error: {e}"

    elif provider == "Anthropic (Claude)":
        try:
            response = client.messages.create(model=model_id, system=system_instruction, messages=[{"role": "user", "content": final_prompt}], max_tokens=4000, temperature=0.3)
            return response.content[0].text
        except Exception as e: return f"Claude Error: {e}"

# ===========================
# 5. AGENT PERSONAS
# ===========================

RESEARCHER_INSTRUCTION = "ROLE: Financial Scrutinizer. GOAL: Find comparative data, hidden fees, regulatory risk. OUTPUT: RAW text dump."
EDITOR_INSTRUCTION = "ROLE: Risk Analyst. STRUCTURE: 1. Hidden Fees 2. Risk Score 3. Comparative TCO Table 4. Verdict."
PERSONALIZER_INSTRUCTION = "ROLE: Ethical Advisor. GOAL: Match user profile to product. OUTPUT: Recommendation letter."

# ===========================
# 6. APP LOGIC
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
# 7. APP INTERFACE
# ===========================

st.title("📈 NexFin Intelligence")
st.caption(f"Powered by **{provider} ({model_id})**")

with st.form("research_form"):
    product_input = st.text_input("Analyze Financial Product:", placeholder="e.g. HDFC Home Loan, SBI Mutual Fund")
    submitted = st.form_submit_button("🚀 Run Analysis")

if submitted and product_input:
    if using_free_key and st.session_state.usage_count >= FREE_USAGE_LIMIT:
        st.error(f"🛑 Free Usage Limit Reached ({FREE_USAGE_LIMIT}/{FREE_USAGE_LIMIT}).")
        st.stop() 
    
    if not api_key:
        st.error("🔑 API Key missing. Please configure settings in the sidebar.")
        st.stop()

    st.session_state.product_name = product_input
    st.session_state.messages = [] 
    st.session_state.general_report = None 
    
    status = st.status(f"🕵️ Initiating Scrutiny via {provider}...", expanded=True)
    
    try:
        status.write(f"🌍 **The Hunter:** Gathering live intelligence...")
        research_data = run_research(product_input)
        st.session_state.research_data = research_data
        
        status.write("🧠 **The Analyst:** Calculating Risk & TCO...")
        report_text = generate_report(product_input, research_data)
        st.session_state.general_report = report_text
        
        if using_free_key:
            st.session_state.usage_count += 1
            st.toast(f"Free Quota Used: {st.session_state.usage_count}/{FREE_USAGE_LIMIT}")
        
        status.update(label="✅ Complete!", state="complete", expanded=False)
        
    except Exception as e:
        status.update(label="❌ Error", state="error")
        st.error(f"System Error: {e}")

if st.session_state.general_report:
    st.divider()
    st.markdown(st.session_state.general_report)
    
    with st.expander("🔍 Raw Data Transparency"):
        if provider == "Google Gemini": st.info("✅ Verified with Google Search")
        else: st.info("✅ Verified with DuckDuckGo Search")
        st.text_area("Raw Notes", st.session_state.research_data, height=200)

    st.divider()
    st.markdown("## 👤 Advisor")
    with st.container(border=True):
        user_profile = st.text_area("Financial Goal:", placeholder="e.g. Loan for house...")
        if st.button("✨ Get Verdict"):
            if user_profile:
                with st.spinner("Simulating..."):
                    rec = generate_personal_rec(st.session_state.product_name, st.session_state.research_data, user_profile)
                    st.markdown(rec)

    st.divider()
    st.markdown(f"## 💬 Chat with {provider}")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Follow-up question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                resp = call_llm(f"Advisor. Context: {st.session_state.research_data}", prompt)
                st.markdown(resp)
        st.session_state.messages.append({"role": "assistant", "content": resp})
