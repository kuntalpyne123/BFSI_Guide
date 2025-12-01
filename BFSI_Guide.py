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
    
    # --- B. KEY MANAGEMENT ---
    
    # 1. GOOGLE GEMINI CONFIG
    if provider == "Google Gemini":
        st.info("⚡ Native Search Grounding (Most Accurate)")
        
        key_source = st.radio("API Key Source:", ("Use Free Default Key", "Enter My Own Key"))
        if key_source == "Use Free Default Key":
            try:
                if "GEMINI_API_KEY" in st.secrets: api_key = st.secrets["GEMINI_API_KEY"]
            except: pass
        else:
            api_key = st.text_input("Enter Gemini API Key", type="password")
        
        model_choice = st.selectbox("Select Model:", ("Flash 2.5 (Fast)", "Pro 2.5 (Stable)"))
        model_id = "gemini-2.5-flash" if "Flash" in model_choice else "gemini-2.5-pro"

    # 2. OPENAI CONFIG
    elif provider == "OpenAI (ChatGPT)":
        st.info("🌐 Web Search enabled via DuckDuckGo")
        api_key = st.text_input("Enter OpenAI API Key", type="password")
        model_id = st.selectbox("Select Model:", ("gpt-5-mini", "gpt-5"))

    # 3. ANTHROPIC CONFIG
    elif provider == "Anthropic (Claude)":
        st.info("🌐 Web Search enabled via DuckDuckGo")
        api_key = st.text_input("Enter Anthropic API Key", type="password")
        model_id = st.selectbox("Select Model:", ("claude-3-5-sonnet-20241022", "claude-3-opus-20240229"))

    # --- C. INITIALIZATION ---
    if not api_key:
        st.warning(f"Please configure {provider} API Key to start.")
        st.stop()
    
    # Initialize Clients
    client = None
    if provider == "Google Gemini":
        try: client = genai.Client(api_key=api_key)
        except Exception as e: st.error(f"Gemini Error: {e}")
    elif provider == "OpenAI (ChatGPT)":
        try: client = openai.OpenAI(api_key=api_key)
        except Exception as e: st.error(f"OpenAI Error: {e}")
    elif provider == "Anthropic (Claude)":
        try: client = anthropic.Anthropic(api_key=api_key)
        except Exception as e: st.error(f"Anthropic Error: {e}")

# ===========================
# 3. WEB SEARCH BRIDGE (The Fix)
# ===========================

def search_web_duckduckgo(query, max_results=5):
    """Fetches live search results using DuckDuckGo (Free)."""
    try:
        results = DDGS().text(query, max_results=max_results)
        return "\n".join([f"- {r['title']}: {r['body']} (Source: {r['href']})" for r in results])
    except Exception as e:
        return f"Search failed: {str(e)}"

# ===========================
# 4. UNIFIED LLM WRAPPER
# ===========================

def call_llm(system_instruction, user_prompt, use_search=False, search_query=None):
    """
    Unified function for Gemini, OpenAI, and Claude.
    If use_search=True:
      - Gemini: Uses native Grounding.
      - OpenAI/Claude: Uses DuckDuckGo Bridge and injects context.
    """
    
    # --- GOOGLE GEMINI HANDLER ---
    if provider == "Google Gemini":
        tools = [Tool(google_search=GoogleSearch())] if use_search else None
        config = GenerateContentConfig(tools=tools, system_instruction=system_instruction, temperature=0.3)
        try:
            return client.models.generate_content(model=model_id, contents=user_prompt, config=config).text
        except Exception as e: return f"Gemini Error: {e}"

    # --- SEARCH INJECTION FOR OTHERS ---
    # For ChatGPT/Claude, we manually fetch data and append it to the prompt
    final_prompt = user_prompt
    if use_search and search_query:
        with st.spinner(f"🕵️ Bridging to live web via DuckDuckGo for {provider}..."):
            web_data = search_web_duckduckgo(search_query)
            final_prompt = f"""
            CONTEXT FROM LIVE WEB SEARCH:
            {web_data}
            
            USER QUERY:
            {user_prompt}
            """

    # --- OPENAI HANDLER ---
    if provider == "OpenAI (ChatGPT)":
        try:
            messages = [{"role": "system", "content": system_instruction}, {"role": "user", "content": final_prompt}]
            response = client.chat.completions.create(model=model_id, messages=messages, temperature=0.3)
            return response.choices[0].message.content
        except Exception as e: return f"OpenAI Error: {e}"

    # --- ANTHROPIC HANDLER ---
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
    # We pass 'search_query' specifically for the DuckDuckGo Bridge
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
