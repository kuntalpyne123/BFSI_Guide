import streamlit as st
import os
import time
import random
from streamlit.errors import StreamlitAPIException

# --- LIBRARY IMPORTS ---
# We import these conditionally or try/except to prevent crashes if libs are missing
try:
    from google import genai
    from google.genai.types import GenerateContentConfig, Tool, GoogleSearch
except ImportError:
    pass # Handle later

try:
    import openai
except ImportError:
    pass

try:
    import anthropic
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
# 2. SIDEBAR CONFIGURATION (New Multi-Provider Logic)
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
        st.info("⚡ Supports Live Google Search Grounding")
        
        # Key Source Selection
        key_source = st.radio(
            "API Key Source:",
            ("Use Free Default Key", "Enter My Own Key"),
            help="Default key uses the app admin's quota."
        )

        if key_source == "Use Free Default Key":
            try:
                if "GEMINI_API_KEY" in st.secrets:
                    api_key = st.secrets["GEMINI_API_KEY"]
                else:
                    st.error("🚨 Default key not found in secrets!")
            except StreamlitAPIException:
                st.error("Secrets not available locally.")
        else:
            api_key = st.text_input("Enter Gemini API Key", type="password")
        
        # Model Selection
        model_choice = st.selectbox(
            "Select Gemini Model:",
            ("Flash 2.5 (Fast)", "Pro 2.5 (Stable)", "Pro 3.0 (Preview)")
        )
        if "Flash" in model_choice: model_id = "gemini-2.5-flash"
        elif "2.5" in model_choice: model_id = "gemini-2.5-pro"
        else: model_id = "gemini-3-pro-preview"

    # 2. OPENAI CONFIG
    elif provider == "OpenAI (ChatGPT)":
        st.warning("⚠️ OpenAI does not support native Search Grounding in API. Results may be dated.")
        api_key = st.text_input("Enter OpenAI API Key", type="password")
        model_id = st.selectbox("Select Model:", ("gpt-5-nano", "gpt-5-mini", "gpt-5"))

    # 3. ANTHROPIC CONFIG
    elif provider == "Anthropic (Claude)":
        st.warning("⚠️ Claude does not support native Search Grounding in API. Results may be dated.")
        api_key = st.text_input("Enter Anthropic API Key", type="password")
        model_id = st.selectbox("Select Model:", ("claude-3-5-sonnet-20241022", "claude-3-opus-20240229"))

    # --- C. INITIALIZATION ---
    if not api_key:
        st.warning(f"Please configure {provider} API Key to start.")
        st.stop()
    
    # Initialize Clients based on provider
    client = None
    if provider == "Google Gemini":
        try:
            client = genai.Client(api_key=api_key)
        except Exception as e:
            st.error(f"Gemini Error: {e}")
    elif provider == "OpenAI (ChatGPT)":
        try:
            client = openai.OpenAI(api_key=api_key)
        except Exception as e:
            st.error(f"OpenAI Error: {e}")
    elif provider == "Anthropic (Claude)":
        try:
            client = anthropic.Anthropic(api_key=api_key)
        except Exception as e:
            st.error(f"Anthropic Error: {e}")

# ===========================
# 3. UNIFIED LLM WRAPPER (The Bridge)
# ===========================

def call_llm(system_instruction, user_prompt, use_search=False):
    """
    A unified function to call ANY provider (Gemini, OpenAI, Claude).
    Handles the syntax differences automatically.
    """
    
    # --- GOOGLE GEMINI HANDLER ---
    if provider == "Google Gemini":
        tools = [Tool(google_search=GoogleSearch())] if use_search else None
        config = GenerateContentConfig(
            tools=tools,
            system_instruction=system_instruction,
            temperature=0.1
        )
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=user_prompt,
                config=config
            )
            return response.text
        except Exception as e:
            return f"Gemini Error: {str(e)}"

    # --- OPENAI HANDLER ---
    elif provider == "OpenAI (ChatGPT)":
        try:
            # Merge system instruction into messages
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ]
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"OpenAI Error: {str(e)}"

    # --- ANTHROPIC HANDLER ---
    elif provider == "Anthropic (Claude)":
        try:
            response = client.messages.create(
                model=model_id,
                system=system_instruction,
                messages=[{"role": "user", "content": user_prompt}],
                max_tokens=4000,
                temperature=0.1
            )
            return response.content[0].text
        except Exception as e:
            return f"Claude Error: {str(e)}"

# ===========================
# 4. AGENT PERSONAS
# ===========================

# Note: I removed the strict requirement for "Google Search Tool" in the instruction
# because OpenAI/Claude don't have it natively.
RESEARCHER_INSTRUCTION = """
ROLE: Lead Financial Scrutinizer.
GOAL: Find comparative data, hidden fees, and regulatory risk.

INSTRUCTIONS:
1. Compare 2-3 similar financial products.
2. Identify hidden fees (prepayment, late payment).
3. Check for regulatory red flags.

OUTPUT: Provide a RAW text dump of findings.
"""

EDITOR_INSTRUCTION = """
ROLE: Senior Risk and Compliance Analyst.
GOAL: Turn raw data into an objective summary.

MANDATORY OUTPUT SECTIONS:
1. Hidden Fee Detection
2. Risk Score (1-10)
3. Stress Test Simulation
4. Comparative Offer (TCO)

STRUCTURE:
## 💰 Total Cost & Risk Summary
## ⚖️ Comparative Offer Table
## 🚨 Hidden Fee Report
## ⭐ Analyst Verdict
"""

PERSONALIZER_INSTRUCTION = """
ROLE: Ethical Financial Advisor.
GOAL: Match user profile to the product.
OUTPUT: Transparent recommendation letter (Strong Match/Poor Match).
"""

# ===========================
# 5. APP LOGIC (Using Wrapper)
# ===========================

def run_research(product_name):
    # Only Gemini uses the 'use_search=True' flag effectively here
    prompt = f"Current Date: December 2025. Investigate rival offers for: {product_name}. Confirm hidden fees."
    return call_llm(RESEARCHER_INSTRUCTION, prompt, use_search=True)

def generate_report(product_name, research_data):
    prompt = f"Research Data:\n{research_data}\n\nGenerate the Master Financial Report."
    return call_llm(EDITOR_INSTRUCTION, prompt)

def generate_personal_rec(product_name, research_data, user_profile):
    prompt = f"Research Data: {research_data}\nUser Profile: {user_profile}\nGenerate verdict."
    return call_llm(PERSONALIZER_INSTRUCTION, prompt)

# ===========================
# 6. APP INTERFACE
# ===========================

st.title("📈 NexFin Intelligence")
st.caption(f"Powered by **{provider} ({model_id})**")

# --- PHASE 1: INPUT ---
with st.form("research_form"):
    product_input = st.text_input("Analyze Financial Product or Document:", 
        placeholder="e.g. HDFC Home Loan Disclosure, SBI Mutual Fund Terms")
    submitted = st.form_submit_button("🚀 Run Financial Scrutiny")

if submitted and product_input:
    st.session_state.product_name = product_input
    st.session_state.messages = [] 
    st.session_state.general_report = None 
    
    status = st.status(f"🕵️ Initiating Scrutiny via {provider}...", expanded=True)
    
    try:
        status.write(f"🌍 **The Hunter:** Gathering intelligence...")
        research_data = run_research(product_input)
        st.session_state.research_data = research_data
        
        status.write("🧠 **The Analyst:** Calculating Risk & TCO...")
        report_text = generate_report(product_input, research_data)
        st.session_state.general_report = report_text
        
        status.update(label="✅ Analysis Complete!", state="complete", expanded=False)
        
    except Exception as e:
        status.update(label="❌ Error", state="error")
        st.error(f"System Error: {e}")

# --- PHASE 2: DISPLAY REPORT ---
if st.session_state.general_report:
    st.divider()
    st.markdown(st.session_state.general_report)
    
    with st.expander("🔍 Regulatory & Raw Data Transparency"):
        if provider == "Google Gemini":
            st.info("✅ Verified with Live Google Search Grounding")
        else:
            st.warning("⚠️ Generated using internal model knowledge (No Live Search). Verification recommended.")
        st.text_area("Raw Scrutiny Notes", st.session_state.research_data, height=200)

    st.divider()

    # --- PHASE 3: PERSONALIZATION ---
    st.markdown("## 👤 Personal Financial Advisor")
    
    with st.container(border=True):
        user_profile = st.text_area("Tell us about your financial goal:", placeholder="e.g. Loan for house, max tax saving...")
        
        if st.button("✨ Get Personalized Verdict"):
            if user_profile:
                with st.spinner("Simulating outcome..."):
                    rec = generate_personal_rec(st.session_state.product_name, st.session_state.research_data, user_profile)
                    st.markdown("### 💌 Advisor Report")
                    st.markdown(rec)

    st.divider()

    # --- PHASE 4: CHAT ---
    st.markdown(f"## 💬 Chat with {provider}")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a follow-up question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chat_response = call_llm(
                    f"You are a BFSI advisor. Context: {st.session_state.research_data}",
                    prompt
                )
                st.markdown(chat_response)
        st.session_state.messages.append({"role": "assistant", "content": chat_response})
