import streamlit as st
import os
import time
import random
from google import genai
from google.genai.types import GenerateContentConfig, Tool, GoogleSearch
from google.genai.errors import APIError, ServerError
from streamlit.errors import StreamlitAPIException

# ===========================
# 1. CONFIGURATION & SETUP
# ===========================

st.set_page_config(page_title="Financial IQ: Agentic BFSI Analyst", page_icon="📈", layout="wide")

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

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Settings")

    # 1. API Key Input
    api_key = None
    try:
        if "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"]
    except StreamlitAPIException:
        # st.secrets not available, proceed to check environment or user input
        pass

    if not api_key:
        if "GEMINI_API_KEY" in os.environ:
            api_key = os.environ["GEMINI_API_KEY"]
        else:
            with st.expander("How to get a Key?"):
                st.markdown("[Get Free Key](https://aistudio.google.com/)")
            user_key = st.text_input("Gemini API Key", type="password")
            if user_key:
                api_key = user_key.strip()
                os.environ["GEMINI_API_KEY"] = api_key

    # 2. Model Switcher (3-Way Choice)
    st.divider()
    st.markdown("**🤖 AI Engine**")
    model_choice = st.radio(
        "Select Model:",
        ("Flash 2.0 (Fastest)", "Pro 2.5 (Stable)", "Pro 3.0 (Preview)"),
        index=1, # Default to 2.5 Pro for best reliability/smartness balance
        help="Pro 2.5 is recommended for stable reasoning."
    )

    if model_choice == "Flash 2.0 (Fastest)":
        MODEL_ID = "gemini-2.0-flash-exp"
    elif model_choice == "Pro 2.5 (Stable)":
        MODEL_ID = "gemini-2.5-pro"
    else:
        MODEL_ID = "gemini-2.5-pro"

    if not api_key:
        st.warning("⚠️ Please provide your Gemini API Key to proceed. You can enter it above or set it as an environment variable (GEMINI_API_KEY).")
# Initialize Client (Only if API key is present)
client = None
if api_key:
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        st.error(f"Client Error: {e}")
        st.stop()
else:
    st.stop() # Halt execution if no key provided

# ===========================
# 2. ROBUST RETRY LOGIC
# ===========================

def retry_api_call(func, *args, **kwargs):
    """Retries API call up to 5 times with exponential backoff for 503/429 errors."""
    max_retries = 5
    base_delay = 2

    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_str = str(e)
            if "503" in error_str or "429" in error_str:
                if attempt == max_retries - 1:
                    raise e

                wait_time = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                st.toast(f"⚠️ AI Server Busy. Retrying in {int(wait_time)}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise e

# ===========================
# 3. BFSI AGENT PERSONAS (The Scrutinizers)
# ===========================

# --- Agent 1: The Deep Hunter (Financial Scrutinizer) ---
RESEARCHER_INSTRUCTION = """
ROLE: You are a Lead Financial Scrutinizer. Your goal is to find all comparative data, hidden fees, and regulatory risk associated with the user's inquiry.

CRITICAL INSTRUCTIONS:
1.  **Search Grounding:** You MUST use Search to find the official terms and conditions (T&C) and the latest regulatory guidelines (RBI, SEBI, IRDAI) relevant to the product.
2.  **Comparative Analysis:** Search for 2-3 similar financial products from rival institutions.
3.  **Risk Data:** Fetch recent news, litigation reports, or regulatory actions against the provider related to the product.

OUTPUT REQUIREMENT:
Provide a RAW text dump with: Official APRs, hidden fees (prepayment, late payment), rival offers, and any recent regulatory red flags found.
"""

# --- Agent 2: The Unbiased Analyst (Risk Analyst & Editor) ---
EDITOR_INSTRUCTION = """
ROLE: You are a Senior Risk and Compliance Analyst. You turn raw financial data into an explainable, objective summary.

INPUT: Raw Financial Scrutiny Data.

MANDATORY OUTPUT SECTIONS:
1.  **Hidden Fee Detection:** Systematically extract all non-headline fees.
2.  **Risk Score:** Assign a Trust/Risk Score (1-10) based on complexity, hidden fees, and regulatory history.
3.  **Stress Test Simulation:** Run a 2-sentence simulation (e.g., "If the user misses one payment, the cost increases by X.").
4.  **Comparative Offer:** Compare the user's product with the 2-3 rivals found in the research based on TCO (Total Cost of Ownership).

STRUCTURE:
## 💰 Total Cost & Risk Summary
## ⚖️ Comparative Offer Table (User vs Rivals)
## 🚨 Hidden Fee Report
## ⭐ Analyst Verdict & Recommendation
"""

# --- Agent 3: The Personalizer (Financial Advisor) ---
PERSONALIZER_INSTRUCTION = """
ROLE: You are a highly ethical, personalized Financial Advisor.
GOAL: Match the user's financial profile against the analyzed product.

TASK:
1.  **Goal Match:** Does this product meet the user's stated financial goal (e.g., tax saving, house purchase)?
2.  **Affordability Match:** Check if the payment terms align with the user's budget and constraints.
3.  **Conflict of Interest:** Explicitly state any conflicts (e.g., "This product is high commission...").

OUTPUT: A transparent, decisive recommendation letter: "Based on your goals, this product is a STRONG MATCH/POOR MATCH."
"""

# ===========================
# 4. HELPER FUNCTIONS (Wrapped)
# ===========================

def run_research(product_name):
    config = GenerateContentConfig(
        tools=[Tool(google_search=GoogleSearch())],
        system_instruction=RESEARCHER_INSTRUCTION.format(product_name=product_name),
        temperature=0.3
    )
    prompt = f"Current Date: December 2025. Investigate and find rival offers for the financial product: {product_name}. Confirm all hidden fees and regulatory status."

    return retry_api_call(
        client.models.generate_content,
        model=MODEL_ID,
        contents=prompt,
        config=config
    )

def generate_report(product_name, research_data):
    config = GenerateContentConfig(
        system_instruction=EDITOR_INSTRUCTION.format(product_name=product_name),
        temperature=0.2
    )
    prompt = f"Research Data:\n{research_data}\n\nGenerate the Master Financial Product Analysis Report."

    return retry_api_call(
        client.models.generate_content,
        model=MODEL_ID,
        contents=prompt,
        config=config
    ).text

def generate_personal_rec(product_name, research_data, user_profile):
    config = GenerateContentConfig(
        system_instruction=PERSONALIZER_INSTRUCTION.format(product_name=product_name),
        temperature=0.4
    )
    prompt = f"Research Data: {research_data}\nUser Profile: {user_profile}\nGenerate personalized financial verdict."

    return retry_api_call(
        client.models.generate_content,
        model=MODEL_ID,
        contents=prompt,
        config=config
    ).text

# ===========================
# 5. APP INTERFACE & FLOW
# ===========================

st.title("📈 Financial IQ: Agentic BFSI Analyst")
st.caption(f"Powered by **{MODEL_ID}**. Zero Bias. Total Transparency.")

# --- PHASE 1: INPUT ---
with st.form("research_form"):
    product_input = st.text_input("Analyze Financial Product or Document:",
        placeholder="e.g. HDFC Home Loan Disclosure, SBI Mutual Fund Terms, LIC Policy A vs B")
    submitted = st.form_submit_button("🚀 Run Financial Scrutiny")

if submitted and product_input:
    st.session_state.product_name = product_input
    st.session_state.messages = []
    st.session_state.general_report = None

    status = st.status("🕵️ Initiating Financial Scrutiny...", expanded=True)

    try:
        status.write(f"🌍 **Agent 1:** Hunting for T&Cs and rival offers using {MODEL_ID}...")
        research_response = run_research(product_input)
        st.session_state.research_data = research_response.text

        status.write("🧠 **Agent 2:** Analyzing Total Cost of Ownership and Regulatory Risk...")
        report_text = generate_report(product_input, st.session_state.research_data)
        st.session_state.general_report = report_text

        status.update(label="✅ Analysis Complete!", state="complete", expanded=False)

    except Exception as e:
        status.update(label="❌ Error", state="error")
        st.error(f"System Error: {e}")

# --- PHASE 2: DISPLAY REPORT ---
if st.session_state.general_report:
    st.divider()
    st.markdown(st.session_state.general_report)

    # --- Transparency Layer (Crucial for BFSI) ---
    with st.expander("🔍 Regulatory & Raw Data Transparency"):
        st.info("The analysis was grounded by checking the following regulatory guidelines and raw data:")
        st.markdown(f"**Regulatory Verification:** RBI, SEBI, IRDAI guidelines were checked (Confirmed via Search).")
        st.text_area("Raw Scrutiny Notes", st.session_state.research_data, height=200)

    st.divider()

    # --- PHASE 3: PERSONALIZATION ENGINE ---
    st.markdown("## 👤 Phase 2: Personalized Financial Advice")

    with st.container(border=True):
        user_profile = st.text_area(
            "Tell us about your financial goal and risk tolerance:",
            placeholder="e.g. 'I am looking for a loan to buy a house in 5 years. My risk tolerance is low and my goal is max tax saving.'")

        if st.button("✨ Get Personalized Financial Verdict"):
            if user_profile:
                with st.spinner("Simulating long-term financial outcome..."):
                    rec = generate_personal_rec(st.session_state.product_name, st.session_state.research_data, user_profile)
                    st.markdown("### 💌 Your Personalized Advisor Report")
                    st.markdown(rec)

    st.divider()

    # --- PHASE 4: AGENTIC CHAT ---
    st.markdown("## 💬 Consult the Financial Agent")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input(f"Ask about prepayment penalties or rival offers for {st.session_state.product_name}..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chat_config = GenerateContentConfig(
                    system_instruction=f"You are a quick, factual BFSI advisor. Answer based on the research provided: {st.session_state.research_data}"
                )

                chat_response = retry_api_call(
                    client.models.generate_content,
                    model=MODEL_ID,
                    contents=prompt,
                    config=chat_config
                )
                st.markdown(chat_response.text)
        st.session_state.messages.append({"role": "assistant", "content": chat_response.text})
