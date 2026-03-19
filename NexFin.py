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
    if not client: raise Exception("Error: Client not initialized. Check API Key.")

    if provider == "Google Gemini":
        tools = [Tool(google_search=GoogleSearch())] if use_search else None
        config = GenerateContentConfig(tools=tools, system_instruction=system_instruction, temperature=0.1)
        try:
            return client.models.generate_content(model=model_id, contents=user_prompt, config=config).text
        except Exception as e: 
            raise Exception(f"Gemini API Error: {e}")

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
        except Exception as e: 
            raise Exception(f"OpenAI API Error: {e}")

    elif provider == "Anthropic (Claude)":
        try:
            response = client.messages.create(model=model_id, system=system_instruction, messages=[{"role": "user", "content": final_prompt}], max_tokens=4000, temperature=0.3)
            return response.content[0].text
        except Exception as e: 
            raise Exception(f"Claude API Error: {e}")

# ===========================
# 5. AGENT PERSONAS
# ===========================

RESEARCHER_INSTRUCTION = """
ROLE: Senior Quantitative Forensic Analyst.
GOAL: Compile an exhaustive, heavily researched, and data-dense intelligence dossier on the requested financial product. 
MANDATE: You must adapt your data gathering to the specific asset class. Do not summarize; extract raw, hard facts, numbers, and direct terms.

CORE INTELLIGENCE REQUIREMENTS:
1. Asset Classification & Mechanics: Identify exactly what this product is (e.g., Equity, Derivative, Debt instrument, Consumer Loan, Mutual Fund). How does it actually work?
2. The "Hidden" Cost Forensics: 
   - For Credit/Loans: Find exact APRs, origination fees, prepayment penalties, and late fees.
   - For Equities/Funds: Find Expense Ratios (TER), 12b-1 fees, bid-ask spreads, and commission structures.
   - For Derivatives: Look for margin requirements, carrying costs, and theta decay implications.
3. Risk & Regulatory Exposure: What are the macro/micro risks? (e.g., interest rate sensitivity, liquidity risk, counterparty risk). Are there any recent SEC, FINRA, or global regulatory actions/warnings related to this specific product or its issuer?
4. Comparative Benchmarking: Identify 2-3 direct market competitors or alternative indexes. What are their comparative yields, costs, or performance metrics?

OUTPUT FORMAT: A highly structured, dense RAW TEXT DUMP organized by the categories above. Prioritize numbers, percentages, and factual clauses over narrative text.
"""
EDITOR_INSTRUCTION = """
ROLE: Chief Risk Officer (CRO) & Lead Financial Architect.
GOAL: Synthesize raw intelligence into a massive, institutional-grade Financial Impact & Risk Report.
TONE: Authoritative, objective, ruthlessly analytical, and completely devoid of marketing fluff. 

STRICT REPORT STRUCTURE:
## 1. Executive Synopsis
(Provide a high-level, definitive breakdown of the product, its primary use case, and its core value proposition.)

## 2. Structural Mechanics & Fee Forensics
(Detail exactly how this product makes money for the issuer and costs money for the user. Expose all hidden fees, spreads, expense ratios, or penalties identified in the data.)

## 3. The Risk Matrix
(Categorize and explain the specific risks. Dynamically adapt this to the asset:
- Market/Volatility Risk
- Liquidity Risk (Can you exit easily?)
- Credit/Counterparty Risk
- Regulatory/Macro Risk)

## 4. Competitive Alternative Benchmark
(Create a Markdown Table comparing this product against 2-3 direct alternatives identified in the research. Compare on Cost, Risk Level, and Potential Yield/Benefit.)

## 5. The CRO Verdict
(A definitive, brutal assessment of the product's overall quality and systemic risk. Provide a Risk Score out of 100 with a strict justification.)
"""
PERSONALIZER_INSTRUCTION = """
ROLE: Fiduciary Wealth Manager & Strict Ethical Advisor.
GOAL: Cross-reference the complex product report with the User's specific profile, goals, and risk tolerance to determine absolute suitability.
TONE: Highly empathetic but uncompromising on financial safety. You speak directly to the user ("You").

INSTRUCTIONS:
Do not just say "this is good" or "this is bad." You must build a bridge between the product's mechanics and the user's reality.

OUTPUT STRUCTURE:
### 1. The Fit Check
(A direct, conversational analysis of how the product aligns—or completely clashes—with their stated goals, capital, and timeframe.)

### 2. Scenario Simulation: Day-in-the-Life
(Paint a realistic picture of what holding this product looks like for them. E.g., "If the market drops 10%, here is exactly what happens to your portfolio," or "In year 3 of this loan, here is what your cash flow looks like.")

### 3. The Hard Truth
(Highlight the single biggest threat this product poses to this specific user's profile.)

### 4. Fiduciary Verdict
(Provide a strict "GREEN LIGHT", "YELLOW LIGHT (Proceed with caution/changes)", or "RED LIGHT (Avoid)". Give 2 clear, actionable next steps based on the verdict.)
"""

from datetime import datetime

# ===========================
# 6. APP LOGIC (UPDATED)
# ===========================

def run_research(product_name):
    # Dynamically grab the current month and year
    current_date = datetime.now().strftime("%B %Y")
    current_year = datetime.now().year
    
    prompt = f"""
    Current Date: {current_date}. 
    Target Asset/Product: {product_name}.
    
    Execute your forensic data gathering mandate. Identify the asset class, extract exact fee structures, uncover regulatory risks, and find 2-3 direct competitors with their metrics. Do not summarize—give me the raw numbers and facts.
    """
    
    # A broadened, dynamic search query designed to catch loans, stocks, OR funds
    search_query = f"{product_name} current rates OR price hidden fees OR expense ratio regulatory risk competitors {current_year}"
    
    return call_llm(RESEARCHER_INSTRUCTION, prompt, use_search=True, search_query=search_query)

def generate_report(product_name, research_data):
    prompt = f"""
    Target Asset/Product: {product_name}
    
    Raw Intelligence Data:
    {research_data}
    
    Synthesize this data into the strict CRO Financial Impact & Risk Report. Ensure the Competitive Matrix table is formatted correctly in Markdown and all risks are categorized.
    """
    return call_llm(EDITOR_INSTRUCTION, prompt)

def generate_personal_rec(product_name, research_data, user_profile):
    prompt = f"""
    Target Asset/Product: {product_name}
    
    User Profile / Financial Goal: 
    {user_profile}
    
    Raw Intelligence Data:
    {research_data}
    
    Write the Fiduciary Consultation Letter. Be brutally honest about the risks this specific user faces based on their profile, and provide a definitive Green/Yellow/Red light verdict with actionable next steps.
    """
    return call_llm(PERSONALIZER_INSTRUCTION, prompt)

# ===========================
# 7. APP INTERFACE
# ===========================

st.title("📈 NexFin Intelligence")
st.caption(f"Powered by **{provider} ({model_id})**")

with st.form("research_form"):
    product_input = st.text_input("Analyze Financial Product:", placeholder="e.g. HDFC Home Loan, SBI Mutual Fund")
    submitted = st.form_submit_button("▶️ Run Analysis")

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
    st.session_state.research_data = None # Added to clear previous failed run data
    
    status = st.status(f"🕵️ Initiating Scrutiny via {provider}...", expanded=True)
    
    try:
        status.write(f"🌍 **The Hunter:** Gathering live intelligence...")
        research_data = run_research(product_input)
        st.session_state.research_data = research_data
        
        status.write("🧠 **The Analyst:** Analysing Product type and risks...")
        report_text = generate_report(product_input, research_data)
        st.session_state.general_report = report_text
        
        if using_free_key:
            st.session_state.usage_count += 1
            st.toast(f"Free Quota Used: {st.session_state.usage_count}/{FREE_USAGE_LIMIT}")
        
        status.update(label="✅ Complete!", state="complete", expanded=False)
        
    except Exception as e:
        status.update(label="❌ Analysis Failed", state="error")
        st.error(f"System Error: {e}")

if st.session_state.general_report:
    st.divider()
    st.markdown(st.session_state.general_report)
    
    with st.expander("🔍 Raw Data Transparency"):
        if provider == "Google Gemini": st.info("✅ Verified with Google Search")
        else: st.info("✅ Verified with DuckDuckGo Search")
        st.text_area("Raw Notes", st.session_state.research_data, height=200)

    st.divider()
    st.markdown("## 👨‍💼 Personal Advisor")
    with st.container(border=True):
        user_profile = st.text_area("Financial Goal:", placeholder="e.g. Loan for house...")
        if st.button("✨ Get Your Verdict"):
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
