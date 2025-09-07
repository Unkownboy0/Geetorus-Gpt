import streamlit as st
import random
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Get Gemini API key from Streamlit secrets
gemini_api_key = st.secrets["gemini_api_key"]

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # or another Gemini model if available
    google_api_key=gemini_api_key,
    temperature=0.3,
    max_tokens=4096,  # Increase this value if needed
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an ethical hacking trainer. Always provide detailed, step-by-step explanations and include complete, runnable code examples in your responses. Do not skip any important details and You are an ethical hacking trainer who helps students understand how ethical hacking works and learn about it. You need to assist students by generating code snippets, suggesting tools, or providing step-by-step instructions based on the task and scenario provided. Everything performed is for educational purposes and with permission,If the user asks for an explanation in Thanglish (Tamil in English letters), or says 'explain in Thanglish', provide your answer in Thanglish as well as English,You can build scripts and automations for users and You are Geetorus GPT, an advanced AI assistant and vibe coder You can help users learn how to develop, script, automate tasks, share project links, and assist with website development, networking, HTML, CSS, JavaScript, and more.",
        ),
        ("human", "Task Type: {task_type}\nScenario: {scenario}\nFocus: {focus}\n"),
    ]
)

chain = prompt | llm

page_bg_img = """
  <style>
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
  header {visibility: hidden;}
    [data-testid="stAppViewContainer"] > .main {
        background-image: linear-gradient(to right, #000000, #00000);
        opacity: 0.8;
        padding: 0;
    }
  .stExpander {
      margin: 0;
      padding: 0;
  }
  .stMarkdown, .stText {
      width: 100%;
      margin: 0;
      padding: 0;
  }
  </style>
  """
st.markdown(page_bg_img, unsafe_allow_html=True)

gradient_text_html = """
  <style>
  .gradient-text {
      font-weight: bold;
      background: -webkit-linear-gradient(left, white, white);
      background: linear-gradient(to right, black, green);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      display: inline;
      font-size: 3em;
  }
  </style>
  <div class="gradient-text">Geetorus GPT</div>
  """
st.markdown(gradient_text_html, unsafe_allow_html=True)

st.caption("AI Integrity: Ethical Hacking, Elevated - Developed by GEETORUS")

st.write('')
st.write('')
st.write("By accessing our platform, users consent to engage in Ethical Hacking activities, strictly limited to legally authorized and defensive purposes. Unauthorized access is prohibited, and users must comply with all relevant laws and regulations. Any malicious activities, interference, or violation of intellectual property rights are strictly forbidden. By accessing our platform, users acknowledge and accept full responsibility for their activities conducted within the platform.")

agree = st.checkbox('I agree')

if agree:
    if "message" not in st.session_state:
        st.session_state["message"] = []

    model = st.radio(
        "Select a mode:",
        options=["Chat", "Malware Analysis", "Code Analysis"],
        index=0,
        horizontal=True,
    )
    st.session_state["model"] = model

    command = st.chat_input("HOW CAN I HELP YOU?")

    if "message" in st.session_state:
        for chat in st.session_state["message"]:
            with st.chat_message(chat["role"]):
                st.write(chat["message"])

    if command:
        with st.chat_message("user"):
            st.write(command)
            st.session_state["message"].append({"role": "user", "message": command})

        with st.chat_message("BOT"):
            with st.spinner('Processing...'):
                # Replace "hack" with "pentest" for safety
                command = command.replace("hack", "pentest").replace("hacking", "pentesting").replace("Hack", "pentest").replace("Hacks", "pentest")
                if st.session_state["model"] == 'Chat':
                    output = chain.invoke(
                        {
                            "task_type": "Chat",
                            "scenario": command,
                            "focus": "Help the user with proper explanation"
                        }
                    ).content
                    st.subheader("Response")
                    st.write(output)
                elif st.session_state["model"] == 'Malware Analysis':
                    output = chain.invoke(
                        {
                            "task_type": "Malware Analysis",
                            "scenario": command,
                            "focus": "Generate detailed reports and analysis of potential malware in the code"
                        }
                    ).content
                    st.subheader("Analysis Report")
                    st.write(output)
                elif st.session_state["model"] == 'Code Analysis':
                    output = chain.invoke(
                        {
                            "task_type": "Code Analysis",
                            "scenario": command,
                            "focus": "Analyze code for security vulnerabilities and best practices and provide an optimized code with report"
                        }
                    ).content
                    st.subheader("Code Analysis")
                    st.write(output)
                st.session_state["message"].append({"role": "BOT", "message": output})

else:
    st.warning("Please log in and agree to the terms to access the platform.")
