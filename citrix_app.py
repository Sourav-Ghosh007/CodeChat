import openai
import os
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

key = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_API_KEY"] = key


# Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Creating a Code Chatbot"

## Prompt Template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful massistant . Please repsonse to the user queries"),
        ("user","Question:{question}")
    ]
)

# Give Parameters during the runtime
def generate_response(question, api_key, engine, temperature, max_tokens):
    try:
        # 1) Set the OpenAI key for both SDK and env‐var
        openai.api_key = api_key
        os.environ["OPENAI_API_KEY"] = api_key

        # 2) Instantiate the LLM with all parameters
        llm = ChatOpenAI(
            model=engine,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=api_key
        )

        # 3) Build and run the chain
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser
        answer = chain.invoke({"question": question})
        return answer

    except Exception as e:
        return f"An error occurred: {str(e)}"


# Title of the app
st.title("Chatbot With OpenAI")



# Sidebar for settings
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your Open AI API Key:",type="password")

# Select the OpenAI model
engine=st.sidebar.selectbox("Select any Open AI model for your task",["gpt-4o","o4-mini","gpt-4.1"])

# Adjust response parameter
temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

# Main interface for user input
st.write("Ask any question. Citrix is here to help you!")
user_input=st.text_input("Question:")

if user_input and api_key:
    response=generate_response(user_input,api_key,engine,temperature,max_tokens)
    st.write(response)

elif user_input:
    st.warning("Please enter the Open AI API Key in the side bar")
else:
    st.write("Please provide the user input")



