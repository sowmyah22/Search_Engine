import streamlit as st 
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun,DuckDuckGoSearchRun
from langchain.agents import initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv
load_dotenv()

## Arxiv and wikipedia tools
arxiv_wrapper=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=500)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)
search=DuckDuckGoSearchRun(name="search")

## Streamlit
st.title("Langchain Chat bot")
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter the Groq API Key",type="password")

## Messages
if  "messages" not in st.session_state:
          st.session_state["messages"]=[
               {"role":"assistant","content":"Ai chat bot"}
          ]
          
## Clear chat button          
if st.sidebar.button("Clear Chat"):
     st.session_state["messages"]=[
               {"role":"assistant","content":"Ai chat bot"}
          ]
     st.rerun()

## Display the chat
for msg in st.session_state.messages:
     st.chat_message(msg["role"]).write(['content'])
     
## Prompt
prompt=st.chat_input(placeholder="What is AI")
if prompt:
     st.session_state.messages.append({"role":"user","content":prompt})
     st.chat_message("user").write(prompt)
     
     ## Model
     model_choice=st.sidebar.selectbox("Choose LLM Model",["gemma2-9b-it","llama3-8b-8192"])
     model=ChatGroq(groq_api_key=api_key,model=model_choice,streaming=True)
     ## tools
     tools=[arxiv,search]
     ## Agent
     search_agent=initialize_agent(tools,model,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_errors=True)

     with st.chat_message("assistant"):
          st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
          response=search_agent.run(prompt,callbacks=[st_cb])
          st.session_state.messages.append({"role":"assistant","content":response})
          st.write(response)




