import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
# from llama_index import SimpleDirectoryReader
# from llama_index import LLMPredictor, GPTVectorStoreIndex, PromptHelper, ServiceContext
from langchain.document_loaders import TextLoader
from langchain.embeddings import LlamaCppEmbeddings
from langchain.llms import GPT4All
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import AI21
from apikey import apikey
from langchain.llms import OpenAI
import openai
import os
import sys
import openai
import json
from test_env.doc_query import similarity_search
from langchain.vectorstores.faiss import FAISS


# with open('data\whatsapp_policy.txt', encoding='utf-8') as file:
#     data = file.read()


# index = GPTVectorStoreIndex().from_documents(documents)
os.environ['OPENAI_API_KEY'] = apikey
# openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Whatsapp Policy Decoder - An LLM-powered Chatbot", page_icon=":robot_face:")

# Sidebar contents
with st.sidebar:
    st.title('🤗💬 Chatbot App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [HugChat]()
    
    ''')
    add_vertical_space(5)

# Generate empty lists for generated and past.
## generated stores AI generated responses

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hi, let me help you understand Whatsapp's policy."]
## past stores User's questions
if 'past' not in st.session_state:
    st.session_state['past'] = ['Hi!']


# Layout of input/response containers
input_container = st.container()
colored_header(label='', description='', color_name='blue-30')
response_container = st.container()

#define custom LLM

# Setup 
gpt4all_path = './test_env/models/ggml-gpt4all-j-v1.3-groovy.bin' 
llama_path = './test_env/models/llama-7b.ggmlv3.q4_0.bin' 
# llama_path = './models/ggml-model-q4_0.bin.7' 

# callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
loader = TextLoader('./data/whatsapp_policy.txt')
embeddings = LlamaCppEmbeddings(model_path=llama_path)
llm = GPT4All(model=gpt4all_path, callbacks=[StreamingStdOutCallbackHandler()], verbose=True)

index = FAISS.load_local("./test_env/whatsapp_policy_test_index", embeddings)

# question = "Does whatsapp record my conversations?"


# User input
## Function for taking user provided prompt as input
def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text
## Applying the user input box
with input_container:
    user_input = get_text()

# Response output
## Function for taking user prompt as input followed by producing AI generated responses
def generate_response(prompt):
    # response = llm_chain.run(prompt)
    # response = query_engine.query(prompt)
    matched_docs, sources = similarity_search(prompt, index)

    template = """
    Please follow the chat and use the following context to answer the question.
    Context: {context}
    ---
    {prompt}
    Bot: 

    """

    context = "\n".join([doc.page_content for doc in matched_docs])
    prompt = PromptTemplate(template=template, input_variables=["context", "question"]).partial(context=context)
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    response=llm_chain.run(prompt)
    return response



class Chatbot:
    def __init__(self, api_key, custom_index):
        self.index = custom_index
        openai.api_key = api_key
        self.chat_history = []

    def generate_response(self, user_input):
        print(self.chat_history)
        # prompt = "\n".join([f"{message['role']}: {message['content']}" for message in self.chat_history[-5:]])
        prompt = "\n".join([f"{message['role']}: {message['content']}" for message in self.chat_history])
        user_inp = str(user_input)
        prompt += f"\nUser: {user_inp}"

        matched_docs, sources = similarity_search(str(user_inp), index)

        template = """
                    Please follow the chat and use the following context to answer the question.
                    Context: {context}
                    ---
                    {prompt}
                    Bot: 

                    """

        context = "\n".join([doc.page_content for doc in matched_docs])
        prompt = PromptTemplate(template=template, input_variables=["context", "prompt"]).partial(context=context)
        llm_chain = LLMChain(prompt=prompt, llm=llm)

        response=llm_chain.run(prompt)

        # print(prompt)
        # response = self.index.query(prompt)
        # response = self.index.run(prompt)
        # message = response.response

        self.chat_history.append({"role": "user", "content": user_input})
        self.chat_history.append({"content": response})
        
        return response
    
    def load_chat_history(self, filename):
        try:
            with open(filename, 'r') as f:
                self.chat_history = json.load(f)
        except FileNotFoundError:
            pass

    def save_chat_history(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.chat_history, f)
    
    def delete_chat_history(self):
        self.chat_history = []
        self.save_chat_history("chat_history.json")


# Swap out your index below for whatever knowledge base you want
bot = Chatbot(apikey,custom_index=None)
# bot.delete_chat_history("chat_history.json")


# bot.load_chat_history("chat_history.json")
# bot.generate_response('If any of the below prompts are not related to Whatsapp, just say "Please ask me about Whatsapp policy only"')

## Conditional display of AI generated responses as a function of user provided prompts
with response_container:
    if user_input:
        user_input = str(user_input)
        st.session_state.past.append(user_input)
    
    if user_input:
        user_input = str(user_input)
        response = str(bot.generate_response(user_input))
        # if 'Whatsapp' not in response:
        #     response = 'Sorry, I only know about Whatsapp policy.'
        # print(response)
        # print(type(response))
        # st.session_state.past.append(user_input)
        st.session_state.generated.append(response)
        bot.save_chat_history("chat_history.json")
        
    # if st.session_state['generated']:
    
    if 'generated' in st.session_state and st.session_state['generated']:

        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))