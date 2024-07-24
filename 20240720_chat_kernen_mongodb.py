# -*- coding: utf-8 -*-
"""
@author: yulialitvinova

"""
import os
import logging
logger = logging.getLogger(__name__)
import asyncio
import json
import getpass

import streamlit as st
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.memory import ConversationBufferMemory

from langchain.agents import initialize_agent, Tool, AgentType #load_tools, 
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.tools import Tool
from langchain_core.tools import Tool
from typing import Optional
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
load_dotenv()

#from langchain.chains import RetrievalQAWithSourcesChain
# from langchain.chains import LLMMathChain
#### numexpr
# from langchain.chains import ConversationChain
# from langchain.chains import LLMChain, LLMMathChain
#from langchain.chains import SimpleSequentialChain, RetrievalQA, ConversationalRetrievalChain

#########from langchain.retrievers. import WebResearchRetriever
#import web_research
from web_research import WebResearchRetriever

# from langchain_community.vectorstores import Chroma

###from streamlit_chromadb_connection.chromadb_connect import ChromadbConnection
# from langchain.vectorstores import FAISS
# from langchain.docstore import InMemoryDocstore
import pymongo
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from urllib.parse import quote_plus
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
############from vectorstores import MongoDBAtlasVectorSearch

#from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_google_community import GoogleSearchAPIWrapper
##############from search import GoogleSearchAPIWrapper
# from langchain_community.tools import DuckDuckGoSearchRun
# from langchain_community.utilities import SearchApiAPIWrapper
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
# from langchain.docstore import Wikipedia
# from langchain_community.tools.wikidata.tool import WikidataAPIWrapper, WikidataQueryRun

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser

import replicate
from transformers import LlamaModel, LlamaConfig

from langchain_core.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
                                )
# from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.prompts import PromptTemplate, HumanMessagePromptTemplate
# from langchain_core.messages import SystemMessage

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from langchain.docstore.document import Document
import time
from datetime import date, datetime
from urllib.parse import urlparse

from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC

# conda list -e > requirements.txt
# git push origin main:main

os.environ['OPENAI_API_KEY'] = st.secrets["openai_api_key"]
os.environ["OPEN_API_BASE"] = "https://api.openai.com/v1"
os.environ["GOOGLE_CSE_ID"] = st.secrets["google_cse_id"] 
os.environ["GOOGLE_API_KEY"] = st.secrets["google_api_key"]
os.environ["SERPAPI_API_KEY"] = st.secrets["serpapi_api_key"]

cse_id_simple_search = st.secrets["cse_id_simple_search"]

mongodb_username = st.secrets["mongodb_username"]
mongodb_password = st.secrets["mongodb_password"]
encoded_username = quote_plus(mongodb_username)
encoded_password = quote_plus(mongodb_password)

##uri = f'mongodb://{encoded_username}:{encoded_password}@your_host:your_port/your_database'
mongodb_atlas_cluster_uri = f'mongodb+srv://{encoded_username}:{encoded_password}@clusterfree.xiknzbp.mongodb.net/?appName=clusterfree'
client = MongoClient(mongodb_atlas_cluster_uri, server_api=ServerApi('1'))

# try:
#     client.admin.command('ping')
#     print("Pinged your deployment. You successfully connected to MongoDB!")
# except Exception as e:
#     print(e)

st.set_page_config(page_title="Gemeinde Kernen",
                   #page_icon=,
                   )
st.title("Pretotype")
# st.header("`Pretotype`")
st.caption("Chat für BürgerInnen: Gemeinde Kernen")
# st.info("`I am an AI that can answer questions by exploring, reading, and summarizing web pages.")

msgs = StreamlitChatMessageHistory() #(key="langchain_messages")
    
memory = ConversationBufferMemory(
    chat_memory=msgs, 
    return_messages=True, 
    memory_key="chat_history", 
    output_key="output",
    #input_key="input",
    #expected_arbitrary_type=BaseChatMessageHistory,
    )

if len(msgs.messages) == 0 or st.sidebar.button("Die Unterhaltung zurücksetzen"): #Reset chat history
    msgs.clear()
    msgs.add_ai_message("Zu welcher Lebenssituation brauchen Sie Unterstützung?")
    st.session_state.steps = {}

avatars = {"human": "user", "ai": "assistant"}
for idx, msg in enumerate(msgs.messages):
    with st.chat_message(avatars[msg.type]):
        # Render intermediate steps if any were saved
        for step in st.session_state.steps.get(str(idx), []):
            if step[0].tool == "_Exception":
                continue
            with st.status(f"**{step[0].tool}**: {step[0].tool_input}", state="complete"):
                st.write(step[0].log)
                st.write(step[1])
        st.write(msg.content)

temperature=0
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=temperature,
    streaming=True,
    )

embeddings_model = OpenAIEmbeddings()
embeddings_size = 1536 #1536 #1024
#HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
#HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-en")

# vectorstore = Chroma(
#     embedding_function=embeddings_model, #OpenAIEmbeddings(), 
#     persist_directory="./99_chroma_test"
#     #persist_directory="https://github.com/yulialitvinova/chatbot_kernen/tree/main/99_chroma_db_kernen"
#     )
database = client["99_kernen"] #db_name = "99_kernen"
collection = database["99_kernen_general"] #collection_name = "99_chroma_db_kernen"
vector_search_index = "vector_index"
vectorstore = MongoDBAtlasVectorSearch( #).from_documents(
    #documents = docs,
    embedding = embeddings_model,
    collection = collection,
    index_name = vector_search_index
)

# vectorstore_urls_sugg = Chroma(
#     embedding_function=embeddings_model,
#     persist_directory="./99_chroma_test_urls"
#     )
database_url = client["99_kernen"]
collection_url = database_url["99_kernen_urls"]
vector_search_index_url = "vector_index"

vectorstore_urls_sugg = MongoDBAtlasVectorSearch(
    embedding=embeddings_model,
    collection=collection_url,
    index_name=vector_search_index_url
    )

# database_history = client["99_kernen"]
# collection_chat_history = database_history["99_kernen_chat_history"]
session_id_today = str(datetime.today().strftime('%Y-%m-%d %H'))
chat_history_mongodb = MongoDBChatMessageHistory(
    session_id = session_id_today, # "test",
    connection_string=mongodb_atlas_cluster_uri,
    database_name="99_kernen", #database,
    collection_name="99_kernen_chat_history", #collection_chat_history,
)

# ###FAISS, alternative to Chroma
# index = faiss.IndexFlatL2(embeddings_size)
# vectorestore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
# ###FAISS

search = GoogleSearchAPIWrapper()
simple_search = GoogleSearchAPIWrapper(google_cse_id=cse_id_simple_search)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=embeddings_size, chunk_overlap=64)
# text_splitter_urls_sugg = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=64)

# prompt_search = PromptTemplate(
#     input_variables=["question"],
#     template="""<<SYS>> \n You are an assistant tasked with improving Google search \
# results. \n <</SYS>> \n\n [INST] Generate THREE Google search queries that \
# are similar to this question. The output should be a numbered list of questions, \
# each should be focused on how the matter is dealt with in Germany, Baden-Württemberg, \
# and should have a question mark at the end: \n\n {question} [/INST]""",
# )

web_retriever_for_tool = WebResearchRetriever.from_llm(
                    vectorstore=vectorstore, ## vectorestore where webpages are stored
                    llm=llm, 
                    search=search, ## Google Search API Wrapper
                    num_search_results=3,
                    text_splitter=text_splitter,
                    #prompt=prompt_search, ### see web_retriever.py PROMPT
                    )

template_query_spec_search="""<<SYS>> \n You are an assistant to citizens in difficult situations. \
        
        To answer the questions, use the information in documents provided to you. \
        The answer must be in the same language as the user's question, or input: if the user asks in German, reply in German. \
        In your response, always use "Sie" to address the user, keine "Du". \
        When applicable, provide your response to the user in bullet points. \
        
        ALWAYS return a "SOURCES" part in your answer. \
        QUESTION: {question} \
        Documents to use to provide the answer: {summaries} \
        FINAL ANSWER: [final response here] \
        SOURCES:  {sources}
        """

QA_CHAIN_PROMPT_spec_search = PromptTemplate(
    template=template_query_spec_search, input_variables=["summaries", "question", "sources"] # , "link"]
    )

#If you do not know the answer reply with 'Es tut mir Leid, ich habe nicht genügend Informationen'.\

#vectorstore_urls_sugg.add_documents(documents)
web_retriever_for_tool_urls = vectorstore_urls_sugg.as_retriever(
    search_kwargs={"k": 4}) # , search_type="mmr"
##### https://python.langchain.com/v0.2/docs/how_to/vectorstore_retriever/

template_query_urls_sugg="""<<SYS>> \n You are an assistant to citizens who seek official information for their difficult situations. \
        
        You need to find the relevant url (i.e., Document(metadata=source)) in retrieved DOCUMENTS, based on the Document(page_content) extracted above.\
        
        Provide the list of the URLs with page titles as your FINAL ANSWER.\
        Reply in the following format: To [execute the action the user asked for], please, visit [URL].\
        Reply in the language of the user's question.\
        
        QUESTION: {question}\
        Retrieved documents: {summaries}\
        FINAL ANSWER: [final response here] \
        SOURCES: [list of the URLs] \
        """
#{source_documents}
#SOURCE: {source_documents}
#SOURCES: [list of the URLs] \
#ALWAYS return a "SOURCES" part in your answer.\

QA_CHAIN_PROMPT_urls_sugg = PromptTemplate(template=template_query_urls_sugg, 
                         input_variables=["summaries", "question"])

class WebResearchTool(BaseTool):
    #name = "search_specific_webpages"
    #description = "Tool to research information on the pages specified in programmable search from Google."
    
    name: str = "search_specific_webpages"
    description: str = "Tool to search information for citizens being in difficult situations. The tool scraps official webpages."
    #api_wrapper: GoogleSearchAPIWrapper
    web_retriever: WebResearchRetriever #.from_llm(vectorstore=vectorstore, llm=llm, search=search, num_search_results=3, text_splitter=text_splitter)
    #RetrievalQAWithSourcesChain
    # def __init__(self, web_retriever: WebResearchRetriever):
    #     self.web_retriever = web_retriever

    def _run(
        self,
        question: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        ):
        #return self.web_retriever.get_relevant_documents(query)
        #result = self.qa_chain({"question": question})
        # qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
        #     llm, 
        #     chain_type="stuff", 
        #     retriever = web_retriever_for_tool, 
        #     chain_type_kwargs = {"prompt": QA_CHAIN_PROMPT_spec_search}
        # )
        qa_chain = QA_CHAIN_PROMPT_spec_search | llm
        returned_documents = web_retriever_for_tool.invoke(question) #{"query": question}
        docs_for_summaries = []
        for document in returned_documents:
            docs_for_summaries.append(document.page_content)
            #print(f"*The next document*: {document.page_content}")

        sources_for_user = []
        for document in returned_documents:
            sources_for_user.append(document.metadata["source"])

        result = qa_chain.invoke({"question": question, "summaries": docs_for_summaries, "sources": sources_for_user})
        return result.content #["answer"]

web_tool = WebResearchTool(web_retriever=web_retriever_for_tool)

class ServiceBWurlsSearchTool(BaseTool):
    name: str = "search_urls_on_service_bw"
    description: str = "Tool to use if search_specific_webpages has not provided enough information. Should be used to provide the user with links relevant to his or her query."
    #retriever: Any #Chroma.as_retriever #  any #VectorStoreRetriever # vectorstore.as_retriever() ### CHECK

    def _run(
        self,
        question: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        ):
        # #driver_path = "/04_search_or_scrapping/chromedriver-win64"
        # #driver_path = "C:/Users/litvi/git/knowledge-chatbot-kernen/04_search_or_scrapping/chromedriver-win64"
        # driver_path = "https://github.com/yulialitvinova/knowledge_chatbot_kernen/tree/main/04_search_or_scrapping/chromedriver-win64"

        # url_search = "https://www.service-bw.de/zufi/lebenslagen/5000312"
        # WINDOW_SIZE = "1920,1080"
        # chrome_options = Options()
        # chrome_options.add_argument("--headless")
        # chrome_options.add_argument(f"--window-size={WINDOW_SIZE}")
        # service = Service(ChromeDriverManager().install())
        # driver = webdriver.Chrome(service=service, options=chrome_options)
        # driver.get(url_search)
        # time.sleep(8)
        # try:
        #     while True:
        #         see_more_button = WebDriverWait(driver, timeout=3).until(
        #             EC.element_to_be_clickable((By.CSS_SELECTOR, 'button[data-testid="pagination-button-load-more"]'))
        #         )       
        #         if see_more_button:
        #             see_more_button.click()
        #             time.sleep(2)
        #         else:
        #             break
        # except Exception as e:
        #     logger.info(f"No more 'see more' buttons or an error occurred: {e}")
        # page_source = driver.page_source
        # driver.quit()
        # soup = BeautifulSoup(page_source, 'html.parser')
        # parsed_url = urlparse(url_search)
        # base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        # documents = []
        # urls = []
        # contains_keywords = ["lebenslagen/", "leistungen/"]
        # for link in soup.find_all('a', href=True):
        #     href = link['href']
        #     link_text = link.get_text(strip=True)
        #     if any(keyword in href for keyword in contains_keywords):
        #         if href.startswith('/'):
        #             href = base_url + href
        #         urls.append((link_text, href))
        #         url_for_docs = [Document(page_content=link_text, metadata={"source": href})]
        #         documents.extend(url_for_docs)
        # #documents = text_splitter_urls_sugg.split_documents(documents)
        # documents = text_splitter.split_documents(documents)
        # vectorstore_urls_sugg.add_documents(documents)
        retriever = vectorstore_urls_sugg.as_retriever(search_kwargs={"k": 2})
        # https://www.mongodb.com/docs/atlas/atlas-vector-search/ai-integrations/langchain/#create-the-atlas-vector-search-index
        returned_documents = retriever.invoke(question)
        qa_chain = QA_CHAIN_PROMPT_urls_sugg | llm
        # qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
        #     llm, 
        #     chain_type="stuff", 
        #     retriever = web_retriever_for_tool_urls, 
        #     verbose=True,
        #     chain_type_kwargs = {"prompt": QA_CHAIN_PROMPT_urls_sugg}
        # )
        result = qa_chain.invoke({"question": question, "summaries": returned_documents})
        return result.content #["answer"]

service_bw_search_tool = ServiceBWurlsSearchTool()

params = {
    "engine": "google",
    "gl": "de",
    "hl": "de",
    "google_domain": "google.de",
    }
# search_serpapi = SerpAPIWrapper(params=params)

# search_ddg = DuckDuckGoSearchRun() #(name="Search")

# llm_math = LLMMathChain(llm=llm)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, 
                                  doc_content_chars_max=300
                                  )
wikipedia = WikipediaQueryRun(
    api_wrapper=api_wrapper,
    ) 
# #docstore=DocstoreExplorer(Wikipedia()) ##### needs another type of agent, without 'memory'
# ##### https://www.pinecone.io/learn/series/langchain/langchain-agents/

# wikidata = WikidataQueryRun(api_wrapper=WikidataAPIWrapper())

tools = [
    Tool(
        name='search_specific_webpages',
        description="Tool to search answers for citizens who need information about their specific situations like \
            family (children and childcare, marriage or divorce, schooling and further education), job, healthcare, \
            or ask where and how they can find support. The tool scraps official webpages.",
        func=web_tool._run,
        ),
    Tool(
        name="search_urls_on_service_bw",
        func=service_bw_search_tool.run,
        description="Very helpful tool if the user asks how she can apply for support or service from officials, needs application forms or asks how she or he can submit an application form, for example, Wie kann ich ... beantragen? Also helpful when the uer claims the link is not correct, redirects to an error page or similar, or that the link is not relevant.",
    ),
    Tool(
        name="simple_search_internet",
        func=simple_search.run,
        description="Tool to search Internet for general current information, i.e., information after 2021. Helpful to provide information on news or upcoming events.",
    ),
    # Tool(
    #     name="simple_search_serpapi",
    #     func=search_serpapi.run,
    #     description="Tool to search Internet for general information after 2020, i.e., search internet for the present-day information. Helpful to provide information on upcoming events or news.",
    #     ),
    # Tool(
    # name="simple_search_ddg",
    # func=search_ddg.run,
    # description="Search Internet for information after 2020, i.e., search internet for the present-day information. Alternative tool to simple_search_googleapiwrapper.",
    # ),
    Tool(
        name="wikipedia",
        func=wikipedia.run,
        description="Tool to search for the term in Wikipedia if the search on the official webpages or search for the present-day information in the Internet haven't provided relevant information.",
        #return_direct=True,        
        ), 
    # Tool(
    #     name='calculator',  
    #     func=llm_math.run,
    #     description='Useful for when you need to answer questions that require calculations like adding, subtraction, multiplying.',
    #     ),
    ]

            # The tool should be used if the tool search_specific_webpages has not provided a valid answer.\
            # The tool is helpful if the user claims in the query that the provided links do not work, are not correct, were not found, redirect to not existing or error pages, etc.\
            # If so, consider history conversation, especially the last user query, and use the tool.

#audio_tool = load_tools(["eleven_labs_text2speech"])
#tools.append(audio_tool)

agent = initialize_agent(
    tools,
    llm, 
    agent="conversational-react-description", #"conversational-react-description", 
    memory=memory, 
    verbose=True,
    max_iterations=5,
    return_intermediate_steps=True, ### from above
    handle_parsing_errors=True, ### from above
    ) 

##### Force choose at least one tool: https://python.langchain.com/v0.1/docs/modules/model_io/chat/function_calling/

agent.agent.llm_chain.prompt.template = """
    You are an agent who provides information to citizens on their situations and circumstances using one of the tools AND the conversation history.
    Your role is to provide the citizens with reliable and relevant information on what they should do, links to the official website where they can find further information, or contact information, or where they can submit neccesary application forms.    
    The initial assumption is that the user is located in Germany, Baden-Württemberg.
        
    To respond, use the following format:
    
    ```
    Question: the user's description of her situation or the user's question.
    Thought: Do I need to use a tool?
    Action: The action to take, should be one of [search_specific_webpages, search_urls_on_service_bw, simple_search_internet, wikipedia].
    Action Input: Input to the action, input to the tool.
    Observation: the result of the action.
    ... (this Thought/Action/Action Input/Observation can repeat up to N times before one of the tools provides a complete response)
    ```

    When you have a response to say to the Human, or if you do not need to use a tool, you must use the format:
    
    ```
    Thought: Do I need to use a tool? No, I now have an answer.
    AI: [final response here; if a tool delivered SOURCES, ALWAYS provide these SOURCES]

    ```
    
    If the user claims in the query (input) that the provided links are not correct, do not work, redirect to not existing pages, or error pages, or show Page was not found,
    that the links are not helpful or do not provide the information the user needs, are not relevant, etc.
    try to use the tool search_urls_on_service_bw 
    FOr this, You must then consider the conversation history, especially the last but one user query, and then use the tool.
    Consider the example:
        Input from the user: Wie kann ich ABC beantragen?
        Your previous response: <some link>.
        Input from the user: The link is incorrect.
        Thought: Do I need to use a tool? -- Yes, 
        Action: try to use the tool [search_urls_on_service_bw]
        Action input: Wie kann ich ABC beantragen?
    
    
    If the final response comes from the tool search_specific_webpages, use the tool output as your final response.
    
    If you do not have the answer after N attempts, reply with 'Es tut mir Leid, ich habe nicht genügend Informationen. Bitte spezifizieren Sie Ihre Anfrage oder besuchen Sie bitte: https://www.service-bw.de/zufi/lebenslagen .'
    Always include links to source pages (SOURCES) into your response. DO NOT generate links yoursef. Include all links provided by a tool.
    
    In your response, always use 'Sie' to address the user, keine 'Du'.
    The answer must be in the same language as the user's question, or input: if the user asks in German, reply in German.
    If possible, provide your response to the user in bullet points.

    Begin!

    Conversation history: {chat_history}
    
    Question = {input}
    
    Thought: {agent_scratchpad}
    """
    # For application forms, provide links to the website with the application form (Antrag).
    # search_specific_webpages, search_urls_on_service_bw, simple_search_googleapiwrapper, wikipedia, calculator
    # search_specific_webpages, search_urls_on_service_bw, 
# , calculator


##### Managing prompt size: https://python.langchain.com/v0.1/docs/expression_language/cookbook/prompt_size/

logging.basicConfig()
logging.getLogger("web_research").setLevel(logging.INFO)

if prompt := st.chat_input(placeholder="Ich bin arbeitslos. An wen muss ich mich wenden?"):
    st.chat_message("user").write(prompt)
    chat_history_mongodb.add_user_message(st.chat_message("user")) ### chat_history_mongodb
    #chat_history_mongodb.add_message(prompt) ### chat_history_mongodb
    
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        cfg = RunnableConfig()
        cfg["callbacks"] = [st_cb]
        response = agent(prompt)
        st.write(response["output"])
        #chat_history_mongodb.add_ai_message(response["output"]) ### chat_history_mongodb
        chat_history_mongodb.add_ai_message(response["output"]) ### chat_history_mongodb

# Ich bin arbeitslos, 45 Jahre alt, habe 3 Kinder (9 Monate alt, 3 und 12 Jahre alt), wohne getrennt von meinem Mann.

    #DuckDuckGoSearchRun(name="Search")
    # Tool(
    #     name="Wikidata-Search",
    #     func=wikidata.run,
    #     description="In Wikidata nach dem Begriff suchen. Muss benutzt werden, wenn nach einer ausfürhlichen Definition gefragt wird.",
    #     return_direct=True,        
    #     ),
    # Tool.from_function(
    #     name="Chat für BürgerInnen in schwieriger Situation",
    #     func=chat_chain.run, # ?? chatcompletionAPI
    #     return_direct=True,
    #     description= "Keine",
    #     # """Nützlich, um allegemeine Fragen zu beantworten, 
    #     # auf die keine Antwort mit anderen Tools gefunden wurden, oder
    #     # wenn es keine anderen Tools gibt.""",
    #     ),
    # Tool(
    #     name="Wikipedia-Suche", #Search
    #     func=docstore.search,
    #     description='Wikipedia durchsuchen.'#search wikipedia
    # ),
    # Tool(
    #     name="Lookup", #Lookup
    #     func=docstore.lookup,
    #     description='In Wikipedia nach dem Begriff suchen.' #lookup a term in wikipedia
    # ),



