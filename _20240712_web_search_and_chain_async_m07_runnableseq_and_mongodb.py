# -*- coding: utf-8 -*-
"""
@author: yulialitvinova

"""
### RESEARCH AUTOMATION (site):
    ##### https://python.langchain.com/v0.1/docs/use_cases/web_scraping/
    ##### https://blog.langchain.dev/automating-web-research/

import os
import logging
import json
import getpass
import asyncio
###########from langchain.retrievers.web_research import WebResearchRetriever
from web_research import WebResearchRetriever
#from langchain.chains import RetrievalQAWithSourcesChain
from langchain_core.runnables import RunnablePassthrough

import chromadb
##from langchain.vectorstores import Chroma
#from langchain_chroma import Chroma
from langchain_community.vectorstores import Chroma
#import faiss
#from langchain_community.vectorstores import FAISS 
#from langchain_community.docstore.in_memory import InMemoryDocstore  
from pymongo.mongo_client import MongoClient
#from pymongo import MongoClient
from pymongo.server_api import ServerApi
from urllib.parse import quote_plus
from langchain_mongodb import MongoDBAtlasVectorSearch
#from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_google_community import GoogleSearchAPIWrapper

#from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
#from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
import replicate
from transformers import LlamaModel, LlamaConfig

##from langchain_community.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    )

##from langchain import PromptTemplate
from langchain_core.prompts import PromptTemplate
#from langchain.prompts import PromptTemplate

#if __name__ == "__main__":

import streamlit as st
os.environ['OPENAI_API_KEY'] = st.secrets["openai_api_key"]
os.environ["OPEN_API_BASE"] = "https://api.openai.com/v1"
os.environ["GOOGLE_CSE_ID"] = st.secrets["google_cse_id"] 
os.environ["GOOGLE_API_KEY"] = st.secrets["google_api_key"]
#os.environ["MONGODB_ATLAS_CLUSTER_URI"] = st.secrets["mongodb_atlas_cluster_uri"]

mongodb_username = st.secrets["mongodb_username"]
mongodb_password = st.secrets["mongodb_password"]
encoded_username = quote_plus(mongodb_username)
encoded_password = quote_plus(mongodb_password)

##uri = f'mongodb://{encoded_username}:{encoded_password}@your_host:your_port/your_database'
mongodb_atlas_cluster_uri = f'mongodb+srv://{encoded_username}:{encoded_password}@clusterfree.xiknzbp.mongodb.net/?appName=clusterfree'
client = MongoClient(mongodb_atlas_cluster_uri, server_api=ServerApi('1'))

logging.basicConfig()
logging.getLogger("web_research").setLevel(logging.INFO)

embedding_size = 1536 # 1536 # 1024 
embeddings_model = OpenAIEmbeddings()  #OpenAIEmbeddings(disallowed_special=())
# index = faiss.IndexFlatL2(embedding_size)  
# vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

# vectorstore = Chroma(
#     embedding_function=embeddings_model, #OpenAIEmbeddings(), 
#     persist_directory="./99_chroma_db_kernen_test",
#     )

database = client["99_kernen"] #db_name = "99_kernen"
collection = database["99_kernen_general"] #collection_name = "99_chroma_db_kernen"
vector_search_index = "vector_index"
#mongodb_collection = client[db_name][collection_name]

try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

vectorstore = MongoDBAtlasVectorSearch( #).from_documents(
    #documents = docs,
    embedding = embeddings_model,
    collection = collection,
    index_name = vector_search_index
)

# LLM
temperature=0
llm = ChatOpenAI(
    temperature=temperature,
    model_name= "gpt-3.5-turbo",
    #"llama-7b-v2-chat", ##### https://blog.streamlit.io/how-to-build-a-llama-2-chatbot/ ##-v2-chat
    #"gpt-3.5-turbo-16k",
    streaming=True)

# Search
search = GoogleSearchAPIWrapper()

# file_path = "./url_database.json"
# with open(file_path, "r") as file:
#     url_database_json = file.read()
# url_database = json.loads(url_database_json)

# prompt_search = PromptTemplate(
#     input_variables=["question"],
#     template="""<<SYS>> \n You are an assistant tasked with improving Google search \
# results. \n <</SYS>> \n\n [INST] Generate THREE Google search queries that \
# are similar to this question. The output should be a numbered list of questions, \
# each should be focused on how the matter is dealt with in Germany, Baden-Württemberg, \
# and should have a question mark at the end: \n\n {question} [/INST]""",
# )
##    ChatHistory: {chat_history}
##    Follow Up Input: {question}

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1536, chunk_overlap=64) # 1024 # 1536

web_research_retriever = WebResearchRetriever.from_llm(
    vectorstore=vectorstore, ## vectorestore where webpages are stored
    llm=llm, ## should be llm_chain???
    search=search, ## Google Search API Wrapper
    num_search_results=3,
    text_splitter=text_splitter,
    #return_source_documents=True, ## unexpected argument
    #prompt=prompt_search,
    #predefined_urls=predefined_urls,
    #url_database=url_database,
    )

template_query="""<<SYS>> \n You are an assistant to citizens in difficult situations. \

        To answer the question, summarize the information in documents provided to you.
        Answer the question in the language of the question.\
        
        ALWAYS return a "SOURCES" part in your answer.\
        QUESTION: {question}\
        Documents to summarize: {summaries}\
        FINAL ANSWER: \
        SOURCES: {sources} \
        """

#{link}
        # "summaries"
        # {summaries} \

QA_CHAIN_PROMPT = PromptTemplate(
        template=template_query, input_variables=['question', 'summaries', 'sources'] # , "link"] #
    )
# prompt3 = PromptTemplate(template=template3, input_variables=["context", "question"])

##### https://api.python.langchain.com/en/latest/_modules/langchain/chains/qa_with_sources/retrieval.html
# qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
#     llm, 
#     chain_type="stuff",
#     retriever=web_research_retriever,
#     #return_source_documents=True, 
#     # ##### https://github.com/langchain-ai/langchain/issues/10575
#     # ##### https://api.python.langchain.com/en/latest/chains/langchain.chains.qa_with_sources.retrieval.RetrievalQAWithSourcesChain.html
#     #combine_prompt=prompt2,
#     #verbose=True,
#     chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
#     )

qa_chain = QA_CHAIN_PROMPT | llm#

# user_input = "Was sind die Voraussetzungen für Landesfamilienpass?"
# result = qa_chain.invoke({"question": user_input})
# print(result['answer'])
# #print(result["sources"])

# async def main():
#     user_input = "Was sind die Voraussetzungen für Landesfamilienpass?"
#     #result = 
#     qa_chain(user_input)#, run_manager=CallbackManagerForRetrieverRun())
#     #print(result)

##### https://stackoverflow.com/questions/49822552/python-asyncio-typeerror-object-dict-cant-be-used-in-await-expression
##### https://api.python.langchain.com/en/latest/_modules/langchain/chains/qa_with_sources/vector_db.html
##### https://github.com/langchain-ai/langchain/discussions/12193
async def result_response(user_input):
    returned_documents = web_research_retriever.invoke(user_input)#['page_content'] # {"query": user_input}
    #print(returned_documents)
    docs_for_summaries = []
    for document in returned_documents:
        docs_for_summaries.append(document.page_content)
        #print(f"*The next document*: {document.page_content}")
    
    sources_for_user = []
    for document in returned_documents:
        sources_for_user.append(document.metadata["source"])

    #print(docs_for_summaries)
    result = qa_chain.invoke({"question": user_input, "summaries": docs_for_summaries, "sources": sources_for_user})
    return result.content
    #return result["sources"]

async def main():
    user_input = "Welche Schularten gibt es?"
    result2 = await result_response(user_input)#, run_manager=CallbackManagerForRetrieverRun())
    print(result2)

##### https://stackoverflow.com/questions/45600579/asyncio-event-loop-is-closed-when-getting-loop
if __name__ == "__main__":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
##### https://github.com/langchain-ai/langchain/issues/10086