import json
import pymongo
import streamlit as st

from urllib.parse import quote_plus
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

import getpass, os, pymongo, pprint
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel

# file_path2 = "./urls_to_look.json"
# with open(file_path2, "r") as file:
#     urls_to_look_json = file.read()

# urls_to_look = json.loads(urls_to_look_json)

# print(f"URL database: {urls_to_look}")

# all_predefined_urls = ["https://www.arbeitsagentur.de/datei/dok_ba022860.pdf",
#                        "https://www.arbeitsagentur.de/datei/dok_ba025855.pdf",
#                        "https://www.bmfsfj.de/resource/blob/142764/a37b6042c496933b6ced2ccd3a27c822/handreichung-fuer-die-beratung-alleinerziehender-broschuere-vamv-data.pdf",
#                        "https://www.bmfsfj.de/resource/blob/94936/83367b6dc5ebb96e5eee993b15c94ad2/prm-24375-broschure-elternzeit-data.pdf",
#                        #"https://www.bmfsfj.de/resource/blob/94106/%2000a03f47fcbe076829ad6403b919e93b/kinder-und-jugendhilfegesetz-sgb-viii-data.pdf",
#                        "https://www.arbeitsagentur.de/datei/dok_ba034925.pdf",
#                        "https://www.bmfsfj.de/resource/blob/94182/763244389dd4e093fa22d4788bbaddeb/kosten-betrieblich-unterstuetzter-kinderbetreuung-data.pdf",
#                        "https://www.bmfsfj.de/resource/blob/139908/72ce4ea769417a058aa68d9151dd6fd3/elterngeld-elterngeldplus-englisch-data.pdf",
#                        "https://www.bmfsfj.de/resource/blob/126698/2a988168756514804bf62756cd1e5507/fra-2018-fundamental-rights-report-2018-en-data.pdf",
#                        ]

# all_predefined_urls_json = json.dumps(all_predefined_urls)
# file_path = "./urls_to_look.json"
# with open(file_path, "w") as file:
#     file.write(all_predefined_urls_json)

mongodb_username = st.secrets["mongodb_username"]
mongodb_password = st.secrets["mongodb_password"]
encoded_username = quote_plus(mongodb_username)
encoded_password = quote_plus(mongodb_password)

##uri = f'mongodb://{encoded_username}:{encoded_password}@your_host:your_port/your_database'
mongodb_atlas_cluster_uri = f'mongodb+srv://{encoded_username}:{encoded_password}@clusterfree.xiknzbp.mongodb.net/?appName=clusterfree'


##uri = f'mongodb://{encoded_username}:{encoded_password}@your_host:your_port/your_database'
# uri = f'mongodb+srv://{encoded_username}:{encoded_password}@clusterfree.xiknzbp.mongodb.net/?appName=clusterfree'

#uri = "mongodb+srv://litvinovay:2024_chat%kernen@clusterfree.xiknzbp.mongodb.net/?appName=clusterfree"
# mongodb+srv://litvinovay:<password>@clusterfree.xiknzbp.mongodb.net/?retryWrites=true&w=majority&appName=clusterfree

# Create a new client and connect to the server
client = MongoClient(mongodb_atlas_cluster_uri, server_api=ServerApi('1'))

database = client["99_kernen"]
collection = database["99_chroma_db_kernen"]

# search_index_model = SearchIndexModel(
#     definition={
#         "fields": [
#             {
#             "type": "vector",
#             "numDimensions": 1024,
#             "path": "embedding", #"<fieldToIndex>",
#             "similarity": "cosine" #"euclidian | cosine | dotProduct"
#             },
#             # {
#             # "type": "filter",
#             # "path": "<fieldToIndex>",
#             # },

#         ]
#     },
#     name="vector_index2",
#     type="vectorSearch"
# )

vector_search_index = "vector_index"
# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

# result = collection.create_search_index(model=search_index_model)
# print(result)

loader = PyPDFLoader("https://www.arbeitsagentur.de/datei/au-pair-in-germany-en_ba030535.pdf")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1536, chunk_overlap=100)
docs = text_splitter.split_documents(data)
# Print the first document
print(docs[0])

vector_store = MongoDBAtlasVectorSearch.from_documents(
    documents = docs,
    embedding = OpenAIEmbeddings(disallowed_special=()),
    collection = collection,
    index_name = vector_search_index
)

query = "immigration"
results = vector_store.similarity_search(query)
print(results)

# ##########
# import os
# import sys
# import sqlite3

# #os.path.dirname(sys.executable)
# print(sqlite3.sqlite_version)
# Instantiate Atlas Vector Search as a retriever
retriever = vector_store.as_retriever(
   search_type = "similarity",
   search_kwargs = { "k": 10 }
)

# Define a prompt template
template = """

Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
"""
custom_rag_prompt = PromptTemplate.from_template(template)

llm = ChatOpenAI()

def format_docs(docs):
   return "\n\n".join(doc.page_content for doc in docs)

# Construct a chain to answer questions on your data
rag_chain = (
   { "context": retriever | format_docs, "question": RunnablePassthrough()}
   | custom_rag_prompt
   | llm
   | StrOutputParser()
)

# Prompt the chain
question = "How can I secure my MongoDB Atlas cluster?"
answer = rag_chain.invoke(question)

print("Question: " + question)
print("Answer: " + answer)

# Return source documents
documents = retriever.get_relevant_documents(question)
print("\nSource documents:")
pprint.pprint(documents)