import os
import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from langchain.docstore.document import Document
import time
from urllib.parse import urlparse

from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC

from langchain_openai import OpenAIEmbeddings
#from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from transformers import AutoTokenizer
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter

import pymongo
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from urllib.parse import quote_plus
from langchain_mongodb import MongoDBAtlasVectorSearch

os.environ['OPENAI_API_KEY'] = st.secrets["openai_api_key"]


os.environ['OPENAI_API_KEY'] = st.secrets["openai_api_key"]
os.environ["OPEN_API_BASE"] = "https://api.openai.com/v1"
os.environ["GOOGLE_CSE_ID"] = st.secrets["google_cse_id"] 
os.environ["GOOGLE_API_KEY"] = st.secrets["google_api_key"]

mongodb_username = st.secrets["mongodb_username"]
mongodb_password = st.secrets["mongodb_password"]
encoded_username = quote_plus(mongodb_username)
encoded_password = quote_plus(mongodb_password)

##uri = f'mongodb://{encoded_username}:{encoded_password}@your_host:your_port/your_database'
mongodb_atlas_cluster_uri = f'mongodb+srv://{encoded_username}:{encoded_password}@clusterfree.xiknzbp.mongodb.net/?appName=clusterfree'
client = MongoClient(mongodb_atlas_cluster_uri, server_api=ServerApi('1'))


try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

temperature=0
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo", ##### model für Deutsch!
    #"a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea", 
    #"llama-7b-v2-chat" ##### https://blog.streamlit.io/how-to-build-a-llama-2-chatbot/
    #"LeoLM/leo-hessianai-13b", #"gpt-3.5-turbo", #"text_davinci-003", #"BAAI/bge-small-en-v1.5"
    streaming = True,
    temperature=temperature,
    )

embeddings_model = OpenAIEmbeddings()
embedding_size = 200
text_splitter_urls_sugg = RecursiveCharacterTextSplitter(chunk_size=embedding_size, chunk_overlap=64)
# vectorstore_urls_sugg = Chroma(
#     embedding_function=embeddings_model, persist_directory="./99_chroma_db_urls")

database_url = client["99_kernen"]
collection_url = database_url["99_kernen_urls"]
vector_search_index_url = "vector_index"

vectorstore_urls_sugg = MongoDBAtlasVectorSearch(
    embedding=embeddings_model,
    collection=collection_url,
    index_name=vector_search_index_url
    )

#####################################################
# # Path to the chromedriver executable
# #driver_path = 'path/to/chromedriver'  # Replace with the actual path to chromedriver
# ### https://developer.chrome.com/docs/chromedriver/downloads
# ### https://googlechromelabs.github.io/chrome-for-testing/
# ### https://katekuehl.medium.com/installation-guide-for-google-chrome-chromedriver-and-selenium-in-a-python-virtual-environment-e1875220be2f
# driver_path = "/04_search_or_scrapping/chromedriver-win64"
# #browser_path = "C:/Users/litvi/knowledge/knowledge_chatbot-1/04_search_or_scrapping/chromedriver-win64"
# # URL to scrape
# url_search = "https://www.service-bw.de/zufi/lebenslagen/5000312"

# # Initialize the Selenium WebDriver
# #options = webdriver.ChromeOptions()
# #driver = webdriver.Chrome(options=options)
# #driver = webdriver.Chrome(browser_path)

# # Setup Chrome options for headless mode
# ### https://www.selenium.dev/documentation/webdriver/browsers/chrome/
# WINDOW_SIZE = "1920,1080"
# chrome_options = Options()
# chrome_options.add_argument("--headless")
# #chrome_options.add_argument("--disable-gpu")
# #chrome_options.add_argument("--no-sandbox")
# #chrome_options.add_argument("--disable-dev-shm-usage")

# ### https://stackoverflow.com/questions/16180428/can-selenium-webdriver-open-browser-windows-silently-in-the-background
# ### https://stackoverflow.com/questions/16180428/can-selenium-webdriver-open-browser-windows-silently-in-the-background/48775203#48775203
# chrome_options.add_argument(f"--window-size={WINDOW_SIZE}")

# service = Service(ChromeDriverManager().install())
# driver = webdriver.Chrome(service=service, options=chrome_options)

# # Open the webpage
# driver.get(url_search)

# # Wait for JavaScript to load the content
# time.sleep(5)  # Adjust the sleep time as necessary

# ## Open all "...weitere anzeigen" buttons when exist
# try:
#     while True:
#         see_more_button = WebDriverWait(driver, timeout=3).until(
#             EC.element_to_be_clickable((By.CSS_SELECTOR, 'button[data-testid="pagination-button-load-more"]'))
#         )       
#         if see_more_button:
#             see_more_button.click()
#             time.sleep(2)  # Wait for the new content to load
#         else:
#             break
# except Exception as e:
#     print(f"No more 'see more' buttons or an error occurred: {e}")

# # Get the page source after JavaScript has executed
# page_source = driver.page_source

# # Close the browser
# driver.quit()

# # Parse the page source with BeautifulSoup
# soup = BeautifulSoup(page_source, 'html.parser')

# # Extract and print all the links
# # for link in soup.find_all('a', href=True):
# #     print(link['href'])

# parsed_url = urlparse(url_search)
# base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"

# documents = []
# urls = []
# contains_keywords = ["lebenslagen/", "leistungen/"]
# ### ATTENTION: does not work without block for "see more" buttons-unroll
# ### see script 04_search_or_scrapping/_20240624_extract_urls_for_leistungen.py

# for link in soup.find_all('a', href=True):
#     href = link['href']
#     link_text = link.get_text(strip=True)
#     #if 'lebenslagen/' in href:
#     if any(keyword in href for keyword in contains_keywords):
#         if href.startswith('/'):
#             href = base_url + href
#         urls.append((link_text, href))
#         url_for_docs = [Document(page_content=link_text, metadata={"source": href})]
#         documents.extend(url_for_docs)
# print(len(urls))
# print(len(documents))
# print(documents[0])

# documents = text_splitter_urls_sugg.split_documents(documents)
# vectorstore_urls_sugg.add_documents(documents)
##############################################

# for link in soup.find_all('a', href=True):
#     href = link['href']
#     if 'leistungen/' in href:
#         if href.startswith('/'):
#             href = base_url + href
#         urls.append(href)

#urls.sort(key = lambda x: x[1])
# url_count = len(urls)
# print(f"Number of URLs: {url_count}")

# for url in urls:
#     print(url)

template_query_urls_sugg="""<<SYS>> \n You are an assistant to citizens in difficult situations. \

        You need to find the relevant url (i.e., Document(metadata=source)) in retrieved DOCUMENTS, based on the Document(page_content) extracted above.\
        
        If you do not know the answer reply with 'Es tut mir Leid, ich habe nicht genügend Informationen'.\
        Provide only a list of URLs as your FINAL ANSWER.\
        
        QUESTION: {question}\
        Retrieved documents: {summaries}\
        FINAL ANSWER: \
        SOURCES: \
        """
#{source_documents}
#SOURCE: {source_documents}
#ALWAYS return a "SOURCES" part in your answer.\

QA_CHAIN_PROMPT_urls_sugg = PromptTemplate(template=template_query_urls_sugg, 
                         input_variables=["summaries", "question"])
#prompt3 = PromptTemplate.from_template(template3)


# qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
#     llm, 
#     chain_type="stuff",
#     #retriever=web_research_retriever,
#     retriever=vectorstore_urls_sugg.as_retriever(search_kwargs={"k": 2}),
#     #return_source_documents=True,
#     #combine_prompt=prompt2,
#     verbose=True,
#     chain_type_kwargs={"prompt": QA_CHAIN_PROMPT_urls_sugg}
#     )

user_input = "Landesfamilienpass"
retriever=vectorstore_urls_sugg.as_retriever(search_kwargs={"k": 2}, )
returned_documents = retriever.invoke(user_input) # get_relevant_documents(user_input)
#print(returned_documents)
qa_chain = QA_CHAIN_PROMPT_urls_sugg | llm


result = qa_chain.invoke({"question": user_input, "summaries": returned_documents})
#print(result["answer"])
print(result.content)