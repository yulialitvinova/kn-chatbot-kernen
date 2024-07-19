import logging
import re
from typing import List, Optional
import pymongo
import json
import os
import requests
from PyPDF2 import PdfReader

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

from langchain_community.document_loaders import AsyncHtmlLoader
#from langchain.document_loaders import  UnstructuredURLLoader ### to load predefined urls
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.document_transformers import Html2TextTransformer
#from llama_index.embeddings.openai import OpenAIEmbedding ### for similarity search by vector
from langchain_community.llms import LlamaCpp


#from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_google_community import GoogleSearchAPIWrapper
################from search import GoogleSearchAPIWrapper

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.language_models import BaseLLM
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter

from langchain.chains import LLMChain
from langchain.chains.prompt_selector import ConditionalPromptSelector

import asyncio
from urllib.parse import urlparse

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time

from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC


from langchain.docstore.document import Document

logger = logging.getLogger(__name__)


class SearchQueries(BaseModel):
    """Search queries to research for the user's goal."""

    queries: List[str] = Field(
        ..., description="List of search queries to look up on Google"
    )


DEFAULT_LLAMA_SEARCH_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""<<SYS>> \n You are an assistant tasked with improving Google search \
results. \n <</SYS>> \n\n [INST] Generate THREE Google search queries that \
are similar to this question. The output should be a numbered list of questions \
each should be focused on how the matter is dealt with in Germany, Baden-Württemberg, \
and each should have a question mark at the end: \n\n {question} [/INST]""",
)

DEFAULT_SEARCH_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an assistant tasked with improving Google search \
results. Generate THREE Google search queries that are similar to \
this question. The output should be a numbered list of questions and each \
each should be focused on how the matter is dealt with in Germany, Baden-Württemberg, \
should have a question mark at the end: {question}""",
)

##### https://api.python.langchain.com/en/latest/_modules/langchain/retrievers/web_research.html

class QuestionListOutputParser(BaseOutputParser[List[str]]):
    """Output parser for a list of numbered questions."""

    def parse(self, text: str) -> List[str]:
        lines = re.findall(r"\d+\..*?(?:\n|$)", text)
        return lines


class WebResearchRetriever(BaseRetriever):
    """`Google Search API` retriever."""

    ## Inputs
    vectorstore: VectorStore = Field(
        ..., description="Vector store for storing web pages"
    )
    llm_chain: LLMChain
    #predefined_urls: Optional[List[str]] = None ### ChatGPT
    search: GoogleSearchAPIWrapper = Field(..., description="Google Search API Wrapper")
    num_search_results: int = Field(1, description="Number of pages per Google search")
    text_splitter: TextSplitter = Field(
        RecursiveCharacterTextSplitter(chunk_size=1536, chunk_overlap=64), #1024 # 1536
        description="Text splitter for splitting web pages into chunks",
    )
    url_database: List[str] = Field(
        default_factory=list, description="List of processed URLs"
    )

    predefined_urls: List[str] = Field(
        default_factory=list, description="List of predefined URLs")

    @classmethod
    def from_llm(
        cls,
        vectorstore: VectorStore,
        llm: BaseLLM,
        search: GoogleSearchAPIWrapper,
        prompt: Optional[BasePromptTemplate] = None,
        num_search_results: int = 1,
        #predefined_urls: Optional[List[str]] = None, ### ChatGPT
        text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
            chunk_size=1536, chunk_overlap=64, #1024 # 1536
        ),
    ) -> "WebResearchRetriever":
        """Initialize from llm using default template.

        Args:
            vectorstore: Vector store for storing web pages
            llm: llm for search question generation
            search: GoogleSearchAPIWrapper
            prompt: prompt to generating search questions
            num_search_results: Number of pages per Google search
            text_splitter: Text splitter for splitting web pages into chunks

        Returns:
            WebResearchRetriever
        """

        if not prompt:
            QUESTION_PROMPT_SELECTOR = ConditionalPromptSelector(
                default_prompt=DEFAULT_SEARCH_PROMPT,
                conditionals=[
                    (lambda llm: isinstance(llm, LlamaCpp), DEFAULT_LLAMA_SEARCH_PROMPT)
                ],
            )
            prompt = QUESTION_PROMPT_SELECTOR.get_prompt(llm)

        ## Use chat model prompt
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            output_parser=QuestionListOutputParser(),
        )

        return cls(
            vectorstore=vectorstore,
            llm_chain=llm_chain,
            search=search,
            num_search_results=num_search_results,
            text_splitter=text_splitter,
            #predefined_urls=predefined_urls, ### ChatGPT
        )

    def clean_search_query(self, query: str) -> str:
        ## Some search tools (e.g., Google) will
        ## fail to return results if query has a
        ## leading digit: 1. "LangCh..."
        ## Check if the first character is a digit
        if query[0].isdigit():
            ## Find the position of the first quote
            first_quote_pos = query.find('"')
            if first_quote_pos != -1:
                ## Extract the part of the string after the quote
                query = query[first_quote_pos + 1 :]
                ## Remove the trailing quote if present
                if query.endswith('"'):
                    query = query[:-1]
        return query.strip()

    def search_tool(self, query: str, num_search_results: int = 1) -> List[dict]: ###AS async
        """Returns num_search_results pages per Google search."""
        query_clean = self.clean_search_query(query)
        result = self.search.results(query_clean, num_search_results) ###AS await
        return result    
   
    def load_and_index_urls(self, new_urls): ### this is a standard way for the new_urls TRUE, after google search
    #if new_urls:
        loader = AsyncHtmlLoader(new_urls, ignore_load_errors=True)
    ## TO-DO: list of URLs that could not be processed -> pass to the processing of unusual urls, resp. URLs loaded with java
        html2text = Html2TextTransformer()
        logger.info("Indexing new urls...")

        try:
            docs = loader.load()
            if docs:
                #print("Documents loaded.")
                logger.info("Documents loaded.")
                contains_letters = [
                        "service-bw",
                        "gesetze-im-internet",
                                ]
                for doc in docs:
                    doc_url = doc.metadata['source']
                    if any(letters in doc_url for letters in contains_letters):
                        self.predefined_urls.append(doc_url)
                    else:
                        self.url_database.append(doc_url)
        except Exception as e:
            #print(f"Error: {e}")
            logger.error(f"Error: {e}")
            return False    
        
        try:
            docs = list(html2text.transform_documents(docs))
            if docs:    
                #print("Documents transformed.")
                logger.info("Documents transformed.")
        except Exception as e:
            #print(f"Error: {e}")
            logger.error(f"Error: {e}")
            return False
        
        try:
            docs = self.text_splitter.split_documents(docs)
            if docs:
                #print("Documents split.")
                logger.info("Documents split.")
        except Exception as e:
            #print(f"Error: {e}")
            logger.error(f"Error: {e}")
            return False
        
        try:
            self.vectorstore.add_documents(docs)
            #self.url_database.extend(new_urls)
        except Exception as e:
            #print(f"Error: {e}")
            logger.error(f"Error: {e}")
            return False

    def find_predefined_urls(self,url_search): ### URL from which start to search
        #driver_path = "C:/Users/litvi/git/kn-chatbot-kernen/04_search_or_scrapping/chromedriver-win64"
        driver_path = "https://github.com/yulialitvinova/chatbot_kernen/tree/main/04_search_or_scrapping/chromedriver-win64"

        WINDOW_SIZE = "1920,1080"
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument(f"--window-size={WINDOW_SIZE}")

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.get(url_search)
        time.sleep(5)

        ## Open all "...weitere anzeigen" buttons when exist
        try:
            while True:
                see_more_button = WebDriverWait(driver, timeout=3).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, 'button[data-testid="pagination-button-load-more"]'))
                )       
                if see_more_button:
                    see_more_button.click()
                    time.sleep(2)  ### Wait for the new content to load
                else:
                    break
        except Exception as e:
            print(f"No more 'see more' buttons or an error occurred: {e}")

        page_source = driver.page_source
        #driver.quit()
        soup = BeautifulSoup(page_source, 'html.parser')

        parsed_url = urlparse(url_search)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"

        contains_keywords = ["lebenslagen/", "leistungen/"] #, "leistungen/"
        predefined_urls = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            #if 'lebenslagen/' in href:
            if any(keyword in href for keyword in contains_keywords):
                if href.startswith('/'):
                    href = base_url + href
                predefined_urls.append(href)

        return predefined_urls
        driver.quit()

    def extract_text_with_source(self, soup, url):
        text_with_source=[]
        
        include_prefix=['jnnorm', 'base-module_editorialContent_']
        ##### test script: 20240625_search_unusual_urls.py
        for element in soup.find_all(): 
            ## Flatten the list of classes and check each one against the prefixes
            classes = element.get('class', [])

            #include = any(cls.startswith(include_prefix) for cls in classes)
            include = any(any(cls.startswith(prefix) for prefix in include_prefix) for cls in classes)

            if include:
                text = element.get_text(strip=True)
                if text:
                    text_with_source.append((url, text))
        
        return text_with_source

    def load_one_of_predef_url_document(self, one_of_predef_url):
        #driver_path = "C:/Users/litvi/git/knowledge_chatbot_kernen/04_search_or_scrapping/chromedriver-win64"
        driver_path = "https://github.com/yulialitvinova/knowledge_chatbot_kernen/tree/main/04_search_or_scrapping/chromedriver-win64"
        
        WINDOW_SIZE = "1920,1080"
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument(f"--window-size={WINDOW_SIZE}")

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.get(one_of_predef_url)
        time.sleep(5)
        page_source = driver.page_source
        driver.quit()

        soup = BeautifulSoup(page_source, 'html.parser')

        ## Get the text from the document, using def--extract_text_with_source:
        text_with_source = self.extract_text_with_source(soup, one_of_predef_url)
        one_of_predef_url_document = [Document(page_content=text, metadata={"source": one_of_predef_url}) for one_of_predef_url, text in text_with_source]
        return one_of_predef_url_document
    
    def extract_pdf_text(self, one_of_url_pdf):
        try:
            response = requests.get(one_of_url_pdf)
            with open('temp.pdf', 'wb') as f:
                f.write(response.content)
            pdf_file = PdfReader('temp.pdf')
            text_data = ''
            for pg in pdf_file.pages:
                text_data += pg.extract_text()
            return text_data
        except Exception as e:
            print(f"Error extracting PDF text: {e}")
            return None
        finally:
            if os.path.exists('temp.pdf'):
                os.remove('temp.pdf')

    def _get_relevant_documents( ###AS async
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """Search Google for documents related to the query input.

        Args:
            query: user query

        Returns:
            Relevant documents from all various urls.
        """

        ## Get search questions
        logger.info("Generating questions for Google Search ...")
        result = self.llm_chain.invoke({"question": query}) ###AS await
        logger.info(f"Questions for Google Search (raw): {result}")
        questions = result["text"]
        logger.info(f"Questions for Google Search: {questions}")

        ## Get urls
        logger.info("Searching for relevant urls...")
        urls_to_look_incl_pdf = []
        urls_to_look_not_pdf = []
        urls_pdf = []
        #self.predefined_urls = [] # ["https://www.service-bw.de/zufi/lebenslagen/5001218"]

        file_path = "./url_database.json"
        with open(file_path, "r") as file:
            url_database_json = file.read()
        self.url_database = json.loads(url_database_json)

        for query in questions:
            ## Google search
            search_results = self.search_tool(query, self.num_search_results) ###AS await
            logger.info(f"Search results: {search_results}")
            for res in search_results:
                if res.get("link", None): #and res["link"] not in urls_to_look:
                    urls_to_look_incl_pdf.append(res["link"])

        ## Relevant urls        
        for new_url in urls_to_look_incl_pdf:
            if ".pdf" in new_url: #is_pdf(new_url):
                urls_pdf.append(new_url)
            else:
                urls_to_look_not_pdf.append(new_url)
        urls_to_look_not_pdf2 = set(urls_to_look_not_pdf)
        urls_pdf2 = set(urls_pdf)
        urls_to_look_incl_pdf2 = set(urls_to_look_incl_pdf)

        ## Check for any new urls that we have not processed
        urls_to_look_not_pdf3 = list(urls_to_look_not_pdf2.difference(self.url_database))

        if urls_to_look_not_pdf3:
            self.load_and_index_urls(urls_to_look_not_pdf3) ###AS await 

        url_database_pdf = []
        for url_pdf_in_database in self.url_database:
            if ".pdf" in url_pdf_in_database:
                url_database_pdf.append(url_pdf_in_database)

        url_database_pdf2 = set(url_database_pdf)
        urls_pdf3 = list(urls_pdf2.difference(url_database_pdf2))
        if urls_pdf3:
            documents = []
            for one_of_url_pdf in urls_pdf3:
                pdf_text = self.extract_pdf_text(one_of_url_pdf)
                # print(pdf_text)
                if pdf_text:
                    #print(f"pdf_text URL from {one_of_url_pdf}")
                    #doc = Document(page_content=pdf_text, metadata={"source": url})
                    doc = [Document(page_content=pdf_text, metadata={"source": one_of_url_pdf})] # for one_of_url_pdf, text in pdf_text]
                    #print(f"Documented created from URL: {one_of_url_pdf}")
                    doc = self.text_splitter.split_documents(doc)
                    documents.extend(doc)
                    #print(f"Text added from documents with URL: {one_of_url_pdf}")
                    self.url_database.append(one_of_url_pdf)
            self.vectorstore.add_documents(documents)

        # url_search = "https://www.service-bw.de/zufi/lebenslagen/5000312"
        # # ### ATTENTION: there is a script that allows to have a list of url_search
        # # ### url_search_list
        # further_urls = self.find_predefined_urls(url_search) ### uncomment this function to automatically write a list of predefined urls
        # if further_urls:
        #     for url in further_urls:
        #         self.predefined_urls.append(url)
        # self.predefined_urls.append(url_search)

        contains_letters = [
            "service_bw",
            "gesetze-im-internet",
        ]

        urls_to_look_updated = list(urls_to_look_incl_pdf2.difference(self.url_database))
        if urls_to_look_updated:
            for url in urls_to_look_updated:
                self.predefined_urls.append(url)
                #if any(letters in url for letters in contains_letters):
                    #self.predefined_urls.append(url)
        #print(f"Predefined URLs to load: {predefined_urls}")

        #print(f"Predefined URLs with URLs to try to upload again: {self.predefined_urls}")
        test_urls = [] # [] 
        #["https://www.service-bw.de/zufi/lebenslagen/5000312", "https://sozialministerium.baden-wuerttemberg.de/de/soziales/familie/leistungen/landesfamilienpass/", "https://www.service-bw.de/zufi/leistungen/173", "https://www.service-bw.de/zufi/lebenslagen/5000388", "https://www.service-bw.de/zufi/lebenslagen/5001175", "https://www.service-bw.de/zufi/lebenslagen/5001621", "https://www.service-bw.de/zufi/leistungen/995", "https://www.service-bw.de/zufi/lebenslagen/5000881", "https://www.service-bw.de/zufi/lebenslagen/5001154", "https://www.service-bw.de/zufi/lebenslagen/5000023", "https://www.service-bw.de/zufi/lebenslagen/5000940", "https://www.service-bw.de/zufi/lebenslagen/5001224", "https://www.service-bw.de/zufi/leistungen/1015", "https://www.service-bw.de/zufi/leistungen/179", "https://www.service-bw.de/zufi/leistungen/949", "https://www.service-bw.de/zufi/leistungen/2100", "https://www.service-bw.de/zufi/leistungen/1249", "https://www.service-bw.de/zufi/lebenslagen/5001342", "https://www.service-bw.de/zufi/leistungen/1536", "https://www.service-bw.de/zufi/lebenslagen/5000789", "https://www.service-bw.de/zufi/leistungen/380", "https://www.service-bw.de/zufi/lebenslagen/5000344", "https://www.service-bw.de/zufi/lebenslagen/5000652", "https://www.service-bw.de/zufi/lebenslagen/5000998", "https://www.service-bw.de/zufi/lebenslagen/5000710", "https://www.service-bw.de/zufi/lebenslagen/5001121", "https://www.service-bw.de/zufi/lebenslagen/5000433", "https://www.service-bw.de/zufi/lebenslagen/5001679", "https://www.service-bw.de/zufi/leistungen/343", "https://www.service-bw.de/zufi/leistungen/192", "https://www.service-bw.de/zufi/leistungen/930", "https://www.service-bw.de/zufi/leistungen/167", "https://www.service-bw.de/zufi/leistungen/165", "https://www.service-bw.de/zufi/lebenslagen/5000118", "https://www.service-bw.de/zufi/lebenslagen/5000123", "https://www.service-bw.de/zufi/leistungen/183", "https://www.service-bw.de/zufi/lebenslagen/5001266", "https://www.service-bw.de/zufi/leistungen/993", "https://www.service-bw.de/zufi/lebenslagen/5000457", "https://www.service-bw.de/zufi/leistungen/670", "https://www.service-bw.de/zufi/leistungen/1231", "https://www.service-bw.de/zufi/lebenslagen/5001056", "https://www.service-bw.de/zufi/lebenslagen/5000976", "https://www.service-bw.de/zufi/lebenslagen/5001130", "https://www.service-bw.de/zufi/leistungen/1363", "https://www.service-bw.de/zufi/lebenslagen/5000733", "https://www.service-bw.de/zufi/leistungen/112", "https://www.service-bw.de/zufi/leistungen/1558", "https://www.service-bw.de/zufi/lebenslagen/5001190", "https://www.service-bw.de/zufi/leistungen/853", "https://www.service-bw.de/zufi/leistungen/780", "https://www.service-bw.de/zufi/leistungen/751", "https://www.service-bw.de/zufi/lebenslagen/5000013", "https://www.service-bw.de/zufi/lebenslagen/5000783", "https://www.service-bw.de/zufi/leistungen/817", "https://www.service-bw.de/zufi/lebenslagen/5000349", "https://www.service-bw.de/zufi/leistungen/96", "https://www.service-bw.de/zufi/lebenslagen/5000600", "https://www.service-bw.de/zufi/leistungen/189", "https://www.service-bw.de/zufi/leistungen/127", "https://www.service-bw.de/zufi/leistungen/1232", "https://www.service-bw.de/zufi/leistungen/2260", "https://www.service-bw.de/zufi/lebenslagen/5001274", "https://www.service-bw.de/zufi/leistungen/929", "https://www.service-bw.de/zufi/leistungen/224", "https://www.service-bw.de/zufi/lebenslagen/5000019", "https://www.service-bw.de/zufi/leistungen/133", "https://www.service-bw.de/zufi/leistungen/150", "https://www.service-bw.de/zufi/lebenslagen/5001000", "https://www.service-bw.de/zufi/lebenslagen/5000814", "https://www.service-bw.de/zufi/leistungen/1439", "https://www.service-bw.de/zufi/lebenslagen/5000737", "https://www.service-bw.de/zufi/leistungen/162", "https://www.service-bw.de/zufi/leistungen/1366", "https://www.service-bw.de/zufi/leistungen/6000953", "https://www.service-bw.de/zufi/leistungen/197", "https://www.service-bw.de/zufi/leistungen/93", "https://www.service-bw.de/zufi/lebenslagen/5001318", "https://www.service-bw.de/zufi/lebenslagen/5001604", "https://www.service-bw.de/zufi/leistungen/1560", "https://www.service-bw.de/zufi/leistungen/1658", "https://www.service-bw.de/zufi/leistungen/1962", "https://www.service-bw.de/zufi/lebenslagen/5000120", "https://www.service-bw.de/zufi/leistungen/599", "https://www.service-bw.de/zufi/leistungen/1557", "https://www.service-bw.de/zufi/leistungen/2165", "https://www.service-bw.de/zufi/leistungen/287", "https://www.service-bw.de/zufi/leistungen/1480", "https://www.service-bw.de/zufi/leistungen/1650", "https://www.service-bw.de/zufi/leistungen/1553", "https://www.service-bw.de/zufi/leistungen/154", "https://www.service-bw.de/zufi/lebenslagen/5000674", "https://www.service-bw.de/zufi/lebenslagen/5000616", "https://www.service-bw.de/zufi/lebenslagen/5000811", "https://www.service-bw.de/zufi/leistungen/185", "https://www.service-bw.de/zufi/leistungen/164", "https://www.service-bw.de/zufi/leistungen/1956", "https://www.service-bw.de/zufi/lebenslagen/5000343", "https://www.service-bw.de/zufi/leistungen/108"] #" https://www.gesetze-im-internet.de/englisch_gg/"
        if test_urls:
            for test_url in test_urls:
                self.predefined_urls.append(test_url)   
        #self.predefined_urls.append(test_urls)
        predefined_urls = set(self.predefined_urls)
        logger.info(f"Predefined URLs, not duplicated: {predefined_urls}")
        predefined_urls_to_load = list(predefined_urls.difference(self.url_database))
        #print(f"Predefined URLs, after check that not in the URL database: {predefined_urls_to_load}")
        if predefined_urls_to_load:
            try:
                documents = []
                for one_of_predef_url in predefined_urls_to_load:
                    #logger.info(f"Loading text from one of the predefined url-s: {one_of_predef_url}")
                    documents.extend(self.load_one_of_predef_url_document(one_of_predef_url))
                    self.url_database.append(one_of_predef_url)
                if documents:
                    self.vectorstore.add_documents(documents)
                #self.url_database.extend(new_urls)
            except Exception as e:
                logger.error(f"Error: {e}")
                return False
        #print(f"New URL database: {self.url_database}")

        url_database_json = json.dumps(self.url_database)
        file_path = "./url_database.json"
        with open(file_path, "w") as file:
            file.write(url_database_json)
            
        ## Search for relevant splits
        ## TODO: make this async
        logger.info("Grabbing most relevant splits from urls...")
        docs = []
        for query in questions:
            #docs.extend(self.vectorstore.similarity_search(query)) ###AS await
            #docs.extend(self.vectorstore.similarity_search_with_relevance_scores(query))
            docs.extend(self.vectorstore.max_marginal_relevance_search(query, k=3, lambda_mult=0.5)) ### optimize for similartiy among selected documents
            # ### embeddings need to be added; the same for similarity_search_by_vector
            
            # docs.extend(self.vectorstore.max_marginal_relevance_search_by_vector(
            #     embedding=OpenAIEmbedding(),
            #     lambda_mult=0.5, k=3))

        ## Get unique docs
        logger.info("Getting unique documents...")
        # unique_documents_dict = {
        #     (doc.page_content, tuple(sorted(doc.metadata.items()))): doc for doc in docs
        #     }
        # unique_documents_dict = {}
        # for doc in docs:
        #     source = doc.source #["source"] #doc.get('source', '')
        #     text = doc.text #["text"] #doc.get('text', '')
        #     key = (source, text)
        #     unique_documents_dict[key] = doc


        unique_documents = docs #list(unique_documents_dict.values())
        logger.info(f"Number of unique documents: {len(unique_documents)}")

        return unique_documents

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> List[Document]:
        raise NotImplementedError

