# --- MCP Server: Services Module ---
# This file contains the core logic of the agent, including its tools,
# reasoning prompt, and all helper functions for interacting with external services.

import os
import getpass
import time
import json
import requests
import re
import logging
import asyncio
from typing import Optional, List
from datetime import datetime, timedelta

# --- Core LangChain & Pydantic Imports ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# --- Modern, Modular LangChain Integration Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_community.tools import TavilySearchResults

# --- Third-party imports for fallback search ---
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import CrawlerRunConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy


# --- Global Configuration ---
PINECONE_INDEX_NAME = "mcp-documentation-index"
JINA_READER_API_URL = "https://r.jina.ai/"
MIN_CONTENT_LENGTH_THRESHOLD = 500

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Pydantic Schema for Rich Data Extraction (UNCHANGED) ---
class LibraryInfo(BaseModel):
    """A structured representation of key information about a software library."""
    library_name: Optional[str] = Field(description="The official name of the library (e.g., 'scikit-learn')")
    package_name: Optional[str] = Field(description="The correct, current package name for installation (e.g., 'scikit-learn')")
    latest_version: Optional[str] = Field(description="The latest stable version number found (e.g., '1.5.0')")
    documentation_url: Optional[str] = Field(description="The direct URL to the main documentation page that was processed.")
    summary: Optional[str] = Field(description="A brief, one-sentence summary of the library's primary purpose.")
    installation_command: Optional[str] = Field(description="The standard installation command (e.g., 'pip install scikit-learn')")
    deprecation_notice: Optional[str] = Field(description="A notice if the original library name is deprecated or has been renamed.")
    ecosystem: Optional[str] = Field(description="The software ecosystem the library belongs to (e.g., 'Python/PyPI', 'JavaScript/npm')")
    confidence_score: Optional[str] = Field(description="Self-evaluated confidence level: 'High', 'Medium', or 'Low'.")
    additional_insights: Optional[str] = Field(description="Important information discovered during low-confidence rescraping.")

# --- Service Initialization, Caching, Confidence Score (UNCHANGED) ---
def load_api_keys():
    """Securely loads API keys, prompting the user if they are not set."""
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API Key: ")
    if "PINECONE_API_KEY" not in os.environ:
        os.environ["PINECONE_API_KEY"] = getpass.getpass("Enter your Pinecone API Key: ")

def initialize_services():
    """Initializes and returns the core AI and database clients."""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    pc = Pinecone()
    return llm, embeddings_model, pc

def _sanitize_filename(name: str) -> str:
    """Replaces characters that are invalid in file names."""
    return re.sub(r'[\\/*?:"<>|]', "_", name)

def save_to_cache(library_name: str, library_data):
    """Saves library data to a local cache file with timestamp."""
    cache_dir = os.path.join("/tmp", "cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    safe_filename = _sanitize_filename(library_name)
    cache_file = os.path.join(cache_dir, f"{safe_filename}_cache.json")

    if hasattr(library_data, 'model_dump'):
        data_dict = library_data.model_dump()
    else:
        data_dict = library_data

    cache_entry = {
        "timestamp": datetime.now().isoformat(),
        "data": data_dict
    }

    try:
        with open(cache_file, 'w') as f:
            json.dump(cache_entry, f, indent=2)
        logger.info(f"💾 Cached data for '{library_name}'")
    except Exception as e:
        logger.error(f"⚠️ Failed to cache data for '{library_name}': {e}")

def load_from_cache(library_name: str, max_age_hours: int = 24):
    """Loads library data from cache if it exists and is not expired."""
    cache_dir = os.path.join("/tmp", "cache")
    safe_filename = _sanitize_filename(library_name)
    cache_file = os.path.join(cache_dir, f"{safe_filename}_cache.json")

    if not os.path.exists(cache_file):
        return None

    try:
        with open(cache_file, 'r') as f:
            cache_entry = json.load(f)

        cache_time = datetime.fromisoformat(cache_entry["timestamp"])
        if datetime.now() - cache_time > timedelta(hours=max_age_hours):
            logger.warning(f"⚠️ Cache expired for '{library_name}' (older than {max_age_hours} hours)")
            return None

        logger.info(f"📋 Cache hit for '{library_name}' (age: {datetime.now() - cache_time})")
        return cache_entry["data"]
    except Exception as e:
        logger.error(f"⚠️ Failed to load cache for '{library_name}': {e}")
        return None

def interpret_confidence_score(confidence_score: str) -> str:
    """Provides a human-readable interpretation of the confidence score."""
    if not confidence_score or not confidence_score.strip():
        return "❓ Unknown Confidence Level"

    confidence_lower = confidence_score.lower()
    if confidence_lower == "high":
        return "✅ High Confidence - Data is comprehensive and reliable"
    elif confidence_lower == "medium":
        return "⚠️ Medium Confidence - Data is good but may have some uncertainties"
    elif confidence_lower == "low":
        return "❌ Low Confidence - Data is limited or potentially unreliable"
    else:
        return f"❓ Unknown Confidence Level: {confidence_score}"


# --- Specialist Agent Tools and Web Search (UNCHANGED) ---
@tool
def pypi_api_tool(package_name: str) -> str:
    """Queries the official PyPI API for a Python package's metadata."""
    print(f" 	-> 🛠️ Using PyPI API Tool for '{package_name}'...")
    try:
        response = requests.get(f"https://pypi.org/pypi/{package_name}/json")
        response.raise_for_status()
        info = response.json().get("info", {})
        urls = info.get("project_urls", {})
        doc_url = urls.get("Homepage") or urls.get("Documentation") or next((url for url in urls.values() if url), None)
        return f"Success! Found documentation URL: {doc_url}" if doc_url else "PyPI API found the package but no documentation URL."
    except requests.exceptions.HTTPError as e:
        return f"Error: Package '{package_name}' not found on PyPI." if e.response.status_code == 404 else f"Error: HTTP error: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

@tool
def npm_api_tool(package_name: str) -> str:
    """Queries the official npm registry API for a JavaScript/Node.js package's metadata."""
    print(f" 	-> 🛠️ Using npm API Tool for '{package_name}'...")
    try:
        response = requests.get(f"https://registry.npmjs.org/{package_name}")
        response.raise_for_status()
        data = response.json()
        doc_url = data.get("homepage") or (data.get("bugs", {}).get("url"))
        return f"Success! Found documentation URL: {doc_url}" if doc_url else "npm API found the package but no documentation URL."
    except requests.exceptions.HTTPError as e:
        return f"Error: Package '{package_name}' not found on npm." if e.response.status_code == 404 else f"Error: HTTP error: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

@tool
def crates_io_api_tool(package_name: str) -> str:
    """Queries the official Crates.io API for a Rust package's metadata."""
    print(f" 	-> 🛠️ Using Crates.io API Tool for '{package_name}'...")
    try:
        response = requests.get(f"https://crates.io/api/v1/crates/{package_name}")
        response.raise_for_status()
        data = response.json()
        doc_url = data.get("crate", {}).get("documentation")
        return f"Success! Found documentation URL: {doc_url}" if doc_url else "Crates.io API found the package but no documentation URL."
    except requests.exceptions.HTTPError as e:
        return f"Error: Package '{package_name}' not found on Crates.io." if e.response.status_code == 404 else f"Error: HTTP error: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

@tool
def web_search_tool(query: str) -> str:
    """Performs a web search using DuckDuckGo first, then falls back to Tavily AI if no valid URLs are found."""
    print(f" 	-> 🛠️ Using Web Search Tool for '{query}'...")

    try:
        print(" 	-> 🔍 Attempting DuckDuckGo search...")
        wrapper = DuckDuckGoSearchAPIWrapper(time="y", max_results=5, region="us-en")
        ddg_results = wrapper.run(query)
        if ddg_results and any(url in ddg_results for url in ['http://', 'https://']):
            print(" 	-> ✅ DuckDuckGo search successful!")
            return ddg_results
        print(" 	-> ⚠️ DuckDuckGo found no valid URLs, falling back to Tavily AI...")
    except Exception as e:
        print(f" 	-> ❌ DuckDuckGo search failed: {e}, falling back to Tavily AI...")

    try:
        if not os.getenv("TAVILY_API_KEY"):
            return "Error: Tavily API key not found. Please set TAVILY_API_KEY."
        print(" 	-> 🤖 Using LangChain-aware Tavily AI as fallback...")
        tavily_tool = TavilySearchResults(max_results=5)
        results = tavily_tool.invoke({"query": query})
        print(" 	-> ✅ Tavily AI search successful!")
        return str(results)
    except Exception as e:
        print(f" 	-> ❌ Tavily AI search also failed: {e}")
        return f"Both DuckDuckGo and Tavily AI searches failed. Error: {e}"

@tool
def smart_web_search_with_retry(query: str, original_results: str = None) -> str:
    """
    Smart web search that automatically retries with alternative strategies when
    the original results indicate broken links, 404 errors, or invalid documentation.
    """
    print(f" 	-> 🔄 Smart web search with retry for '{query}'...")

    needs_retry = False
    if original_results:
        url_problem_indicators = [
            "url might not be the actual documentation", "404 error", "not found",
            "broken link", "invalid url", "page not available", "link returns a 404",
            "wrong documentation", "incorrect url", "misleading link"
        ]
        if any(indicator.lower() in original_results.lower() for indicator in url_problem_indicators):
            needs_retry = True

    if needs_retry:
        print(" 	-> ⚠️ URL/documentation problems detected. Attempting alternative search strategies...")
        if not os.getenv("TAVILY_API_KEY"):
            return "Error: Tavily API key not found for retry. Please set TAVILY_API_KEY."

        tavily_tool = TavilySearchResults(max_results=5)
        retry_strategies = [
            f"{query} documentation official",
            f"{query} developer guide api reference",
            f"{query} API guide tutorial examples",
            f"{query} GitHub source code repository"
        ]

        for i, strategy_query in enumerate(retry_strategies):
            print(f" 	-> 🔍 Retry Strategy {i+1}: '{strategy_query}'")
            try:
                results = tavily_tool.invoke({"query": strategy_query})
                if results:
                    print(" 	-> ✅ Alternative search successful!")
                    return f"Alternative search results for '{query}':\n\n" + str(results)
            except Exception as e:
                print(f" 	-> ❌ Strategy {i+1} failed: {e}")

        return "All retry strategies failed to find a working URL."
    else:
        print(" 	-> ✅ No URL/documentation problems detected. Using normal web search.")
        return web_search_tool(query)

# --- Agent Creation (UNCHANGED) ---
def create_universal_agent(llm):
    """Creates the multi-ecosystem agent with the most robust reasoning process."""
    tools = [pypi_api_tool, npm_api_tool, crates_io_api_tool, smart_web_search_with_retry]

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert documentation research assistant... Your goal is to find the single, best, official, content-rich documentation URL..."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


# --- Data Pipeline Functions ---

def ensure_pinecone_index_ready(pc: Pinecone, embeddings_model):
    """Checks if the Pinecone index exists. If not, creates it and waits until it is ready."""
    logger.info(f"Ensuring Pinecone index '{PINECONE_INDEX_NAME}' is ready...")
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        logger.info(f"Index not found. Creating new index '{PINECONE_INDEX_NAME}'...")
        try:
            dimension = len(embeddings_model.embed_query("sample text"))
            pc.create_index(
                name=PINECONE_INDEX_NAME, dimension=dimension, metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            while not pc.describe_index(PINECONE_INDEX_NAME).status['ready']:
                logger.info("Waiting for index to become ready...")
                time.sleep(5)
            logger.info("✅ Index is ready.")
        except Exception as e:
            logger.error(f"❌ Error creating Pinecone index: {e}")
            raise
    else:
        logger.info("✅ Index already exists and is ready.")

def _fetch_with_jina(url: str) -> Optional[str]:
    """Fetches clean content using the Jina AI Reader API."""
    try:
        api_url = f"{JINA_READER_API_URL}{url}"
        logger.info(f"🧼 [Primary] Attempting to fetch content with Jina AI from: {url}")
        response = requests.get(api_url, timeout=30, headers={"User-Agent": "LibrarianAI/1.0"})
        response.raise_for_status()
        logger.info("✅ [Primary] Successfully fetched content with Jina AI.")
        return response.text
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ [Primary] Jina AI API request failed: {e}")
        return None

async def _deep_crawl_with_crawl4ai(url: str) -> Optional[str]:
    """
    Performs a deep crawl starting from the given URL to gather rich context
    from multiple linked pages using the latest Crawl4ai method.
    """
    try:
        logger.info(f"🤖 [Deep Crawl] Initiating deep crawl for: {url}")

        user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
        )
        crawler_params = {"user_agent": user_agent, "headless": True}

        # 1. Define the deep crawl strategy
        strategy = BFSDeepCrawlStrategy(max_depth=1) # Crawl the starting page + 1 level of links

        # 2. Configure the run to use the deep crawl strategy
        run_config = CrawlerRunConfig(
            deep_crawl_strategy=strategy,
            markdown_generator=DefaultMarkdownGenerator()
        )

        all_content = []
        async with AsyncWebCrawler(**crawler_params) as crawler:
            # 3. Use the standard arun() method. It will return a list of results.
            results = await crawler.arun(url=url, config=run_config)
            
            # 4. Iterate through the list of results from the crawl
            if results:
                for result in results:
                    if result.success and result.markdown:
                        logger.info(f"  -> Crawled and extracted content from: {result.url}")
                        all_content.append(str(result.markdown))
        
        if not all_content:
            logger.error("❌ Deep crawl finished but found no markdown content.")
            return None
        
        logger.info(f"✅ Deep crawl successful. Aggregated content from {len(all_content)} pages.")
        return "\n\n--- (New Page Content) ---\n\n".join(all_content)

    except Exception as e:
        logger.error(f"❌ Deep crawl failed with a critical error.")
        import traceback
        traceback.print_exc()
        return None
def get_clean_content(doc_url: str) -> Optional[str]:
    """
    Fetches clean content from a single URL using Jina AI.
    """
    return _fetch_with_jina(doc_url)

def ingest_documentation(library_name: str, doc_url: str, embeddings_model, content_to_ingest: Optional[str] = None):
    """
    Deletes old vectors for the library, then gets clean content and stores the new vectors.
    """
    logger.info(f"\n📚 Starting ingestion for '{library_name}'...")
    
    doc_id = f"lib-{_sanitize_filename(library_name)}"
    
    try:
        logger.info(f"🧹 Deleting old vectors with doc_id: '{doc_id}'...")
        vectorstore = PineconeVectorStore.from_existing_index(PINECONE_INDEX_NAME, embeddings_model)
        vectorstore.delete(filter={"doc_id": doc_id})
        logger.info("✅ Old vectors deleted successfully.")
    except Exception as e:
        logger.warning(f"⚠️ Could not delete old vectors (they may not exist): {e}")

    if content_to_ingest is None:
        content_string = get_clean_content(doc_url)
    else:
        content_string = content_to_ingest

    if not content_string:
        logger.error("❌ Halting ingestion as no content was provided or could be fetched.")
        return False

    try:
        base_doc = Document(page_content=content_string, metadata={"source": doc_url})
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_documents([base_doc])
        
        for split in splits:
            split.metadata["doc_id"] = doc_id

        logger.info(f"Embedding {len(splits)} new document chunks and storing in Pinecone...")
        PineconeVectorStore.from_documents(splits, embeddings_model, index_name=PINECONE_INDEX_NAME)
        logger.info("✅ New content ingested successfully.")
        return True
    except Exception as e:
        logger.error(f"❌ An error occurred during chunking or embedding: {e}")
        return False

def extract_structured_info(library_name: str, llm, embeddings_model, doc_url: str = None) -> Optional[LibraryInfo]:
    """Retrieves context from Pinecone and uses an LLM to extract rich, structured information."""
    logger.info(f"\n⛏️ Extracting structured info for '{library_name}' using RAG...")
    try:
        ingestion_success = ingest_documentation(library_name, doc_url, embeddings_model)
        if not ingestion_success:
            return None

        vectorstore = PineconeVectorStore.from_existing_index(PINECONE_INDEX_NAME, embeddings_model)
        retriever = vectorstore.as_retriever(search_kwargs={'k': 8})
        
        docs = retriever.invoke(f"Information about {library_name} including purpose, installation, and version.")
        context_text = "\n\n".join([doc.page_content for doc in docs])

        if not context_text:
            logger.warning(f"⚠️ Could not retrieve any context for '{library_name}' from Pinecone.")
            return LibraryInfo(library_name=library_name, confidence_score="Low")

        structured_llm = llm.with_structured_output(LibraryInfo)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at extracting key information..."),
            ("human", "Extract the required information about '{topic}'... Context:\n{context}")
        ])

        chain = prompt | structured_llm
        response = chain.invoke({"topic": library_name, "context": context_text})

        if response and doc_url:
            response.documentation_url = doc_url

        # --- CORRECTED: A more robust "allow-list" check for confidence ---
        # The fallback will now trigger unless the score is explicitly "High" or "Medium".
        is_confident = response.confidence_score and response.confidence_score.lower() in ["high", "medium"]

        if doc_url and not is_confident:
            logger.warning(f"⚠️ Low or Unknown confidence for '{library_name}'. Triggering deep crawl fallback...")
            
            rich_content = asyncio.run(_deep_crawl_with_crawl4ai(doc_url))
            
            if rich_content:
                reingestion_success = ingest_documentation(library_name, doc_url, embeddings_model, content_to_ingest=rich_content)

                if reingestion_success:
                    logger.info("Re-running structured extraction with richer context from deep crawl...")
                    new_docs = retriever.invoke(f"Information about {library_name} including purpose, installation, and version.")
                    new_context_text = "\n\n".join([doc.page_content for doc in new_docs])
                    new_response = chain.invoke({"topic": library_name, "context": new_context_text})
                    
                    if new_response:
                        new_response.documentation_url = doc_url
                        new_response.additional_insights = (
                            "Low confidence triggered a deep crawl to improve data quality.\n"
                            + (new_response.additional_insights or "")
                        )
                    
                    logger.info("✅ Structured information extracted successfully after deep crawl.")
                    return new_response
            else:
                logger.error("❌ Deep crawl fallback failed to produce content. Returning initial low-confidence result.")

        logger.info("✅ Structured information extracted successfully.")
        return response
    except Exception as e:
        logger.error(f"❌ Error during structured data extraction: {e}")
        return None
