---
title: Librarian MCP Server
emoji: 📚
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# Librarian - AI-Powered Documentation Intelligence System

> *"Your personal AI librarian that never stops searching for the documentation you need"*

> **🚧 Development Status**: Core infrastructure and self-correcting data pipeline are complete! MCP server implementation is the next major goal. See [Development Roadmap](https://www.google.com/search?q=%23-development-roadmap) for details.

Meet **Librarian**, your intelligent AI assistant for navigating the vast world of software documentation. Like a skilled librarian who knows exactly where to find the information you need, this system intelligently discovers, processes, and organizes documentation from multiple software ecosystems (Python, JavaScript, Rust, etc.).

Librarian combines a sophisticated AI agent with a multi-stage, self-correcting data pipeline. It uses smart search strategies to find the most relevant and content-rich documentation, automatically deep-crawls websites to gather more context when needed, and never gives up on finding the information you need.

## 💡 **Why Librarian Exists**

**The Problem**: Every developer has been there - you're in the zone, coding away, when suddenly you hit a wall. Package errors, dependency conflicts, and documentation confusion kill your coding vibe. You spend hours debugging instead of building.

**The Solution**: Librarian is designed as an **MCP** (Model Context Protocol) **Server** that connects to your code editor through your AI model. No more context switching, no more dependency nightmares. Just pure, uninterrupted coding flow.

**The Vision**: We believe developers should be able to **vibe code** without pressure. Librarian handles the heavy lifting of finding accurate documentation, resolving package conflicts, and providing the information you need - all through your existing AI workflow.

**Connect** Librarian to your editor, and start building with confidence. No more package errors, no **more documentation confusion. Just you, your code, and your AI assistant working together seamlessly.**

## 🏗️ **Modular Architecture**

The codebase is structured into three main components for better maintainability and scalability:

### 📁 **File Structure**


librarian/
├── config.py         # Configuration, schemas, and environment variables
├── services.py       # Core logic, agent, tools, and data pipeline functions
├── main.py           # Main orchestrator and pipeline runner
├── agent.py          # Simple launcher (backward compatibility)
├── requirements.txt  # Python dependencies
└── .env              # API keys (create this file)


### 🔧 **Module Breakdown**

#### 1. **`config.py`** - Configuration & Schemas

* Environment variable loading and validation.

* Global constants (Pinecone index name, Jina Reader API URL).

* Pydantic `LibraryInfo` schema for structured data output.

#### 2. **`services.py`** - Core Logic & Tools

* All agent tools (`pypi_api_tool`, `npm_api_tool`, `crates_io_api_tool`, `smart_web_search_with_retry`).

* Creation of the sophisticated, multi-step AI agent.

* Pinecone index management, including a "delete-then-add" refresh strategy.

* Hybrid content extraction pipeline (Jina AI + Crawl4ai).

* The self-correcting RAG logic with a deep crawl fallback.

#### 3. **`main.py`** - Pipeline Orchestrator

* Clean, focused main function that runs the entire pipeline.

* Intelligent caching logic that avoids storing low-quality results.

* Model and service initialization.

## 🚀 **Librarian's Key Capabilities**

* **Advanced AI Agent**: A highly intelligent agent with a multi-step reasoning process to find the best possible documentation URL, avoiding common pitfalls like generic homepages.

* **Multi-Ecosystem Support**: Natively understands Python (PyPI), JavaScript (npm), and Rust (Crates.io).

* **Hybrid Content Extraction**: Uses the fast Jina AI Reader API for initial scrapes and falls back to the powerful `Crawl4ai` library for more complex tasks.

* **Self-Correcting Data Pipeline**: Automatically detects when an initial scrape yields low-quality or empty content.

* **Deep Crawl Fallback**: When confidence is low, it automatically triggers a **deep crawl** with `Crawl4ai` to explore the documentation site, gather richer context from multiple pages, and re-process the data.

* **Always-Fresh Vector Storage**: Before ingesting new documentation into Pinecone, it automatically deletes all old vectors for that library, ensuring the index is always up-to-date.

* **Intelligent Caching**: Avoids "cache poisoning" by only saving results that have a "High" or "Medium" confidence score.

## 🔍 **How Librarian Works: A Self-Correcting Pipeline**

Librarian uses a sophisticated, multi-stage process to ensure it gets the highest quality information.

1. **Intelligent Discovery**: The AI agent, equipped with a robust set of instructions, uses a combination of specialist API tools and contextual web searches to find the most **content-rich** documentation page, actively avoiding generic homepages or empty directory pages.

2. **Initial** Scrape & **Ingestion**: The system uses the fast Jina AI Reader to scrape the content from the URL found by the agent. It then deletes any old information for that library from the Pinecone vector store and ingests this new content.

3. **First-Pass Analysis (RAG)**: The system performs a Retrieval-Augmented Generation task, using the newly ingested content to extract structured information and a self-evaluated **confidence score**.

4. **Confidence Check & Fallback Trigger**: The system checks the result. If the confidence score is "Low" or "Unknown" (indicating the initial scrape was insufficient), it triggers the deep crawl fallback.

5. **Deep Crawl Exploration**: The system deploys `Crawl4ai` to perform a **deep crawl**, starting at the original URL and exploring linked pages to gather a much larger and richer set of content.

6. **Re-Ingestion & Final Analysis**: The low-quality content is deleted from Pinecone and replaced with the new, aggregated content from the deep crawl. The RAG process is run a second time on this high-quality context to produce the final, accurate result.

7. **Smart Caching**: The final result is only saved to the local cache if its confidence score is "High" or "Medium", ensuring that failed or low-quality runs don't prevent future attempts.

## 📋 **Setup** Instructions

### 1. **Install Dependencies**


pip install -r requirements.txt

Install browser dependencies for Crawl4ai
playwright install


### 2. **Create Environment File**

Create a `.env` file in the project root:


GOOGLE_API_KEY="your_google_api_key"
PINECONE_API_KEY="your_pinecone_api_key"
TAVILY_API_KEY="your_tavily_api_key"  # Recommended for enhanced search


### 3. **Run the Application**


python main.py


## 🚧 **Development Roadmap**

### **Phase** 1: **Core Infrastructure & Intelligence** ✅ **COMPLETE**

* \[x\] Multi-ecosystem documentation processing agent.

* \[x\] Hybrid content extraction pipeline (Jina AI + Crawl4ai).

* \[x\] Always-fresh vector storage with Pinecone.

* \[x\] Intelligent, self-correcting RAG pipeline.

* \[x\] **Deep crawl fallback** for low-confidence results.

* \[x\] Smart caching system that avoids storing bad data.

### **Phase 2: MCP Server Implementation** 🔄 **IN PROGRESS**

* \[ \] Implement Model Context Protocol server interface using FastAPI.

* \[ \] Create a `/get_documentation` endpoint to run the full pipeline.

* \[ \] Create an `/ask` endpoint to perform RAG against the Pinecone index.

### **Phase 3: Editor Integration** 📋 **PLANNED**

* \[ \] VS Code extension development.

* \[ \] Neovim plugin integration.

* \[ \] Real-time package intelligence in the editor.

### **Phase 4: Advanced Features** 🎯 **FUTURE**

* \[ \] Proactive dependency conflict prediction.

* \[ \] Community-driven documentation improvements.
