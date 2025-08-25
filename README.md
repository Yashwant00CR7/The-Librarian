# Librarian - AI-Powered Documentation Intelligence System



> *"Your personal AI librarian that never stops searching for the documentation you need"*



> **🚧 Development Status**: Core infrastructure is complete! MCP server implementation is currently in progress. See [Development Roadmap](#-development-roadmap) for details.



Meet **Librarian**, your intelligent AI assistant for navigating the vast world of software documentation. Like a skilled librarian who knows exactly where to find the information you need, this system intelligently discovers, processes, and organizes documentation from multiple software ecosystems (Python, JavaScript, Rust, etc.).



Librarian combines the power of AI with smart search strategies to find the most relevant and up-to-date documentation, even when links are broken or outdated. It's your personal documentation expert that never gives up on finding the information you need.



## 💡 **Why Librarian Exists**



**The Problem**: Every developer has been there - you're in the zone, coding away, when suddenly you hit a wall. Package errors, dependency conflicts, and documentation confusion kill your coding vibe. You spend hours debugging instead of building.



**The Solution**: Librarian is designed as an **MCP (Model Context Protocol) Server** that connects to your code editor through your AI model. No more context switching, no more dependency nightmares. Just pure, uninterrupted coding flow.



**The Vision**: We believe developers should be able to **vibe code** without pressure. Librarian handles the heavy lifting of finding accurate documentation, resolving package conflicts, and providing the information you need - all through your existing AI workflow.



**Connect Librarian to your editor, and start building with confidence. No more package errors, no more documentation confusion. Just you, your code, and your AI assistant working together seamlessly.**



## 🏗️ **New Modular Architecture**



The codebase has been restructured into three main components for better maintainability and scalability:



### 📁 **File Structure**

```

librarian/

├── config.py          # Configuration, schemas, and environment variables

├── services.py        # Core logic, tools, and functions

├── main.py           # Main orchestrator and pipeline runner

├── agent.py          # Simple launcher (backward compatibility)

├── requirements.txt  # Python dependencies

└── .env             # API keys (create this file)

```



### 🔧 **Module Breakdown**



#### 1. **`config.py`** - Configuration & Schemas

- Environment variable loading and validation

- Global constants (Pinecone index name, Jina Reader API URL)

- Pydantic `LibraryInfo` schema for structured data output

- Environment validation function



#### 2. **`services.py`** - Core Logic & Tools

- All agent tools (`pypi_api_tool`, `npm_api_tool`, `crates_io_api_tool`, `web_search_tool`)

- Agent creation and management

- Pinecone index management

- Jina AI content extraction

- Documentation ingestion and processing

- Structured information extraction using RAG



#### 3. **`main.py`** - Pipeline Orchestrator

- Clean, focused main function

- Model initialization

- Pipeline orchestration

- Error handling and logging



## 🚀 **Librarian's Key Capabilities**



- **AI-Powered Content Extraction**: Uses Jina AI Reader API for clean, LLM-friendly content

- **Multi-Ecosystem Support**: Python (PyPI), JavaScript (npm), Rust (Crates.io)

- **Intelligent Agent Routing**: Automatically selects the best tool for each ecosystem

- **Smart Fallback Search**: DuckDuckGo for fast results, Tavily AI as intelligent backup

- **Intelligent Retry System**: Automatically retries with alternative strategies when documentation links are broken

- **Vector Storage**: Pinecone integration for efficient document retrieval

- **Structured Output**: Rich, structured data extraction using Gemini 1.5 Flash

- **Confidence Scoring**: Self-evaluated reliability metrics for extracted data

- **Deep Rescraping**: Automatic additional content analysis when confidence is low



## 🔌 **MCP Server Integration - Your Coding Flow Companion**



Librarian is designed as an **MCP (Model Context Protocol) Server** that will integrate seamlessly with your existing AI workflow. **Note: The MCP server implementation is currently under active development.**



### **Current Status** 🚧

- ✅ **Core Infrastructure**: Complete documentation processing pipeline with AI-powered content extraction

- ✅ **Smart Search**: Multi-ecosystem support (Python, JavaScript, Rust) with intelligent fallback

- ✅ **Caching System**: Local caching layer for improved performance and cost efficiency

- ✅ **Confidence Scoring**: Self-evaluated reliability metrics for extracted data

- ✅ **Vector Storage**: Pinecone integration for efficient document retrieval

- 🔄 **MCP Server**: Currently implementing the Model Context Protocol server interface

- 🔄 **Editor Integration**: Working on seamless connection to VS Code, Neovim, and other editors



### **How It Will Work** (When Complete)

1. **Connect Once**: Set up Librarian as an MCP server in your development environment

2. **Stay in Flow**: Your AI model automatically queries Librarian when you need package info

3. **No Context Switching**: Get documentation, dependency info, and package details without leaving your editor

4. **Vibe Coding**: Focus on building, not debugging package conflicts



### **What You'll Get**

- **Real-time Package Intelligence**: Instant access to package versions, dependencies, and compatibility

- **Documentation on Demand**: No more hunting through multiple websites

- **Dependency Conflict Prevention**: Librarian warns you about potential issues before they happen

- **Seamless Integration**: Works with VS Code, Neovim, and any editor that supports MCP



### **The Vision**

**Pure, uninterrupted coding flow.** No more package errors killing your vibe. No more dependency confusion breaking your rhythm. Just you, your code, and your AI assistant working together to build amazing things.



## 📋 **Setup Instructions**



### 1. **Install Dependencies**

   ```bash

   pip install -r requirements.txt

   ```



### 2. **Create Environment File**

Create a `.env` file in the project root:

   ```env

GOOGLE_API_KEY="your_google_api_key"

PINECONE_API_KEY="your_pinecone_api_key"

TAVILY_API_KEY="your_tavily_api_key"  # Optional: Enhances fallback search

   ```



**Note**:

- `GOOGLE_API_KEY` and `PINECONE_API_KEY` are required

- `TAVILY_API_KEY` is optional but recommended for enhanced fallback search capabilities

- Get your free Tavily API key from [Tavily](https://tavily.com/)



### 3. **Run the Application**

```bash

# Option 1: Use the new modular structure (Recommended)

python main.py



# Option 2: Use the backward-compatible launcher

python agent.py

```



### 4. **Current Usage**

**Note**: The MCP server interface is still under development. Currently, you can:

- Process documentation for specific libraries

- Extract structured information with confidence scoring

- Use the caching system for improved performance

- Test the core pipeline with the example in `main.py`



**For MCP Server Integration**: This feature is coming soon! The foundation is complete, and we're actively working on the MCP protocol implementation.



## 🎯 **Usage Examples**



### **Basic Usage**

```python

from main import main



# Run the full pipeline

main()

```



### **Custom Library Processing**

```python

from services import create_universal_agent, ingest_documentation

from config import validate_environment



# Validate environment

validate_environment()



# Create agent and process specific libraries

agent = create_universal_agent(llm)

# ... custom processing logic

```



## 😤 **Developer Pain Points - Solved**



### **Before Librarian (The Struggle)**

- ❌ **Package Hell**: "Why does this package have 47 different versions?"

- ❌ **Documentation Chaos**: "Is this the right docs? The link is broken..."

- ❌ **Dependency Nightmares**: "This worked yesterday, why is it broken today?"

- ❌ **Context Switching**: "Let me Google this... wait, what was I building again?"

- ❌ **Vibe Killer**: "I was in the zone, now I'm debugging package conflicts"



### **With Librarian (The Solution)**

- ✅ **Package Intelligence**: Instant version compatibility and dependency info

- ✅ **Documentation Clarity**: Always find the right, working documentation

- ✅ **Conflict Prevention**: Librarian warns you before you hit dependency issues

- ✅ **Stay in Flow**: No context switching, just pure coding

- ✅ **Vibe Coding**: Build without interruption, create without frustration



### **Real-World Scenario**

```

You: "I need to add image processing to my app"

Librarian: "Here's the latest Pillow version, compatible with Python 3.9+,

           no conflicts with your current packages, and here's the official docs"

You: *continues coding without breaking flow*

```



**That's the power of Librarian. No more package errors killing your vibe.**



## 🔍 **How Librarian Works**



1. **Library Identification**: Librarian analyzes the library name and determines its ecosystem

2. **Documentation Discovery**: Uses specialized APIs (PyPI, npm, Crates.io) or smart web search as fallback

3. **Smart Fallback Search**:

   - **Primary**: Fast DuckDuckGo search for reliable results

   - **Backup**: Tavily AI-powered search when DuckDuckGo is insufficient

   - **Automatic**: Seamlessly switches between search engines based on result quality

4. **Content Extraction**: Jina AI Reader strips noise and returns clean Markdown

5. **Vector Storage**: Content is chunked and stored in Pinecone for efficient retrieval

6. **Information Extraction**: LLM extracts structured data using RAG (Retrieval-Augmented Generation)

7. **Deep Rescraping**: When confidence is low, automatically re-analyzes the source for additional insights



### **Deep Rescraping Feature**

When the system detects low confidence in extracted data, it automatically:

- **Re-scrapes** the documentation site for fresh content

- **Analyzes** the content for important information that wasn't captured in structured fields

- **Extracts** key insights like:

  - Performance considerations and limitations

  - Security notes and warnings

  - Integration requirements

  - Known gotchas and best practices

  - Recent changes or updates

  - Community recommendations

- **Adds** these insights to the `additional_insights` field in the output JSON



### **Intelligent Retry System**

When the system encounters broken links, 404 errors, or invalid documentation, it automatically:

- **Detects Error Patterns**: Identifies common error messages indicating broken documentation

- **Triggers Alternative Searches**: Uses multiple retry strategies with different keywords

- **Retry Strategies**:

  - **Strategy 1**: `{query} documentation official` - Focuses on official documentation

  - **Strategy 2**: `{query} developer guide api reference` - Targets developer resources

  - **Strategy 3**: `{query} API guide tutorial examples` - API-specific documentation

  - **Strategy 4**: `{query} GitHub source code repository` - Source code and repositories

- **Seamless Recovery**: Automatically finds working documentation without user intervention



## 🛠️ **Customization**



### **Adding New Ecosystems**

1. Add new API tool in `services.py`

2. Update the agent prompt in `create_universal_agent()`

3. Add the tool to the tools list



### **Modifying Output Schema**

1. Update the `LibraryInfo` class in `config.py`

2. Adjust the extraction prompt in `extract_structured_info()`



### **Changing Processing Logic**

1. Modify functions in `services.py`

2. Update the pipeline flow in `main.py`



## 🚧 **Development Roadmap**



### **Phase 1: Core Infrastructure** ✅ **COMPLETE**

- [x] Multi-ecosystem documentation processing

- [x] AI-powered content extraction with Jina Reader

- [x] Intelligent search with fallback strategies

- [x] Vector storage with Pinecone

- [x] Caching system for performance

- [x] Confidence scoring and deep rescraping



### **Phase 2: MCP Server Implementation** 🔄 **IN PROGRESS**

- [ ] Implement Model Context Protocol server interface

- [ ] Add MCP tools for package information retrieval

- [ ] Create MCP resources for documentation access

- [ ] Implement proper MCP error handling and logging



### **Phase 3: Editor Integration** 📋 **PLANNED**

- [ ] VS Code extension development

- [ ] Neovim plugin integration

- [ ] Universal editor support through MCP

- [ ] Real-time package intelligence in editor



### **Phase 4: Advanced Features** 🎯 **FUTURE**

- [ ] Proactive library watchlist and updates

- [ ] Dependency conflict prediction

- [ ] Package compatibility analysis

- [ ] Community-driven documentation improvements



## 📝 **Output Format**



The system generates structured JSON files containing:

- Library name and package name

- Latest version information

- Summary and purpose

- Installation commands

- Ecosystem identification

- Deprecation notices (if applicable)

- Confidence score (High/Medium/Low)

- Additional insights (when confidence is low, includes deep rescraping results)



## 🔧 **Troubleshooting**



### **Common Issues**

- **Environment Variables**: Ensure `.env` file exists with correct API keys

- **Pinecone Index**: The system automatically creates the index if it doesn't exist

- **API Limits**: Monitor usage of Google AI and Pinecone APIs



### **Debug Mode**

The system includes comprehensive logging. Check console output for detailed progress information.



## 🤝 **Contributing**



When contributing to Librarian:

1. Keep the modular structure intact

2. Add new tools to `services.py`

3. Update configuration in `config.py`

4. Maintain backward compatibility in `agent.py`



## 📝 **License**



This project is open source. Please refer to the project license for details.