# --- MCP Server: Main Entry Point ---
# This file creates a FastAPI web server to expose the Librarian's capabilities
# as a Model Context Protocol (MCP) server.

import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from contextlib import asynccontextmanager

# --- Local Imports from your project ---
# These functions contain the core intelligence of the Librarian.
from services import (
    initialize_services,
    create_universal_agent,
    _sanitize_filename,
    ensure_pinecone_index_ready,
    extract_structured_info,
    load_from_cache,
    save_to_cache,
    interpret_confidence_score,
    # find_documentation_url,
    logger,
    PineconeVectorStore # Import for the /ask endpoint
)

from main import find_documentation_url

# --- Data Models for API Requests ---
# These Pydantic models define the expected JSON structure for incoming requests.

class ProcessRequest(BaseModel):
    """The request model for processing a new library."""
    library_name: str

class AskRequest(BaseModel):
    """The request model for asking a question about a library."""
    library_name: str
    question: str

# --- Global State Management ---
# This dictionary will hold our initialized services (LLM, agent, etc.)
# so they are created only once when the server starts, not on every request.
server_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This function runs on server startup and shutdown. It's the perfect place
    to initialize our models and services so they are ready to handle requests.
    """
    print("--- Server is starting up... ---")
    load_dotenv()
    
    # Initialize all the core components of the Librarian
    llm, embeddings_model, pc = initialize_services()
    ensure_pinecone_index_ready(pc, embeddings_model)
    agent_executor = create_universal_agent(llm)
    
    # Store the initialized components in our server_state dictionary
    server_state["llm"] = llm
    server_state["embeddings_model"] = embeddings_model
    server_state["agent_executor"] = agent_executor
    
    print("--- Models and services initialized. Server is ready. ---")
    yield
    # This part runs on shutdown
    print("--- Server is shutting down... ---")
    server_state.clear()


# --- FastAPI Application ---
app = FastAPI(
    title="Librarian MCP Server",
    description="An AI-powered service to find, process, and query software documentation.",
    lifespan=lifespan
)

# --- API Endpoints ---

@app.post("/process_library", response_model=dict)
async def process_library_endpoint(request: ProcessRequest):
    """
    This endpoint runs the full Librarian pipeline for a given library name.
    It checks the cache, finds the documentation, ingests it, and returns
    the structured information.
    """
    library_name = request.library_name
    logger.info(f"Received request to process library: {library_name}")

    # 1. Check Cache
    cached_data = load_from_cache(library_name)
    if cached_data:
        logger.info(f"Cache hit for '{library_name}'. Returning cached data.")
        return cached_data

    # 2. Find URL with the Agent
    doc_url = find_documentation_url(server_state["agent_executor"], library_name)
    if not doc_url:
        raise HTTPException(status_code=404, detail=f"Agent failed to find a valid URL for '{library_name}'.")

    # 3. Extract Info (this function now contains the full ingestion and fallback logic)
    library_info = extract_structured_info(
        library_name=library_name,
        llm=server_state["llm"],
        embeddings_model=server_state["embeddings_model"],
        doc_url=doc_url
    )

    if not library_info:
        raise HTTPException(status_code=500, detail="Failed to extract structured information after ingestion.")

    # 4. Save to cache ONLY if confidence is high or medium
    info_dict = library_info.model_dump()
    if library_info.confidence_score and library_info.confidence_score.lower() in ["high", "medium"]:
        save_to_cache(library_name, info_dict)
    else:
        logger.warning(f"Skipping cache for '{library_name}' due to low or unknown confidence.")

    return info_dict

@app.post("/ask", response_model=dict)
async def ask_endpoint(request: AskRequest):
    """
    This is the RAG endpoint. It answers a question about a library
    by retrieving relevant context from the Pinecone database.
    """
    library_name = request.library_name
    question = request.question
    logger.info(f"Received question about '{library_name}': '{question}'")

    try:
        # 1. Connect to the existing Pinecone index
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=os.getenv("PINECONE_INDEX_NAME", "mcp-documentation-index"),
            embedding=server_state["embeddings_model"]
        )

        # 2. Create a retriever to search for relevant documents
        # We filter by the doc_id to only search within the specified library's docs
        doc_id = f"lib-{_sanitize_filename(library_name)}"
        retriever = vectorstore.as_retriever(
            search_kwargs={'k': 5, 'filter': {'doc_id': doc_id}}
        )

        # 3. Perform the search
        docs = retriever.invoke(question)

        if not docs:
            return {"answer": "I could not find any relevant information in the documentation for that library. It might not have been processed yet."}

        # 4. Combine the context and return it
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])
        
        return {
            "library": library_name,
            "question": question,
            "context": context
        }

    except Exception as e:
        logger.error(f"Error during RAG retrieval for '{library_name}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve information. Error: {e}")


# --- Run the Server ---
if __name__ == "__main__":
    """
    This block allows you to run the server directly from the command line
    for testing and development.
    """
    print("Starting Librarian MCP Server with Uvicorn...")
    uvicorn.run("mcp_server:app", host="0.0.0.0", port=8000, reload=True)
