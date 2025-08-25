import os
import re
import time
from dotenv import load_dotenv

# --- Local Imports from your project ---
from services import (
    initialize_services,
    create_universal_agent,
    ensure_pinecone_index_ready,
    extract_structured_info,
    load_from_cache,
    save_to_cache,
    interpret_confidence_score,
    logger # Import logger to use it
)

# --- Main Application Logic ---

def find_documentation_url(agent_executor, library_name: str) -> str | None:
    """
    Uses the agent to find the official documentation URL for a library.
    """
    print(f"\n🕵️ Agent is searching for documentation for '{library_name}'...")
    try:
        response = agent_executor.invoke({"input": f"Find the documentation URL for the {library_name} library."})
        output = response.get("output", "")
        
        url_match = re.search(r'https?://[^\s,"]+', output)
        if url_match:
            doc_url = url_match.group(0).strip().strip(' ./\t\n')
            print(f"✅ Agent found URL: {doc_url}")
            return doc_url
        else:
            print(f"❌ Agent failed to find a valid URL in its output: {output}")
            return None
            
    except Exception as e:
        print(f"❌ An error occurred while the agent was searching: {e}")
        return None

def main():
    """
    The main pipeline orchestrator for the Librarian AI system.
    """
    load_dotenv()
    llm, embeddings_model, pc = initialize_services()
    ensure_pinecone_index_ready(pc, embeddings_model)
    agent_executor = create_universal_agent(llm)
    
    # --- Define the Single Library to Process ---
    library_to_process = "conscience"

    libraries_to_process = [
          # "sklearn",
          # "Pillow",
          # "requests",
          # "@angular/core",
          # "conscience",
          # "tokio",
          # "tensorflow-gpu",
          # "godot",
          # "BeautifulSoup4",
          # "inquirer"
    ]
    
    print("-" * 60)
    print(f"🚀 STARTING LIBRARIAN PIPELINE FOR: '{library_to_process}'")
    print("-" * 60)

    # --- Check Cache First ---
    cached_data = load_from_cache(library_to_process)
    if cached_data:
        print("\n✅ Found valid data in cache. Displaying cached results.")
        confidence_meaning = interpret_confidence_score(cached_data.get("confidence_score"))
        print(f"\n--- CACHED RESULTS for {library_to_process} ---")
        print(f"  Confidence: {confidence_meaning}")
        for key, value in cached_data.items():
            if value:
                print(f"  - {key.replace('_', ' ').title()}: {value}")
        print("-" * 60)
        return # End the process if we have a good cache hit

    # --- Run Agent to Find URL ---
    doc_url = find_documentation_url(agent_executor, library_to_process)
    
    if not doc_url:
        print(f"\n❌ Halting pipeline for '{library_to_process}': Agent failed to find a valid URL.")
        return

    # --- Extract Structured Information ---
    library_info = extract_structured_info(
        library_name=library_to_process,
        llm=llm,
        embeddings_model=embeddings_model,
        doc_url=doc_url
    )

    # --- Process and Display Final Results ---
    if library_info:
        print(f"\n✅ Pipeline completed successfully for '{library_to_process}'!")
        confidence_meaning = interpret_confidence_score(library_info.confidence_score)
        
        print(f"\n--- FINAL RESULTS for {library_to_process} ---")
        print(f"  Confidence: {confidence_meaning}")
        
        info_dict = library_info.model_dump()
        for key, value in info_dict.items():
            if value:
                print(f"  - {key.replace('_', ' ').title()}: {value}")

        # --- CORRECTED CACHING LOGIC ---
        # Only save to cache if the confidence is high or medium.
        if library_info.confidence_score and library_info.confidence_score.lower() in ["high", "medium"]:
            save_to_cache(library_to_process, library_info)
        else:
            logger.warning(f"⚠️ Skipping cache for '{library_to_process}' due to low or unknown confidence.")
    else:
        print(f"\n❌ Halting pipeline: Failed to extract structured information for '{library_to_process}'.")
        
    print("-" * 60)

if __name__ == "__main__":
    main()
