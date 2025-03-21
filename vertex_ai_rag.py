from vertexai import rag
from vertexai.generative_models import GenerativeModel, Tool
import vertexai
import logging
import sys
from datetime import datetime

# Configure logging with two levels only
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("vertex_ai_rag.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("vertex-ai-rag")

# Utility function to convert timestamp to readable format
def format_timestamp(timestamp):
    """Convert a timestamp with seconds to a readable date/time format"""
    if not timestamp or not hasattr(timestamp, 'seconds'):
        return "Unknown"
    try:
        dt = datetime.fromtimestamp(timestamp.seconds)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(timestamp.seconds) if hasattr(timestamp, 'seconds') else "Unknown"

# Create a RAG Corpus, Import Files, and Generate a response

# Replace these values with your own
PROJECT_ID = "PROJECT_ID" # Your Google Cloud Project ID, e.g., "vertex-ai-experminent"
display_name = "PROJECT_ID_rag_corpus" # A name for your RAG corpus
paths = [] # List of file paths, e.g., ["gs://your-bucket-name/your-document.pdf"]


# LEVEL 1: Main steps with clear logging
logger.info("===== VERTEX AI RAG PROCESS STARTED =====")
logger.info("---------")

# LEVEL 1: Step 1 - Initialize Vertex AI API
logger.info("STEP 1: Initializing Vertex AI API")
# LEVEL 2: Details of the step
logger.info(f"Project ID: {PROJECT_ID}, Location: us-central1")
try:
    vertexai.init(project=PROJECT_ID, location="us-central1")
    logger.info("✓ Vertex AI API initialized")
except Exception as e:
    logger.error(f"✗ API initialization failed: {str(e)}")
    raise
logger.info("---------")

# LEVEL 1: Step 2 - Create RAG Corpus
logger.info("STEP 2: Creating RAG Corpus")
# LEVEL 2: Details of the step
logger.info(f"Embedding model: text-embedding-005, Corpus name: {display_name}")
try:
    embedding_model_config = rag.RagEmbeddingModelConfig(
          vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
              publisher_model="publishers/google/models/text-embedding-005"
          )
    )
    backend_config = rag.RagVectorDbConfig(rag_embedding_model_config=embedding_model_config)
    rag_corpus = rag.create_corpus(
        display_name=display_name,
        backend_config=backend_config,
    )
    logger.info(f"✓ RAG corpus created: {rag_corpus.name}")
except Exception as e:
    logger.error(f"✗ RAG corpus creation failed: {str(e)}")
    raise
logger.info("---------")

# LEVEL 1: Step 3 - List RAG Corpora
logger.info("STEP 3: Listing RAG Corpora")
try:
    corpora = list(rag.list_corpora())
    logger.info(f"✓ Found {len(corpora)} corpora")
    for i, corpus in enumerate(corpora):
        logger.info(f"  Corpus {i+1}: {corpus.name if hasattr(corpus, 'name') else str(corpus)}")
except Exception as e:
    logger.error(f"✗ Listing corpora failed: {str(e)}")
    raise
logger.info("---------")

# LEVEL 1: Step 4 - Import Files
logger.info("STEP 4: Importing Files to RAG Corpus")
# LEVEL 2: Details of the step
corpus_name = rag_corpus.name
logger.info(f"Target corpus: {corpus_name}")
logger.info(f"Files to import: {paths}")
logger.info("Chunk size: 512, Overlap: 100")
try:
    transformation_config = rag.TransformationConfig(
          chunking_config=rag.ChunkingConfig(
              chunk_size=512,
              chunk_overlap=100,
          ),
      )
    rag.import_files(
        corpus_name,
        paths,
        transformation_config=transformation_config,
        max_embedding_requests_per_min=1000,
    )
    logger.info("✓ Files imported successfully")
except Exception as e:
    logger.error(f"✗ File import failed: {str(e)}")
    raise
logger.info("---------")

# LEVEL 1: Step 5 - List Files in Corpus
logger.info("STEP 5: Listing Files in RAG Corpus")
try:
    files = list(rag.list_files(corpus_name))
    logger.info(f"✓ Found {len(files)} files in corpus")
    for i, file in enumerate(files):
        # LEVEL 2: File details
        create_time = format_timestamp(file.create_time if hasattr(file, 'create_time') else None)
        source = file.gcs_source.uris[0] if hasattr(file, 'gcs_source') and hasattr(file.gcs_source, 'uris') and file.gcs_source.uris else "Unknown"
        status = file.file_status.state if hasattr(file, 'file_status') and hasattr(file.file_status, 'state') else "Unknown"
        logger.info(f"  File {i+1}: {file.display_name if hasattr(file, 'display_name') else 'Unknown'} (Created: {create_time}, Status: {status})")
        logger.info(f"    Source: {source}")
except Exception as e:
    logger.error(f"✗ Listing files failed: {str(e)}")
    raise
logger.info("---------")

# LEVEL 1: Step 6 - Direct Context Retrieval
logger.info("STEP 6: Performing Direct Context Retrieval")
# LEVEL 2: Details of the step
query = "What is RAG and why it is helpful?"
logger.info(f"Query: '{query}'")
logger.info("Retrieval config: top_k=3, distance_threshold=0.5")
try:
    rag_retrieval_config=rag.RagRetrievalConfig(
        top_k=3,
        filter=rag.Filter(vector_distance_threshold=0.5)
    )
    response = rag.retrieval_query(
        rag_resources=[rag.RagResource(rag_corpus=corpus_name)],
        text=query,
        rag_retrieval_config=rag_retrieval_config,
    )
    
    # LEVEL 2: Retrieval results
    chunk_count = len(response.chunks) if hasattr(response, 'chunks') else 0
    logger.info(f"✓ Retrieved {chunk_count} chunks of context")
    
    if hasattr(response, 'chunks') and response.chunks:
        for i, chunk in enumerate(response.chunks):
            text_preview = chunk.text[:100].replace('\n', ' ') + '...' if hasattr(chunk, 'text') else "No text"
            file_name = chunk.source.file.display_name if hasattr(chunk, 'source') and hasattr(chunk.source, 'file') and hasattr(chunk.source.file, 'display_name') else "Unknown"
            score = f"{chunk.relevance_score:.4f}" if hasattr(chunk, 'relevance_score') else "Unknown"
            logger.info(f"  Chunk {i+1}: From '{file_name}' (Relevance: {score})")
            logger.info(f"    Preview: {text_preview}")
    
    print(response)
except Exception as e:
    logger.error(f"✗ Context retrieval failed: {str(e)}")
    raise
logger.info("---------")

# LEVEL 1: Step 7 - Generate Content with RAG
logger.info("STEP 7: Generating Content with RAG")
# LEVEL 2: Details of the step
logger.info(f"Query: '{query}'")
logger.info("Model: gemini-2.0-flash-001")
try:
    rag_retrieval_tool = Tool.from_retrieval(
        retrieval=rag.Retrieval(
            source=rag.VertexRagStore(
                rag_resources=[rag.RagResource(rag_corpus=corpus_name)],
                rag_retrieval_config=rag_retrieval_config,
            ),
        )
    )
    rag_model = GenerativeModel(
        model_name="gemini-2.0-flash-001", tools=[rag_retrieval_tool]
    )
    response = rag_model.generate_content(query)
    
    # LEVEL 2: Generation results
    logger.info("✓ Response generated successfully")
    
    # Get response text
    response_text = response.text if hasattr(response, 'text') else "No text available"
    logger.info("=== RESPONSE TEXT ===")
    logger.info(response_text)
    
    # Get usage info if available
    if hasattr(response, 'usage_metadata'):
        usage = response.usage_metadata
        token_info = []
        if hasattr(usage, 'prompt_token_count'):
            token_info.append(f"Prompt: {usage.prompt_token_count}")
        if hasattr(usage, 'candidates_token_count'):
            token_info.append(f"Response: {usage.candidates_token_count}")
        if hasattr(usage, 'total_token_count'):
            token_info.append(f"Total: {usage.total_token_count}")
        if token_info:
            logger.info(f"Token usage: {', '.join(token_info)}")
    
    print(response.text)
except Exception as e:
    logger.error(f"✗ Content generation failed: {str(e)}")
    raise
logger.info("---------")

logger.info("===== VERTEX AI RAG PROCESS COMPLETED =====")

# Example response:
#   RAG stands for Retrieval-Augmented Generation.
#   It's a technique used in AI to enhance the quality of responses
# ...

# Function to delete a RAG corpus
def delete_rag_corpus(corpus_name):
    """
    Delete a RAG corpus by its full resource name.
    
    Args:
        corpus_name (str): Full resource name of the corpus
            (e.g. 'projects/PROJECT_ID/locations/LOCATION/ragCorpora/CORPUS_ID')
    
    Returns:
        None
    """
    logger.info(f"Deleting RAG corpus: {corpus_name}")
    try:
        rag.delete_corpus(corpus_name)
        logger.info(f"✓ Successfully deleted corpus: {corpus_name}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to delete corpus: {str(e)}")
        return False

if __name__ == "__main__":
    # Clean up the corpus after completing the RAG process
    logger.info("STEP 8: Cleaning up RAG resources")
    
    # Delete the corpus we created in this run
    try:
        logger.info(f"Deleting corpus created in this run: {corpus_name}")
        if delete_rag_corpus(corpus_name):
            logger.info("✓ Created corpus deleted successfully")
        else:
            logger.error("✗ Failed to delete corpus")
    except Exception as e:
        logger.error(f"✗ Failed to delete created corpus: {str(e)}")
    
    logger.info("===== CLEANUP COMPLETED =====")