# Vertex AI RAG 

This project demonstrates how to implement Retrieval-Augmented Generation (RAG) using Google Cloud's Vertex AI. The implementation allows you to create a RAG corpus, import files, and generate AI responses enhanced with context from your documents.

This project was created by following the step-by-step guide from the [Vertex AI RAG quickstart for Python](https://cloud.google.com/vertex-ai/generative-ai/docs/rag-quickstart).

## Features

- Create and manage RAG corpus in Vertex AI
- Import documents from Google Cloud Storage
- Configure chunking and embedding models
- Perform direct context retrieval
- Generate AI responses enhanced with document context using Gemini models
- Automatically clean up RAG corpora after use

## Requirements

- Google Cloud account with Vertex AI access
- Python 3.8+
- Required Python packages (see requirements.txt)

## Setup

1. Enable required Google Cloud services:
   ```bash
   gcloud services enable aiplatform.googleapis.com --project=PROJECT_ID
   gcloud services enable storage.googleapis.com --project=PROJECT_ID
   ```

2. Set up IAM permissions:
   ```bash
   gcloud projects add-iam-policy-binding vertex-ai-experminent --member="user:YOUR_EMAIL@domain.com" --role="roles/aiplatform.user" 
   gcloud projects add-iam-policy-binding vertex-ai-experminent --member="user:YOUR_EMAIL@domain.com" --role="roles/storage.objectAdmin"
   ```

3. Create a Google Cloud Storage bucket and upload your PDF files:
   ```bash
   # Create a new GCS bucket (skip if you already have one)
   gsutil mb -l us-central1 gs://your-bucket-name
   
   # Upload PDF files to the bucket
   gsutil cp your-document.pdf gs://your-bucket-name/
   
   # Verify files were uploaded successfully
   gsutil ls gs://your-bucket-name/
   ```
   
   Note: Remember the path to your files (e.g., `gs://your-bucket-name/your-document.pdf`) as you'll need it in step 6.

4. Clone this repository

5. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

6. Update the configuration in `vertex_ai_rag.py`:
   - Set your PROJECT_ID (e.g., "vertex-ai-experminent")
   - Configure your corpus display name
   - Add paths to your documents in Google Cloud Storage:
     ```python
     paths = ["gs://your-bucket-name/your-document.pdf"] 
     ```

## Usage

Run the main script:

```
python vertex_ai_rag.py
```

The script demonstrates:
- Creating a RAG corpus
- Importing documents
- Direct context retrieval
- Enhanced generation with RAG
- Automatic cleanup of the created corpus when finished

### Resource Management

To prevent resource accumulation, the script automatically deletes the RAG corpus it creates after completing the process. This ensures you don't accumulate unused resources in your Google Cloud project.

## Documentation

For more information on Vertex AI RAG, see the official [Google Cloud documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/rag-quickstart). 
