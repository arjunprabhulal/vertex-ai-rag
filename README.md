# Vertex AI RAG with Gemini 2 Flash

![Vertex AI RAG with Gemini 2 Flash](images/vertex-ai-rag.gif)

A complete implementation of Retrieval-Augmented Generation (RAG) workflow using Google Cloud's Vertex AI and Gemini 2.0 Flash.

## Step-by-Step RAG Workflow

1. Create and manage RAG corpus in Vertex AI
2. Import documents from Google Cloud Storage
3. Configure chunking and embedding models
4. Perform direct context retrieval
5. Generate AI responses enhanced with document context using Gemini models
6. Clean up RAG corpora after demo

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
   gcloud projects add-iam-policy-binding PROJECT_ID --member="user:YOUR_EMAIL@domain.com" --role="roles/aiplatform.user" 
   gcloud projects add-iam-policy-binding PROJECT_ID --member="user:YOUR_EMAIL@domain.com" --role="roles/storage.objectAdmin"
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
   
   Note: Remember the path to your files (e.g., `gs://your-bucket-name/your-document.pdf`) as you'll need it for the corpus creation.

4. Set up authentication credentials:
   ```bash
   # Download your service account key from GCP Console and set the environment variable
   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account-key
   
   # On Windows, use:
   # set GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\your\service-account-key
   ```
   
   Note: This step is critical for authenticating with Google Cloud services. You need a service account with the appropriate permissions.

5. Clone this repository:
   ```bash
   git clone https://github.com/arjunprabhulal/vertex-ai-rag.git
   cd vertex-ai-rag
   ```

6. Install dependencies:
   ```
   pip install -r requirements.txt
   pip install --upgrade google-cloud-aiplatform
   ```

7. Update the configuration in `vertex_ai_rag.py`:
   - Set your PROJECT_ID
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

## Documentation

For more information on Vertex AI RAG, see the official [Google Cloud documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/rag-overview).

## Author

For more articles on AI/ML and Generative AI, follow me on Medium: https://medium.com/@arjun-prabhulal
