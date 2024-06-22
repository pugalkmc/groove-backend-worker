import os
import sys
import logging
import json
import time
from bson import ObjectId
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
import google.generativeai as genai
from db import sources_collection
from config import GOOGLE_API_KEY

# Configure the Google Generative AI API
genai.configure(api_key=GOOGLE_API_KEY)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
CACHE_DIR = "cache"
BATCH_SIZE = 1000
DIMENSION = 768

# Set up the model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro-latest",
    generation_config=generation_config,
    safety_settings=safety_settings,
)

class Document:
    def __init__(self, page_content, metadata, lookup_str='', lookup_index=0):
        self.page_content = page_content
        self.metadata = metadata
        self.lookup_str = lookup_str
        self.lookup_index = lookup_index

def document_to_dict(doc):
    return {
        "page_content": doc.page_content,
        "metadata": doc.metadata,
        "lookup_str": doc.lookup_str,
        "lookup_index": doc.lookup_index
    }

def dict_to_document(doc_dict):
    return Document(
        page_content=doc_dict["page_content"],
        metadata=doc_dict["metadata"],
        lookup_str=doc_dict.get("lookup_str", ''),
        lookup_index=doc_dict.get("lookup_index", 0)
    )

def load_progress(_id, stage):
    file_path = os.path.join(CACHE_DIR, f'{_id}_{stage}.json')
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as file:
                return [dict_to_document(doc_dict) for doc_dict in json.load(file)]
        except json.JSONDecodeError as e:
            logger.error(f"Error loading progress file {file_path}: {e}")
    return None

def save_progress(_id, stage, data):
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    file_path = os.path.join(CACHE_DIR, f'{_id}_{stage}.json')
    try:
        with open(file_path, 'w') as file:
            json.dump([document_to_dict(doc) if isinstance(doc, Document) else doc for doc in data], file)
    except Exception as e:
        logger.error(f"Error saving progress file {file_path}: {e}")

def load_embedding_progress(_id):
    file_path = os.path.join(CACHE_DIR, f'{_id}_embedding.json')
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as file:
                return json.load(file)
        except json.JSONDecodeError as e:
            logger.error(f"Error loading embedding progress file {file_path}: {e}")
    return None

def save_embedding_progress(_id, data):
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    file_path = os.path.join(CACHE_DIR, f'{_id}_embedding.json')
    try:
        with open(file_path, 'w') as file:
            json.dump(data, file)
    except Exception as e:
        logger.error(f"Error saving embedding progress file {file_path}: {e}")

def extract_and_store(index, BATCH_SIZE, links, _id, namespace):
    try:
        web_content = load_progress(_id, 'scraping')
        if not web_content:
            web_content = estimate_and_scrap_websites(links)
            save_progress(_id, 'scraping', web_content)

        chunks = load_progress(_id, 'chunking')
        if not chunks:
            chunks = chunk_text(web_content)
            sources_collection.update_one({'_id': ObjectId(_id)}, {'$set': {'chunkLength': len(chunks)}})
            save_progress(_id, 'chunking', chunks)

        formatted_chunks = load_embedding_progress(_id)
        if not formatted_chunks:
            formatted_chunks = embedding_gemini(chunks, _id)
            save_embedding_progress(_id, formatted_chunks)

        logger.info("Started upserting vector embedding into Pinecone")
        for i in range(0, len(formatted_chunks), BATCH_SIZE):
            batch_data = formatted_chunks[i:i + BATCH_SIZE]
            index.upsert(vectors=batch_data, namespace=namespace)
            logger.info(f"{i + BATCH_SIZE}/{len(formatted_chunks)} Chunks upserted")
        logger.info("Upserted into Pinecone! Success")

        # Clean up cache after successful completion
        os.remove(os.path.join(CACHE_DIR, f'{_id}_scraping.json'))
        os.remove(os.path.join(CACHE_DIR, f'{_id}_chunking.json'))
        os.remove(os.path.join(CACHE_DIR, f'{_id}_embedding.json'))

    except Exception as e:
        logger.error(f"An error occurred for ID: {_id}: {e}")
        return False
    return True

def extract_text_from_website(url):
    loader = WebBaseLoader(url)
    try:
        return loader.load()
    except Exception as e:
        logger.error(f"Failed to extract text from {url}: {e}")
        return []

def estimate_and_scrap_websites(urls):
    total_urls = len(urls)
    start_time = time.time()
    times = []
    web_content = []
    active_url = []

    for idx, url in enumerate(urls):
        active_url.append(url)
        if len(active_url) < 10:
            continue
        url_start_time = time.time()

        # Extract content from the website
        documents = extract_text_from_website(url)
        web_content.extend(documents)
        active_url = []

        # Calculate time taken for this URL
        url_end_time = time.time()
        time_taken = url_end_time - url_start_time
        times.append(time_taken)

        # Calculate average time per URL and estimate remaining time
        avg_time_per_url = sum(times) / len(times)
        remaining_urls = total_urls - (idx + 1)
        estimated_remaining_time = avg_time_per_url * remaining_urls

        # Update the logger with the estimated time
        logger.info(f"Processed {idx + 1}/{total_urls} websites. Estimated remaining time: {estimated_remaining_time:.2f} seconds")

    if len(active_url) > 0:
        documents = extract_text_from_website(active_url)
        web_content.extend(documents)
        active_url = []

    end_time = time.time()
    total_time_taken = end_time - start_time
    logger.info(f"Total time taken for {total_urls} websites: {total_time_taken:.2f} seconds")

    return web_content

def chunk_text(documents, chunk_size=1000, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_texts = text_splitter.split_documents(documents)
    chunks = []
    for i, doc in enumerate(split_texts):
        chunks.append(Document(page_content=f"{doc.page_content}\nSource: {doc.metadata['source']}", metadata=doc.metadata, lookup_str=doc.lookup_str, lookup_index=i))
    return chunks

def embed_bulk_chunks(chunks, model_name="models/embedding-001", task_type="retrieval_document"):
    try:
        # Create embeddings
        embeddings = genai.embed_content(
            model=model_name,
            content=[chunk.page_content for chunk in chunks],
            task_type=task_type
        )
        return embeddings['embedding']
    except Exception as e:
        logger.error(f"An error occurred during embedding: {e}")
        return []

def embedding_gemini(chunks, tag):
    total_chunks = len(chunks)
    processed_chunks = 0
    total_processed_chunks = []
    for start_index in range(0, total_chunks, 100):
        chunk_data = chunks[start_index:start_index + 100]
        embeddings = embed_bulk_chunks(chunk_data)
        if not embeddings:
            continue

        # Process each embedding and metadata
        for i, embedding in enumerate(embeddings):
            processed_chunks += 1
            chunk = chunk_data[i]
            metadata = {"tag": tag, "source": chunk.page_content}
            total_processed_chunks.append({"id": f"{tag}_{processed_chunks}", "values": embedding, "metadata": metadata})
        logger.info(f"Processing chunk {processed_chunks}/{total_chunks}")
    
    return total_processed_chunks

def generate_answer(retrieved_chunks, query):
    context = "\n".join([chunk['metadata']['source'] for chunk in retrieved_chunks])

    prompt_parts = [
        f"input: {query}\ncontext: {context}",
        "output: ",
    ]

    response = model.generate_content(prompt_parts)
    return response.text
