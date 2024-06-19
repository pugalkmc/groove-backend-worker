from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from bson import ObjectId
import google.generativeai as genai
import time
import sys
from db import sources_collection
from config import GOOGLE_API_KEY

# Configure the Google Generative AI API

import json
import os
import sys
import logging
from bson import ObjectId

PROGRESS_FILE = "progress.json"

genai.configure(api_key=GOOGLE_API_KEY)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as file:
            return json.load(file)
    return {}

def save_progress(progress):
    with open(PROGRESS_FILE, 'w') as file:
        json.dump(progress, file)

def extract_and_store(index, BATCH_SIZE, links, _id, namespace):
    progress = load_progress()
    current_progress = progress.get(_id, {"step": "start"})

    try:
        if current_progress["step"] == "start":
            logger.info(f"Step: start. Scraping websites for ID: {_id}")
            web_content = estimate_and_scrap_websites(links)
            progress[_id] = {"step": "scraped", "web_content": web_content}
            save_progress(progress)
        else:
            web_content = current_progress["web_content"]

        if current_progress["step"] in ["start", "scraped"]:
            logger.info(f"Step: scraped. Chunking text for ID: {_id}")
            chunks = chunk_text(web_content)
            sources_collection.update_one({'_id': ObjectId(_id)}, {'$set': {'chunkLength': len(chunks)}})
            progress[_id] = {"step": "chunked", "chunks": chunks}
            save_progress(progress)
        else:
            chunks = current_progress["chunks"]

        if current_progress["step"] in ["start", "scraped", "chunked"]:
            logger.info(f"Step: chunked. Embedding chunks for ID: {_id}")
            formatted_chunks = embedding_gemini(chunks, _id)
            progress[_id] = {"step": "formatted", "formatted_chunks": formatted_chunks, "chunks_processed": 0}
            save_progress(progress)
        else:
            formatted_chunks = current_progress["formatted_chunks"]

        if current_progress["step"] in ["start", "scraped", "chunked", "formatted"]:
            logger.info(f"Step: formatted. Upserting chunks into Pinecone for ID: {_id}")
            chunks_to_process = formatted_chunks[current_progress["chunks_processed"]:]
            for i in range(0, len(chunks_to_process), BATCH_SIZE):
                batch_data = chunks_to_process[i:i + BATCH_SIZE]
                index.upsert(vectors=batch_data, namespace=namespace)
                sys.stdout.write("\r")
                sys.stdout.write(f"{current_progress['chunks_processed'] + i + 1} Chunks upserted")
                sys.stdout.flush()

                progress[_id] = {
                    "step": "formatted",
                    "formatted_chunks": formatted_chunks,
                    "chunks_processed": current_progress["chunks_processed"] + i + BATCH_SIZE
                }
                save_progress(progress)

            logger.info(f"Upserted into Pinecone! Success for ID: {_id}")
            del progress[_id]
            save_progress(progress)

    except Exception as e:
        logger.error(f"An error occurred for ID: {_id}: {e}")
        return False

    return True


def extract_text_from_website(url):
    loader = WebBaseLoader(url)
    documents = loader.load()
    return documents

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
        documents = extract_text_from_website(active_url)
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

        # Update the screen with the estimated time
        sys.stdout.write("\r")
        sys.stdout.write(f"Processed {idx + 1}/{total_urls} websites. Estimated remaining time: {estimated_remaining_time:.2f} seconds")
        sys.stdout.flush()

    if len(active_url) > 0:
        documents = extract_text_from_website(active_url)
        web_content.extend(documents)
        active_url = []

    end_time = time.time()
    total_time_taken = end_time - start_time
    logger.info(f"\nTotal time taken for {total_urls} websites: {total_time_taken:.2f} seconds")

    return web_content

def chunk_text(documents, chunk_size=1000, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)

def embed_bulk_chunks(chunks, model_name="models/embedding-001", task_type="retrieval_document"):
    tries = 1
    while tries<=5:
        try:
            # Create embeddings
            embeddings = genai.embed_content(
                model=model_name,
                content=chunks,
                task_type=task_type
            )
            return embeddings['embedding']
        except Exception as e:
            tries += 1
            logger.info(f"An error occurred: {e}")
    return []

def embedding_gemini(chunks, tag):
    chunks = [chunk.page_content for chunk in chunks]
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
            metadata = {"tag": tag, "source": chunk_data[i]}
            total_processed_chunks.append({"id": f"{tag}_{processed_chunks}", "values": embedding, "metadata": metadata})
        sys.stdout.write("\r")
        sys.stdout.write(f"Processing chunk {processed_chunks}/{total_chunks}")
        sys.stdout.flush()

    return total_processed_chunks

def generate_answer(retrieved_chunks, query):
    context = "\n".join(retrieved_chunks)

    prompt_parts = [
        f"input: {query}\ncontext: {context}",
        "output: ",
    ]

    response = model.generate_content(prompt_parts)
    return response.text
