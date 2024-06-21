# Groove Chatbot Backend Worker

## Overview
This repository contains the backend worker for the Groove Chatbot. Its primary purpose is to handle high CPU-consuming tasks, offloading these from the main Express.js server. By doing so, it ensures efficient processing and responsiveness of the chatbot.

## Purpose
The backend worker is designed to manage the intensive processes required for the Groove Chatbot, specifically those tasks associated with handling and processing source links submitted from the frontend.

## Features
1. **Crawl Links**:
   - The worker crawls all links related to the provided website link. This process dives deep to retrieve all nested links.

2. **Content Scraping**:
   - Scrapes the contents of each crawled link, extracting necessary information.

3. **Chunking**:
   - As the chatbot operates on a Retrieval-Augmented Generation (RAG) model, the scraped text is split into manageable chunks.

4. **Vector Embedding**:
   - Converts the chunked text into vector embeddings. This enables efficient semantic search capabilities.

5. **Store Vector and Chunks**:
   - Formats and stores the chunked text and their corresponding vector embeddings into a Pinecone database.

## Workflow
1. **Submission of Source Links**:
   - The frontend submits source links to the backend worker.

2. **Link Crawling**:
   - The worker initiates a crawling process to retrieve all relevant links from the provided website.

3. **Content Scraping**:
   - Extracts content from each of the crawled links.

4. **Text Chunking**:
   - The scraped content is divided into smaller, manageable text chunks.

5. **Vector Embedding Generation**:
   - Each text chunk is converted into a vector embedding for semantic search.

6. **Storage**:
   - The chunked text and vector embeddings are stored in the Pinecone database for future retrieval and use by the chatbot.

## Technologies Used
- **Fastapi**: Main server framework for handling requests.
- **Pinecone**: Vector database for storing and managing vector embeddings.
- **Web Scraping Library**: langchain
