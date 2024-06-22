from langchain.text_splitter import RecursiveCharacterTextSplitter
from db import sources_collection
from bson import ObjectId
from gemini_config import embedding_gemini
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Document:
    def __init__(self, page_content):
        self.page_content = page_content
        self.metadata = {'source': 'any.pdf', 'page': 0}
        self.lookup_str = ""
        self.lookup_index = ""

def split_text_for_pdf(pages, chunk_size=1000, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(pages)

def upsert_into_pinecone(index, formatted_chunks,namespace, BATCH_SIZE):
    try:
        logger.info("Started upserting vector embedding into Pinecone")
        for i in range(0, len(formatted_chunks), BATCH_SIZE):
            batch_data = formatted_chunks[i:i + BATCH_SIZE]
            index.upsert(vectors=batch_data, namespace=namespace)
            logger.info(f"{i + BATCH_SIZE}/{len(formatted_chunks)} Chunks upserted")
        logger.info("Upserted into Pinecone! Success")
        return True
    except:
        return False


def pdf_task_manager(_id, index):
    try:
        BATCH_SIZE=500
        source = sources_collection.find_one({ '_id': ObjectId(_id) })
        if not source:
            return False
        page_contents = source['values']
        manager = source['manager']
        pages = [Document(page) for page in page_contents]
        row_chunks = split_text_for_pdf(pages)
        sources_collection.update_one({'_id': ObjectId(_id)}, {'$set': {'chunkLength': len(row_chunks)}})
        formatted_chunks = embedding_gemini(row_chunks, _id)
        upsert_into_pinecone(index, formatted_chunks, manager, BATCH_SIZE)
        return True
    except Exception as e:
        print(e)
        return False
