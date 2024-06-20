import os
os.environ['USER_AGENT'] = 'pugal'

from flask import Flask, jsonify, request
from bson import ObjectId
import threading
import logging
import schedule
import time
from db import sources_collection
from pinecone import Pinecone, ServerlessSpec
from functions import extract_path_from_url
from scraper import crawl
import gemini_config
from config import PINECONE_API_KEY

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up Flask application
app = Flask(__name__)

current_job = [None]
job_queue = []
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "common"
BATCH_SIZE = 1000
DIMENSION = 768

# Ensure the index exists
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(INDEX_NAME, dimension=DIMENSION, spec=ServerlessSpec(cloud='aws', region='us-east-1'))

index = pc.Index(INDEX_NAME)

def scrape_and_store(source_id):
    try:
        logger.info(f"Starting scrape_and_store for source_id: {source_id}")
        source_to_scrape = sources_collection.find_one({'_id': ObjectId(source_id)})
        if source_to_scrape and not source_to_scrape['isScraped']:
            url, domain = extract_path_from_url(source_to_scrape['tag'])
            links = crawl(url, domain)
            sources_collection.update_one({'_id': ObjectId(source_id)}, {'$set': {'isScraped': True, 'values': links}})
            status = gemini_config.extract_and_store(index, BATCH_SIZE, links, str(source_to_scrape['_id']), str(source_to_scrape['manager']))
            if status:
                sources_collection.update_one({'_id': ObjectId(source_id)}, {'$set': {'isStoredAtVectorDb': True}})
        elif source_to_scrape and source_to_scrape['isScraped'] and not source_to_scrape['isStoredAtVectorDb']:
            links = source_to_scrape['values']
            status = gemini_config.extract_and_store(index, BATCH_SIZE, links, str(source_to_scrape['_id']), str(source_to_scrape['manager']))
            if status:
                sources_collection.update_one({'_id': ObjectId(source_id)}, {'$set': {'isStoredAtVectorDb': True}})
        logger.info(f"Finished scrape_and_store for source_id: {source_id}")
    except Exception as e:
        logger.error(f"Error processing {source_id}: {str(e)}", exc_info=True)
    finally:
        with threading.Lock():
            current_job[0] = None
            if job_queue:
                next_job = job_queue.pop(0)
                current_job[0] = next_job
                thread = threading.Thread(target=scrape_and_store, args=(next_job,))
                thread.start()

def job():
    logger.info("Searching for job")
    sources_to_scrape = sources_collection.find({'isStoredAtVectorDb': False})
    for source in sources_to_scrape:
        source_id = str(source['_id'])
        if current_job[0] == source_id:
            continue
        if source_id in job_queue:
            continue
        job_queue.append(source_id)

    if not current_job[0] and job_queue:
        next_job = job_queue.pop(0)
        current_job[0] = next_job
        thread = threading.Thread(target=scrape_and_store, args=(next_job,))
        thread.start()

logger.info("Started searching for first job")
job()

@app.route('/api/project/source/link/<string:id>', methods=['POST'])
def start_scraping(id):
    try:
        _id = ObjectId(id)
        source_to_scrape = sources_collection.find_one({'_id': _id, 'isStoredAtVectorDb': False})
        if source_to_scrape:
            with threading.Lock():
                if id not in job_queue and id != current_job[0]:
                    job_queue.append(id)
                    return jsonify({'message': 'Scraping job queued successfully.'}), 200
                else:
                    return jsonify({'error': 'Job is already in progress or queued.'}), 400
        else:
            return jsonify({'error': 'No link sources found for the manager.'}), 404
    except Exception as e:
        logger.error(f"Error in start_scraping: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

def run_scheduler():
    schedule.every(1).minutes.do(job)
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    scheduler_thread = threading.Thread(target=run_scheduler)
    scheduler_thread.start()
    app.run(debug=False, host='0.0.0.0', port=5000)


# class Document:
#     def __init__(self, page_content, metadata):
#         self.page_content = page_content
#         self.metadata = metadata