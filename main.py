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

# Configure logging to suppress Werkzeug startup messages
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Set up Flask application
app = Flask(__name__)

# Your existing application code here...
current_jobs = set()
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "common"
BATCH_SIZE = 1000
DIMENSION = 768

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(INDEX_NAME, dimension=DIMENSION, spec=ServerlessSpec(cloud='aws', region='us-east-1'))

index = pc.Index(INDEX_NAME)

def scrape_and_store(source_id):
    try:
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
    except Exception as e:
        print("New error")
        logging.error(f"Error processing {source_id}: {str(e)}")
    finally:
        with threading.Lock():
            if source_id in current_jobs:
                current_jobs.remove(source_id)
        if source_id in current_jobs:
            current_jobs.remove(source_id)

def job():
    print("Checking for new job")
    sources_to_scrape = sources_collection.find({'isStoredAtVectorDb': False})
    for source in sources_to_scrape:
        source_id = str(source['_id'])
        if source_id in current_jobs:
            continue
        with threading.Lock():
            current_jobs.add(source_id)
        print("New job found to be pending", source_id)
        thread = threading.Thread(target=scrape_and_store, args=(source_id,))
        thread.start()

job()

@app.route('/api/project/source/link/<string:id>', methods=['POST'])
def start_scraping(id):
    try:
        _id = ObjectId(id)
        source_to_scrape = sources_collection.find_one({'_id': _id, 'isStoredAtVectorDb': False})
        if source_to_scrape:
            with threading.Lock():
                if id not in current_jobs:
                    current_jobs.add(id)
                    thread = threading.Thread(target=scrape_and_store, args=(id,))
                    thread.start()
                    return jsonify({'message': 'Scraping job started successfully.'}), 200
                else:
                    return jsonify({'error': 'Job is already in progress.'}), 400
        else:
            return jsonify({'error': 'No link sources found for the manager.'}), 404
    except Exception as e:
        logging.error(f"Error in start_scraping: {str(e)}")
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
