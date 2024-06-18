import os
os.environ['USER_AGENT'] = 'pugal'
from flask import Flask, jsonify, request
from bson import ObjectId
import threading
import os
from functions import extract_path_from_url
from scraper import crawl
import gemini_config
import logging
import schedule
import time
from db import sources_collection
from pinecone import Pinecone, ServerlessSpec
# from flask_cors import CORS
from config import PINECONE_API_KEY

current_jobs = set()

app = Flask(__name__)
# CORS(app)

# @app.before_request
# def before_request():
#     if request.remote_addr == 'specific_server_ip':
#         CORS(app, resources={r"/*": {"origins": "http://specificserver.com"}})
#     else:
#         CORS(app, resources={r"/*": {"origins": "*"}})

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
        logging.error(f"Error processing {source_id}: {str(e)}")
    finally:
        current_jobs.remove(source_id)

def job():
    sources_to_scrape = sources_collection.find({'isStoredAtVectorDb': False})
    for source in sources_to_scrape:
        source_id = str(source['_id'])
        if source_id in current_jobs:
            continue
        if not source.get('isScraped'):
            current_jobs.add(str(source.get('_id')))
            thread = threading.Thread(target=scrape_and_store, args=(str(source['_id']),))
            thread.start()
        elif source.get('isScraped') and not source.get('isStoredAtVectorDb'):
            current_jobs.add(str(source.get('_id')))
            thread = threading.Thread(target=scrape_and_store, args=(str(source['_id']),))
            thread.start()

job()

@app.route('/api/project/source/link/<string:id>', methods=['POST'])
def start_scraping(id):
    try:
        _id = ObjectId(id)
        source_to_scrape = sources_collection.find_one({'_id': _id, 'isStoredAtVectorDb': False})
        if source_to_scrape:
            thread = threading.Thread(target=scrape_and_store, args=(id,))
            thread.start()
            return jsonify({'message': 'Scraping job started successfully.'}), 200
        else:
            return jsonify({'error': 'No link sources found for the manager.'}), 404
    except Exception as e:
        logging.error(f"Error in start_scraping: {str(e)}")
        return jsonify({'error': str(e)}), 500

def run_scheduler():
    schedule.every(10).minutes.do(job)
    while True:
        schedule.run_pending()
        time.sleep(1)

logging.basicConfig(level=logging.INFO)
scheduler_thread = threading.Thread(target=run_scheduler)
scheduler_thread.start()
