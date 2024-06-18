from pymongo import MongoClient
from config import MONGODB_URL

client = MongoClient(MONGODB_URL)
db = client['aibot']
sources_collection = db['sources']