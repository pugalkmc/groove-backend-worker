from pymongo import MongoClient

client = MongoClient('mongodb+srv://pugalkmc:pugalkmc@cluster0.dzcnjxc.mongodb.net')
db = client['aibot']
sources_collection = db['sources']