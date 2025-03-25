import json
from pymongo import MongoClient

# MongoDB connection
client = MongoClient('localhost', 27017)
db = client['medical_database']

# collection creation
descriptions_collection = db['descriptions']
cases_collection = db['cases']

with open('MedPix-2-0/Descriptions.json') as desc_file:
    descriptions_data = json.load(desc_file)
    
with open('MedPix-2-0/Case_topic.json') as case_file:
    cases_data = json.load(case_file)

descriptions_collection.insert_many(descriptions_data)
cases_collection.insert_many(cases_data)

# use 'U_id' as index
descriptions_collection.create_index('U_id')
cases_collection.create_index('U_id')

print("Data loaded and index created!")
