from flask import Blueprint, render_template, request, jsonify
import json
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import ast
import json
from datetime import datetime
import re

app_blueprint=Blueprint('app_blueprint',__name__,static_folder="./assets")

@app_blueprint.route('/')
def index():
    data={
        'content_types' :load_word_dictionary('../models/vocab/contenttypes.json'),
        'social_media' : load_word_dictionary('../models/vocab/network.json'),
        'tags' :load_word_dictionary('../models/vocab/taglist.json'),
        "icons":  {
            "Video": "fas fa-video",
            "Carousel": "fas fa-images",
            "Photo": "fas fa-camera",
            "Document": "fas fa-file-pdf",
            "Text": "fas fa-text-slash",
            "Link": "fas fa-external-link-alt",
        }
    }
    
    tags_json = json.dumps(data['tags'])

    return render_template('index.html', data=data, tags_json=tags_json)

@app_blueprint.route('/pred', methods=['GET', 'POST'])
@app_blueprint.route('/pred', methods=['GET', 'POST'])
def pred():
    predict = {}
    if request.method == 'POST':
        data = request.json
        content_types = load_word_dictionary('../models/vocab/contenttypes.json')
        social_media = load_word_dictionary('../models/vocab/network.json')
        tags = load_word_dictionary('../models/vocab/taglist.json')
        dur = [2592000]
        for network in data["networks"]:
            ntext=findbyvalue(network+1, social_media)
            model = joblib.load(f"../models/trained_model_{ntext}.joblib")
            predict[ntext]={}
            for content in data["content"]:
                ctext = findbyvalue(content+1, content_types)
                predict[ntext][ctext] = {}
                d = {
                    'content_type': [content],
                    'network': [network],
                    'tags': [len(data["tags"])],
                    'phone_numbers': [len(extract_phone_numbers(data["post"]))],
                    'emails': [len(extract_emails(data["post"]))],
                    'hashtags': [len(extract_hashtags(data["post"]))],
                    'charcount': [len(data["post"])],
                    'link_count': [count_links(data["post"])]
                }
                p = pd.DataFrame(d)
                prediction= model.predict(p),
                prediction=prediction[0][0]
                # print(prediction)
                predic={
                    'engagements':prediction[0], 
                    'impressions':prediction[1], 
                    'reactions':prediction[2], 
                    'likes':prediction[3], 
                    'engaged_users':prediction[4],
                    'subscribers':prediction[5],
                    "profile_clicks":prediction[6],
                    "shares":prediction[7]
                }
                    # print(predic)
                predict[ntext][ctext] = predic

    return json.dumps(predict)

def findbyvalue(value_to_find,original_dict):
    keys_with_value_1 = [key for key, value in original_dict.items() if value == value_to_find]
    return keys_with_value_1[0]

def load_word_dictionary(file_path):
    try:
        with open(file_path, 'r') as json_file:
            return json.load(json_file)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    
# Function to extract hashtags from a text
def extract_hashtags(text):
    return re.findall(r'#\w+', text)

# Function to extract emails from a text
def extract_emails(text):
    return re.findall(r'\S+@\S+', text)

# Function to extract phone numbers from a text
def extract_phone_numbers(text):
    return re.findall(r'\+?\d{10,12}', text)

# Function to count the number of links in a text
def count_links(text):
    return len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))
