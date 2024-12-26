import spacy
import json
from spacy.tokens import DocBin

## Creating blank SpaCy language model
nlp = spacy.blank("en")
doc_bin = DocBin()  ## DocBin to store training data

## loading the json file 
with open('train.json', 'r') as file:
    data = json.load(file)

## converting to SpaCy format
TRAIN_DATA = []

for item in data:
    text = item["content"]  ## grab content
    entities = []

    ## process annotations
    for annotation in item["annotation"]:
        label = annotation["label"]  ## grab individual label

        for point in annotation["points"]:
            start = point["start"]
            end = point["end"]
            entities.append((start, end, label))

    
    ## append to training data
    TRAIN_DATA.append((text, {"entities": entities}))

## Train data to SpaCy Doc Object
for text, annotations in TRAIN_DATA: