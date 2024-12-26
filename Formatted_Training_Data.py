import spacy
import json
from spacy.tokens import DocBin
from Data_Train_JSON import convert_training_data
## Creating blank SpaCy language model
nlp = spacy.blank("en")
doc_bin = DocBin()  ## DocBin to store training data

## convert the training file
entity_data = convert_training_data("train.json")

def convert_to_spans(entity_data):
    '''
    Converts the entity dats to spans format

    Extended description of function:
    This function takes the entity data and converts it to the spans format
    which is required because simple NER does not allow overlapping spans

    Returns:
    span_data: List of tuples
    '''
    span_data = []

    for text, annotations in entity_data:
        spans = annotations["entities"]
        span_data.append((text, {"spans": spans}))
    return span_data


def get_labels(data):
    '''
    Extracts unique labels from the data

    Extended description of function:
    This function takes the data and extracts the unique labels from the data

    Returns:
    labels: List
    '''
    labels = []

    for _, annotations in data: ## the _ is a placeholder for the text
        for entity in annotations["entities"]:
            label = entity[2]
            if label not in labels:
                labels.append(label)
    return labels


TRAIN_DATA = convert_to_spans(entity_data)
labels = get_labels(entity_data)

## Adding SpanCategorizer component
spancat = nlp.add_pipe("spancat", config={"spans_key": "my_spans"})

## Adding labels to the SpanCategorizer
for label in labels:
    spancat.add_label(label)

## TRAIN_DATA to SpaCy Doc object
doc_bin = DocBin()  ## DocBin to store training data

for text, annotations in TRAIN_DATA:
    doc = nlp.make_doc(text)
    spans = [       ## create spans
        doc.char_span(start, end, label=label)
        for start, end, label in annotations["spans"]
        if doc.char_span(start, end, label=label) is not None
    ]
    doc.spans["my_spans"] = spans   ## assigning the spans to the my_spans key
    doc_bin.add(doc)  ## add the Doc to the DocBin

## Save DocBin to the disk
doc_bin.to_disk("training_data.spacy")


