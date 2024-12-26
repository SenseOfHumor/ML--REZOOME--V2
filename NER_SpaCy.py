import pandas as pd
import spacy
import random
from spacy.training import Example
from tqdm import tqdm

## Load and clean the data
df = pd.read_csv("job_skills.csv")
df = df[["Title"]].dropna().drop_duplicates()
titles = df["Title"].tolist()

title_list = []
for title in titles:
    if len(title.split(",")) > 1:

        ## split the titles at ","
        split_titles = title.split(",")

        ## remove any leading or trailing whitespaces
        split_titles = [t.strip() for t in split_titles]
        title_list.extend(split_titles)

# print("Unique job titles:", len(titles))
# print("Total job titles:", len(title_list))

def create_training_data(titles: list, label="SKILL") -> list:
    '''
    Creates training data for SpaCy NER model

    Extended description of function:
    Function takes a list of titles and a label as input.
    It then creates training data in the format required for
    training a SpaCy NER model.

    Returns:
    training_data: List of tuples
    '''
    training_data = []

    for title in titles:
        start = 0
        end = len(title)
        training_data.append(
            (title, {"entities": [(start, end, label)]})
        )
    return training_data

TRAIN_DATA = create_training_data(title_list, label="SKILL")
print(TRAIN_DATA[:5])

## Creating a blank SpaCy model
nlp = spacy.blank("en")

## creating a NER pipeline
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner", last=True)

else:
    ner = nlp.get_pipe("ner")

## adding labels to the NER component
ner.add_label("SKILL")

## Init 
nlp.initialize()

## Create an Example object
training_examples = []
for text, annotations in TRAIN_DATA:
    doc = nlp.make_doc(text)
    example = Example.from_dict(doc, annotations)
    training_examples.append(example)

# Training the NER model
num_iterations = 100
with tqdm(total=num_iterations, desc="Training NER", unit=" epoch") as pbar:    ## progress bar
    for iteration in range(num_iterations):
        random.shuffle(training_examples)
        losses = {}

        # Update model
        nlp.update(
            training_examples,
            drop=0.3,
            losses=losses
        )

        pbar.update(1)

# 8) Save the model to disk
nlp.to_disk("ner_skill_model")
print("Model saved to disk at 'ner_skill_model'")