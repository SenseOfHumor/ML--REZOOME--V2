import pandas as pd
df = pd.read_csv("05_person_skills.csv")
print(df.head())

## check for NaN
print(f"Null values (before cleanup):\n{df.isnull().sum()}")

## drop the NaN values
df.dropna(inplace=True)
print(f"Null values (after cleanup):\n{df.isnull().sum()}")

## Extract skills column
skills = df["skill"].tolist()
print("Total items: ",len(skills))
print(skills[:5])   ## first 5 skills

## Format training data
TRAIN_DATA = []   ## array to hold the training data

for skill in skills:
  TRAIN_DATA.append((skill, {"entities": [(0, len(skill), "ACTION")]}))
print(TRAIN_DATA[:5])   ## first 5 entities (formatted as SpaCy training data)

## Data to SpaCy format
import spacy
from spacy.tokens import DocBin

## Creating blank SpaCy language model
nlp = spacy.blank("en")
doc_bin = DocBin()  ## DocBin to store training data

## Train data to SpaCy Doc object
for i, (text, annotations) in enumerate(TRAIN_DATA):
  doc = nlp.make_doc(text)  ## blank document with the text
  ents = []

  for start, end, label in annotations["entities"]:
    span = doc.char_span(start, end, label=label)   ## span for each entity
    if not span:
      continue
    else:
      ents.append(span)
  doc.ents = ents   ## assign the entities to the Doc
  doc_bin.add(doc)  ## add the Doc to the DocBin


  if i % 100 == 0:  # Print progress every 100 iterations
    print(f"Processed {i} entries")

  ## Save DocBin to the disk
  doc_bin.to_disk("train.spacy")
