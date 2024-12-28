import spacy
from spacy.tokens import DocBin

# Initialize a blank SpaCy model
nlp = spacy.blank("en")

# Prepare the synthetic dataset
synthetic_resumes = [
    {
        "text": "John Doe is a software engineer with 5 years of experience at TechCorp, skilled in Python, JavaScript, and cloud computing.",
        "entities": [
            (0, 8, "PERSON"),
            (12, 29, "JOB_TITLE"),
            (59, 66, "ORG"),
            (77, 83, "SKILL"),
            (85, 96, "SKILL"),
            (102, 117, "SKILL"),
        ],
    },
    {
        "text": "Jane Smith is a data scientist proficient in Python, R, and machine learning. She holds a Master's degree in Data Science from Stanford University.",
        "entities": [
            (0, 10, "PERSON"),
            (14, 29, "JOB_TITLE"),
            (46, 52, "SKILL"),
            (54, 55, "SKILL"),
            (61, 77, "SKILL"),
            (104, 122, "DEGREE"),
            (126, 138, "FIELD"),
            (144, 164, "ORG"),
        ],
    },
]

# Convert to SpaCy's DocBin format
doc_bin = DocBin()
for entry in synthetic_resumes:
    doc = nlp.make_doc(entry["text"])
    ents = []
    for start, end, label in entry["entities"]:
        span = doc.char_span(start, end, label=label)
        if span is None:
            print(f"Skipping entity in '{entry['text']}' due to misaligned boundaries: {(start, end, label)}")
        else:
            ents.append(span)
    doc.ents = ents
    doc_bin.add(doc)

# Save to disk
doc_bin.to_disk("train.spacy")
