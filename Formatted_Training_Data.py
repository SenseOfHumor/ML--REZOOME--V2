import spacy
import json
from spacy.tokens import DocBin
from spacy.training import Example

# 1) Create a blank SpaCy pipeline
nlp = spacy.blank("en")

# 2) Function to read line-delimited JSON and convert to (text, {"entities": ...}) 
def convert_training_data(line_delimited_file):
    """
    Reads line-delimited JSON (each line is one JSON object).
    Converts each record to (text, {"entities": [(start, end, label), ...]})
    """
    TRAIN_DATA = []

    # Read and parse each line
    with open(line_delimited_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            text = item["content"]
            entities = []

            for annotation in item["annotation"]:
                # Skip if no label
                if not annotation["label"]:
                    print("Empty label found:", annotation)
                    continue

                label = annotation["label"][0]  # single label
                for point in annotation["points"]:
                    start = point["start"]
                    end   = point["end"]
                    entities.append((start, end, label))

            TRAIN_DATA.append((text, {"entities": entities}))
    return TRAIN_DATA

# 3) Convert the training data to "spans" format
def convert_to_spans(training_data):
    """
    Converts (text, {"entities": [(start, end, label), ...]})
    to (text, {"spans": {"my_spans": [(start, end, label), ...]}})
    """
    span_data = []
    for text, annot_dict in training_data:
        entities = annot_dict["entities"]
        span_data.append((text, {"spans": {"my_spans": entities}}))
    return span_data

# 4) Read from the original, line-delimited JSON
raw_training_data = convert_training_data("pretrain.json")

# 5) Convert "entities" to "spans"
TRAIN_DATA = convert_to_spans(raw_training_data)

# 6) Set up your labels
labels = [
    "Skills", "College Name", "Graduation Year", "Designation",
    "Companies worked at", "Email Address", "Location",
    "Name", "Degree", "Years of Experience"
]

# 7) Add SpanCategorizer to pipeline
spancat = nlp.add_pipe("spancat", config={"spans_key": "my_spans"})
print("Labels in spancat before initialization:", spancat.labels)

# 8) Add labels to spancat
for label in labels:
    spancat.add_label(label)
    print(f"Added label: {label}")

# 9) Convert TRAIN_DATA → spaCy Examples, ensuring valid Span objects
examples = []
for text, annots in TRAIN_DATA:
    doc = nlp.make_doc(text)
    
    # Build actual Span objects
    my_spans = []
    for start, end, label in annots["spans"]["my_spans"]:
        # Validate offsets (avoid negative or out-of-bounds)
        if 0 <= start < end <= len(doc.text):
            span = doc.char_span(start, end, label=label)
            if span is not None:
                my_spans.append(span)
        else:
            # Debug: if you want to see invalid offsets
            print(f"Invalid span: ({start}, {end}) in text: '{text[start:end]}'")
    
    example = Example.from_dict(doc, {"spans": {"my_spans": my_spans}})
    examples.append(example)

# 10) Initialize spancat on examples
spancat.initialize(lambda: examples, nlp=nlp)
print("Labels in spancat after initialization:", spancat.labels)

# 11) Build a DocBin and store each doc’s “my_spans”
doc_bin = DocBin()
for text, annots in TRAIN_DATA:
    doc = nlp.make_doc(text)
    doc_spans = []
    for start, end, label in annots["spans"]["my_spans"]:
        # same check as above
        if 0 <= start < end <= len(doc.text):
            span = doc.char_span(start, end, label=label)
            if span:
                doc_spans.append(span)
    doc.spans["my_spans"] = doc_spans
    doc_bin.add(doc)

# 12) Finally, save out the DocBin
doc_bin.to_disk("train_data.spacy")
print("Training data saved to train_data.spacy")
