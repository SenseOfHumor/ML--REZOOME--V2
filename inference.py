import spacy

nlp = spacy.load("./output/model-best")  # Load your trained model
text = "Michael Brown is an AI engineer at DataCorp skilled in NLP, TensorFlow, and PyTorch."
doc = nlp(text)

print("Entities detected:")
for ent in doc.ents:
    print(f"Entity: {ent.text}, Label: {ent.label_}")

print("\nTokens and their attributes:")
for token in doc:
    print(f"Token: {token.text}, Is entity: {token.ent_type_}")
