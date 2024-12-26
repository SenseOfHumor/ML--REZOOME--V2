import json




# Path to the improperly formatted JSON file
file_path = "pretrain.json"

# Read the file and fix the structure
with open(file_path, "r") as file:
    lines = file.readlines()  # Read all lines
    data = [json.loads(line) for line in lines]  # Parse each line as JSON

# Save the fixed JSON as an array
with open("train.json", "w") as fixed_file:
    json.dump(data, fixed_file, indent=4)


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


## check format
print(TRAIN_DATA[:2])