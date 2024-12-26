import json

TRAIN_DATA = []

def fix_training_data(file_path):
    '''
    Re-Formats unstructured JSON data into an array of JSON objects

    Extended description of function:
    Functions takes in a file path to a JSON file, reads the file
    and updates the structure of the JSON data.

    Returns:
    None
    '''

    # Path to the improperly formatted JSON file
    file_path = "pretrain.json"

    # Read the file and fix the structure
    with open(file_path, "r") as file:
        lines = file.readlines()  # Read all lines
        data = [json.loads(line) for line in lines]  # Parse each line as JSON

    # Save the fixed JSON as an array
    with open("train.json", "w") as fixed_file:
        json.dump(data, fixed_file, indent=4)


def convert_training_data(file_path):
    '''
    Converts JSON data to SpaCy training data format

    Extended description of function:
    Function reads a JSON file, extracts the content and annotations
    and converts the data to SpaCy training data format.

    Returns:
    TRAIN_DATA: List of tuples
    '''
    ## loading the json file 
    with open('train.json', 'r') as file:
        data = json.load(file)

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
    return TRAIN_DATA


## check workings
fix_training_data("pretrain.json")
data = convert_training_data("train.json")
print(data[:2])