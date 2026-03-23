import random
import json
from pathlib import Path
data_save_path = Path(__file__).parent.parent / "data" / "ner_data"

"""
Initialize synthetic request generator for DistilBert model fine-tuning
Saving .json request dataset 
"""
translate = {
    "cane": "dog",
    "cavallo": "horse",
    "elefante": "elephant",
    "farfalla": "butterfly",
    "gallina": "chicken",
    "gatto": "cat",
    "mucca": "cow",
    "pecora": "sheep",
    "scoiattolo": "squirrel",
    "ragno": "spider"
}
target_animals = []

for val in translate.values():
    target_animals.append(val)

templates = [
    "Is it {target} ?",
    "Is it {target}",
    "Is it a photo of {target}",

    "It is {target} here",
    "It is {target} on this picture",
    "It is {target}",

    "It looks like {target} , am I right ?",
    "It looks like {target} , isn't it ?",
    "It looks like {target} , how do you think ?",

    "Recently I saw a strange animal , it looks like {target} , but I am not sure",
    "Recently I saw a strange animal , it looks like {target} , do you recognize it ?",

    "{target}",
    "{target} ?",
    "A {target} ?",

    "So cute {target} !",
    "Look at this {target} , so beautiful",
    "Look what I saw yesterday , what a pretty {target}",

    "This {target} looks ordinal , nothing special",
    "Is this {target} healthy ? Looks pretty strange"
]

data_set = {'tokens' : [] , 'ner_tags' : []}

for _ in range(1000):
    target = random.choice(target_animals)
    template = random.choice(templates)

    sentence = template.format(target=target)
    tokens = sentence.split()
    tags = [1 if word in target_animals else 0 for word in tokens]

    data_set['tokens'].append(tokens)
    data_set['ner_tags'].append(tags)


with open(data_save_path / "ner_dataset.json", "w", encoding="utf-8") as f:
    json.dump(data_set, f, indent=4)

print("Dataset saved as ner_dataset.json")
