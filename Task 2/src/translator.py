import os
from pathlib import Path

"""
Translating italian by-default dir names to english
"""

dataset_path = Path(__file__).parent.parent / "data" / "animals10"

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

for file_name in os.listdir(dataset_path):
    current_path = os.path.join(dataset_path, file_name)

    if file_name in translate.keys():
        os.rename(current_path, os.path.join(dataset_path, translate[file_name]))

