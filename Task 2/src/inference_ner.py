from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer
from pathlib import Path
import torch
import argparse


model_path = Path(__file__).parent.parent / 'models' / 'ner_model'


class NERInference:

    def __init__(self):
        """
        Initialize saved NER model and tokenizer
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.classes = ['butterfly','cat','chicken','cow','dog','elephant','horse','sheep','spider','squirrel']
    def predict(self, text: str):

        """
        Method to make prediction using NER model
            Args:
                text (str): user's request
            Returns: ''.join(extracted_parts) a single word, or None if no animal found
        """
        inputs = self.tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)

        predictions = torch.argmax(outputs.logits, dim=2)

        predicted_tags = predictions[0].tolist()

        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        # tokens:         ['[CLS]', 'look', 'at', 'this', 'ele', '##phant', '[SEP]']
        # predicted_tags: [   0,      0,      0,     0,      1,      1,        0 ]

        print(f"DEBUG Tokens: {tokens}")
        print(f"DEBUG Tags:   {predicted_tags}")

        extracted_parts = []

        for i in range(len(predicted_tags)):
            if predicted_tags[i] == 1:
                raw_token = tokens[i]
                if ("##" not in raw_token) and len(extracted_parts) > 0:
                    if ''.join(extracted_parts) in self.classes:
                        result = ''.join(extracted_parts)
                        return result
                    elif ''.join(extracted_parts) not in self.classes:
                        extracted_parts = []
                clean_token = tokens[i].replace('##', '')
                extracted_parts.append(clean_token)

        if ''.join(extracted_parts) in self.classes:
            return ''.join(extracted_parts)

        return None




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="NER model inference")

    parser.add_argument('--request', type=str, required=True, help="Input request")

    args = parser.parse_args()

    ner_model = NERInference()

    print(f"User's request: {args.request }")
    result = ner_model.predict(args.request)

    print(f"Predicted class: {result}")