import argparse
import json
from pathlib import Path

from datasets import Dataset
from ner_model_module import NERAnimalModel
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)

data_path = Path(__file__).parent.parent / "data" / "ner_data" / "ner_dataset.json"
save_path = Path(__file__).parent.parent / "models" / "ner_model"
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize_and_align_labels(data):

    """
    Labeling data for model training, solving the problem of length missmatch of tokenized and labeled data"
        Args:
            data (dict): read from json file dataset for model training expecting {tokens: [] , ner_tags: []}
        Returns:
            tokenized_inputs (dict): training ready data, expected shape {labels: [] , ner_tags: []}
    """

    tokenized_inputs = tokenizer(data["tokens"], truncation=True, is_split_into_words=True)

    new_labels = []

    for i, original_tags in enumerate(data['ner_tags']):

        word_ids = tokenized_inputs.word_ids(batch_index=i)

        aligned_tags_for_sentence = []
        previous_word_idx = None

        for word_idx in word_ids:

            if word_idx == None:
                aligned_tags_for_sentence.append(-100)

            elif word_idx != previous_word_idx:
                aligned_tags_for_sentence.append(original_tags[word_idx])

            else:
                aligned_tags_for_sentence.append(-100)

            previous_word_idx = word_idx

        pass
        new_labels.append(aligned_tags_for_sentence)

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


def main():

    """
    Training script with arguments parsing implementation.
    Note: all args have default values, script can run without arguments input.
    """

    parser = argparse.ArgumentParser(description='Train NER model')

    parser.add_argument('--data_path', default= data_path, type = str, help='Path to the data directory')
    parser.add_argument('--epochs', default= 1, type = int, help='Number of epochs')
    parser.add_argument('--lr', default= 2e-5, type = int, help='Learning rate')
    parser.add_argument('--batch_size', default= 32, type = int, help='Batch size')
    parser.add_argument('--save_path', default= save_path, type = str, help='Save path')

    args = parser.parse_args()

    with open(args.data_path, 'r' , encoding = 'utf-8') as f:
        data_dict = json.load(f)

    ner_dataset = Dataset.from_dict(data_dict)

    tokenized_dataset = ner_dataset.map(tokenize_and_align_labels, batched=True)

    ner_classifier = NERAnimalModel()
    ner_classifier_model = ner_classifier.model

    data_collator = DataCollatorForTokenClassification(tokenizer)

    training_args = TrainingArguments(
        output_dir="./models/ner_checkpoints",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=10,
    )

    trainer = Trainer(
        model=ner_classifier_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    print(f"Model saved in {args.save_path}")
    trainer.save_model(args.save_path)


if __name__ == '__main__':
    main()


