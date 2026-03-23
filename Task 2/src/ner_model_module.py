from transformers import AutoModelForTokenClassification

class NERAnimalModel:

    def __init__(self, model_checkpoint='distilbert-base-uncased', num_labels=2):
        """
        Initialization of NER model for 2-classes of tokens
        """
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_checkpoint,
            num_labels=num_labels
        )