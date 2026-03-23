import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from transformers import utils
utils.logging.set_verbosity_error()

from src.inference_cv import CVInference
from src.inference_ner import NERInference
import argparse
from pathlib import Path


base_image_dir = Path(__file__).parent.parent / 'data' / 'animals10'

class PipeLine:
    """
    Initialize PipeLine for 2 models predictions comparison
    """
    def __init__(self):
        self.ner_inference = NERInference()
        self.cv_inference = CVInference()

    def check_match(self, request, image_path):

        """
        Function to check if CV model's prediction matches NER model's prediction
            Args:
                request (str): User's text request
                image_path (str): Path to the image to classify
            Output:
                check (bool): True if predictions are same, False if they are not
        """

        ner_output = self.ner_inference.predict(request)
        cv_output = self.cv_inference.predict(image_path)

        print(f"Found in text: {ner_output}")
        print(f"Found in image: {cv_output}")

        if ner_output == cv_output:
            return True

        else:
            return False

def main():

    arg_parser = argparse.ArgumentParser(description='Final pipeline')

    arg_parser.add_argument('--request', type=str ,required=True, help='Text request')
    arg_parser.add_argument('--image_path', type=str ,required=True, help='Path to the image (class dir and name only)')

    args = arg_parser.parse_args()

    pipeline =PipeLine()
    check = pipeline.check_match(args.request, base_image_dir / args.image_path)
    print(check)

    return check

if __name__ == '__main__':
    main()
