import numpy as np
import argparse
from pathlib import Path
from keras.src.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Sequential

from tensorflow.keras.preprocessing.image import load_img, img_to_array


model_path = Path(__file__).parent.parent / 'models' / 'cv_model.keras'
base_image_dir = Path(__file__).parent.parent / 'data' / 'animals10'

class CVInference:

    def __init__(self):
        """
        Initialize the CV Inference class implements loaded weights of CV model.
        """
        self.base_model = MobileNetV2(
            input_shape= (224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        self.base_model.trainable = False
        self.model = Sequential([
            self.base_model,
            GlobalAveragePooling2D(),
            Dense(10, activation='softmax')
        ])

        self.model.load_weights(model_path)


        self.classes =  ['butterfly','cat','chicken','cow','dog','elephant','horse','sheep','spider','squirrel']

    def predict(self, img_path):

        """
        Function for image classification
            Args:
                img_path: Path to the image to be classified
            Returns:
                prediction (str): Image's predicted class name
        """

        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        prediction = self.model.predict(image)
        prediction = np.argmax(prediction)

        prediction = self.classes[prediction]

        return prediction


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CV model inference")


    parser.add_argument('--image_path', type=str, required=True, help="Image path (class dir and name)")

    args = parser.parse_args()

    cv_model = CVInference()

    print(f"Picture to classify: {base_image_dir / args.image_path}")
    result = cv_model.predict(base_image_dir / args.image_path)

    print(f"Predicted class: {result}")