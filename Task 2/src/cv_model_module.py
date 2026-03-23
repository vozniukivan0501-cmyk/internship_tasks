from keras.src.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Sequential

class CVAnimalClassifier:
    def __init__(self, num_classes=10, input_shape=(224, 224, 3) ):

        """
        Initialize model for colored image classification.
            Args:
                num_classes (int): Number of classes to classify
                input_shape (tuple): Shape of input image
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = self._build_model()

    def _build_model(self):

        """
        Using MobileNetV2 with the top layer of neurons removed for fine-tuning
        Note: base_model.trainable = False for not to interrupt in pre-trained MobileNetV2 model weights
        Returns:
             cv_model: ready to training model
        """
        base_model = MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )

        base_model.trainable = False

        cv_model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(self.num_classes, activation='softmax')
        ])

        return cv_model



