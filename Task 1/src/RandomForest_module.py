from MnistClassifierInterface_module import MnistClassifierInterface
from sklearn.ensemble import RandomForestClassifier

class RandomForest(MnistClassifierInterface):

    def __init__(self):
        """
        Initialize the random forest classifier.
        """

        self.model = RandomForestClassifier(random_state=42)


    def train(self, X_train, y_train):
        """
        Trains Random Forest model
        Note: Reshaping 3D input arrays into 2D arrays (n_samples, width * height) to fulfill sklearn input requirements
            Args:
                X_train: array-like training data expected shape (n_samples, width, height)
                y_train: array-like training data expected shape (n_samples)
        """

        X_train_flat = X_train.reshape(len(X_train), -1)
        self.model.fit(X_train_flat, y_train)


    def predict(self, X_test):
        """
        Classifying using trained model
            Note: Reshaping 3D input arrays into 2D arrays (n_samples, width * height)
            Args:
                X_test: array-like test data expected shape (n_samples, width, height)
            Returns:
                preds: array-like prediction of integer class values, expected shape (n_samples)
        """


        X_test_flat = X_test.reshape(len(X_test), -1)
        preds = self.model.predict(X_test_flat)

        return preds