import tensorflow as tf
from translator import dataset_path
import argparse
from pathlib import Path
from cv_model_module import CVAnimalClassifier

data_dir = dataset_path
models_dir = Path(__file__).parent.parent / 'models'


def main():

    """
    Script for argument parsing and cv_model training
    Note: All args have default values, script can run without args input
    """

    parser = argparse.ArgumentParser(description="Train CV Model for Animals10")

    parser.add_argument('--data_dir', type=str, default=data_dir, help='Path to the dataset')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--save_path', type=str, default=f'{models_dir}/cv_model.keras', help='Where to save the model')

    args = parser.parse_args()

    print(f"Dataset path: {args.data_dir}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")

    """
    Note: using same seed for training/val dataset split to avoid similar images in both datasets.
    """
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        args.data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(224, 224),
        batch_size=args.batch_size
    )

    val_dataset = tf.keras.utils.image_dataset_from_directory(
        args.data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(224, 224),
        batch_size=args.batch_size
    )

    classifier = CVAnimalClassifier()
    classifier.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    classifier.model.fit(train_dataset, epochs=args.epochs, verbose=2, validation_data=val_dataset)

    classifier.model.save(args.save_path)
    print(f"Saved model in {args.save_path}")


if __name__ == '__main__':
    main()