import argparse
import tensorflow as tf
import os
from utils.dataloader import load_data
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description="gene sequence classifier with CNN model"
    )
    parser.add_argument(
        "--output_version", dest="output_version", help="String appended", type=str
    )

    parser.add_argument(
        "--model_dir",
        dest="model_dir",
        help="Directory path for model .h5 file.",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        help="Directory path for saving evaluation",
        type=str,
        default=None
    )
    args = parser.parse_args()

    return args

def plot_f1(test_pred, path_saving, nb_classes=250):
    plt.figure(figsize=(10,10))
    target_names = [f'class {i}' for i in range(nb_classes)]
    report = classification_report(target_test, np.argmax(test_pred, axis=1), output_dict=True, target_names=target_names)
    f1scores = []
    for i in range(250):
        f1scores.append(report[f"class {i}"]["f1-score"])
    plt.title('F1-score per class')
    plt.xlabel('Class')
    plt.ylabel('F1-score')
    plt.plot(f1scores)
    plt.savefig(path_saving)
    #plt.show()

def test(features_test, target_test, args):
    """
    Test the model
    :param dataloader: dataloader
    :param args: arguments
    :return: None
    """
    # load model
    #model = build_model(args.backbone, dataloader.input_shape, dataloader.num_classes, dataloader.vocabulary)
    model_file = list(filter( lambda file: 'weights' not in file, os.listdir(args.model_dir)))[0]
    print(model_file)
    print("Loading model from", os.path.join(args.model_dir, model_file))
    model = tf.keras.models.load_model(os.path.join(args.model_dir, model_file))
    print("Model loaded")

    # test model
    print("Testing model")
    test_pred_model = model.predict(features_test)
    accuracy = accuracy_score(target_test, np.argmax(test_pred_model, axis=1))
    print("test-acc = " + str(accuracy))

    # save test accuracy
    if args.output_dir is None:
        args.output_dir = args.model_dir

    # save accuracy in a csv file
    accuracy_data = [["Test Accuracy", str(accuracy)]]
    with open(os.path.join(args.output_dir, "accuracy.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(accuracy_data)
    
    print(classification_report(target_test, np.argmax(test_pred_model, axis=1)) )
   
    #reploting the report to convert into df
    report = classification_report(target_test, np.argmax(test_pred_model, axis=1), output_dict=True)
    df = pd.DataFrame.from_dict(report).transpose()
    print("Saving test accuracy in %s" % args.output_dir)
    df.to_csv(os.path.join(args.output_dir, "report_evaluation.csv"))

    # save confusion matrix
    #useless, too much class to see it
    print("Saving confusion matrix in %s" % args.output_dir)
    cm = confusion_matrix(target_test, np.argmax(test_pred_model, axis=1))
    df = pd.DataFrame(cm)
    df.to_csv(os.path.join(args.output_dir, "confusion_matrix.csv"))

    # save f1-score per class
    print("Saving f1-score per class in %s" % args.output_dir)
    plot_f1(test_pred_model, os.path.join(args.output_dir, "f1-score_per_class.png"))



if __name__ == "__main__":

    ###  initial arguments
    args = parse_args()

    print("Loading data")
    features, target, features_dev, target_dev, features_test, target_test = load_data()

    print("Testing model (it could take time)")
    test(features_test, target_test, args)