from utils import tools
from utils.model import build_model
from utils.dataloader import load_data
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import logging
import datetime
import numpy as np
import argparse


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description="gene sequence classifier with CNN model"
    )
    parser.add_argument(
        "--output_version", dest="output_version", help="String appended", type=str, default=""
    )
    parser.add_argument(
        "--data_dir",
        dest="data_dir",
        help="Directory path for data.",
        default="./random_split",
        type=str,
    )

    parser.add_argument(
        "--backbone", dest="backbone", help="Model backbone", default="", type=str
    )

    parser.add_argument(
        "--num_epochs",
        dest="num_epochs",
        help="Maximum number of training epochs.",
        default=5,
        type=int,
    )
    parser.add_argument(
        "--batch_size", dest="batch_size", help="Batch size.", default=256, type=int
    )
    
    parser.add_argument(
        "--save_checkpoint",
        dest="save_checkpoint",
        help="Directory path to save checkpoint.",
        default="./output",
        type=str,
    )

    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    return args


def train(data_loader, args):
    logger.info("Starting training...")
    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "logs/fit/" + date
    out_dir = "output/" + date
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    features = data_loader['train']
    target = data_loader['train_target']
    features_dev = data_loader['validation']
    target_dev = data_loader['validation_target']

    # Get input shape and number of classes
    input_shape = (features.shape[1], )
    classes = np.unique(target)
    num_classes = len(classes)
    vocabulary = len(np.unique(features))-1
    # Build model
    model = build_model(args.backbone, input_shape, num_classes, vocabulary)

    # Compile model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    train_writer = tf.summary.create_file_writer(log_dir + '/train')
    val_writer = tf.summary.create_file_writer(log_dir + '/validation')
    test_writer = tf.summary.create_file_writer(log_dir + '/test')

    # Train model
    #neptune_cbk = NeptuneCallback(run=run, base_namespace='metrics')
    batch_size = args.batch_size
    epoch = args.num_epochs
    history = model.fit(features, target, epochs = epoch, batch_size = batch_size,
                    validation_data = (features_dev, target_dev),
                    #class_weight=dict(enumerate((1 / np.bincount(target)) * (len(target) / len(np.bincount(target))))),
                callbacks = [ tensorboard_callback,
                             tf.keras.callbacks.EarlyStopping(patience = 6,
                                                                monitor = 'val_loss',
                                                                mode = 'min',
                                                                restore_best_weights=True)])

    # Endding word for CI/CD
    logger.info("Finished training...")

    #saving the model and plot
    save_model(model, out_dir, args)
    save_figures(history, out_dir, args)
    return model

def save_figures(history, out_dir, args):
    #Loss curve
    plt.figure(figsize = (12, 5))
    plt.plot(history.history['loss'], label = "loss")
    plt.plot(history.history['val_loss'], label = "validation loss")
    plt.legend()
    plt.title("Loss")
    plt.savefig(out_dir+"/loss_"+args.backbone+args.output_version+".png")
    plt.close()

    #Accuracy curve
    plt.figure(figsize = (12, 5))
    plt.plot(history.history['accuracy'], label = "accuracy")
    plt.plot(history.history['val_accuracy'], label = "validation accuracy")
    plt.legend()
    plt.title("Accuracy")
    plt.savefig(out_dir+"/accuracy_"+args.backbone+args.output_version+".png")
    plt.close()

def save_model(model, out_dir, args):
    output_version = args.output_version

    model.save(out_dir+"/args_"+args.backbone+output_version+".h5")
    print("Model saved in "+out_dir+"/args_"+args.backbone+output_version+".h5")
    model.save_weights(out_dir+"/args_"+args.backbone+output_version+"_weights.h5")
    print("Model weights saved in "+out_dir+"/args_"+args.backbone+output_version+"_weights.h5")



if __name__ == "__main__":

    ###  initial arguments
    args = parse_args()

    data_dir = args.data_dir

    # seq_max_len = 120

    # Initialize saving folder
    writer_folder = "output/writer/{}".format(args.output_version)
    snapshot_folder = "output/snapshots/{}".format(args.output_version)
    #writer = SummaryWriter(writer_folder)
    #args.writer = writer
    args.snapshot_folder = snapshot_folder
    args.writer_folder = writer_folder

    if not os.path.exists(
        snapshot_folder
    ):  # add folder if output directory doesn't exist
        os.makedirs(snapshot_folder)
        with open(os.path.join(snapshot_folder, "log.txt"), "w") as fp:
            pass

    if not os.path.exists(writer_folder):
        os.makedirs(writer_folder)

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    log_file = f"{snapshot_folder}/log.txt"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Print paths
    logger.info("Log file is %s" % log_file)
    logger.info("Data path is %s" % data_dir)
    logger.info("Export path is %s" % args.save_checkpoint)

    #pl.seed_everything(seed)  # set seed for reproducity

    ### data loading and preprocessing
    features, target, features_dev, target_dev, features_test, target_test = load_data()
    dataloaders = {'train': features, 'validation': features_dev, 'test': features_test,
                      'train_target': target, 'validation_target': target_dev, 'test_target': target_test}
    logger.info("Model backbone is %s" % args.backbone)

    

    # Train model
    model = train(dataloaders, args)