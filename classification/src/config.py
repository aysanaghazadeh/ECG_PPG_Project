import torch
import os
import argparse
import wandb


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATASET = 'ptb'
MODEL = 'generative'
PATH_TO_DATA = '../data'
DATASET_FOLDER = 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'
PATH_TO_MODELS = './models'
PATH_TO_REPORTS = './reports'
PATH_TO_RESULTS = './results'
TEST_SIZE = 0.2
LEARNING_RATE = 1e-1
MOMENTUM = 9e-1
OPTIMIZER = 'SGD'
LOSS = 'MSE'
NUM_CLASSES = 2
BATCH_SIZE = 8
NUM_EPOCHS = 10000
SAMPLING_RATE = 100
SCHEDULER_GAMMA = 0.8
INPUT_SIZE = 500
OUTPUT_SIZE = 500


def parse_args(arguments=None):
    parser = argparse.ArgumentParser(description="Medical Signal Classification task")
    parser.add_argument(
        "-m", "--model",
        default=MODEL,
        help="model"
    )
    parser.add_argument(
        "-ds", "--dataset_name",
        default=DATASET,
        help="dataset name"
    )
    parser.add_argument(
        "-ptd", "--path_to_data",
        default=PATH_TO_DATA,
        help="path to the data folder"
    )
    parser.add_argument(
        "-dsf", "--dataset_folder",
        default=DATASET_FOLDER,
        help="dataset folder"
    )
    parser.add_argument(
        "-ptds", "--path_to_dataset",
        default=os.path.join(PATH_TO_DATA, DATASET_FOLDER),
        help="path to the dataset folder"
    )
    parser.add_argument(
        "-ptm", "--path_to_models",
        default=PATH_TO_MODELS,
        help="path to models"
    )
    parser.add_argument(
        "-ptrp", "--path_to_reports",
        default=PATH_TO_REPORTS,
        help="path to reports to be saved"
    )
    parser.add_argument(
        "-ptrs", "--path_to_results",
        default=PATH_TO_RESULTS,
        help="path to outputs of the generation model to be saved"
    )
    parser.add_argument(
        "-ts", "--test_size",
        default=TEST_SIZE,
        help="test size for train test split"
    )
    parser.add_argument(
        "-lr", "--learning_rate",
        default=LEARNING_RATE,
        help="enter the learning rate"
    )
    parser.add_argument(
        "-mo", "--momentum",
        default=MOMENTUM,
        help="Enter momentum value"
    )
    parser.add_argument(
        "-o", "--optimizer",
        default=OPTIMIZER,
        help="optimizer"
    )
    parser.add_argument(
        "-l", "--loss",
        default=LOSS,
        help="loss type"
    )
    parser.add_argument(
        "-nc", "--num_classes",
        default=NUM_CLASSES,
        help="number of classes for classification"
    )
    parser.add_argument(
        "-bc", "--batch_size",
        default=BATCH_SIZE,
        help="batch size in the training phase"
    )
    parser.add_argument(
        "-ne", "--num_epochs",
        default=NUM_EPOCHS,
        help="number of epochs in training phase"
    )
    parser.add_argument(
        "-sr", "--sampling_rate",
        default=SAMPLING_RATE,
        help="sampling rate in data reading"
    )
    parser.add_argument(
        "-sg", "--scheduler_gamma",
        default=SCHEDULER_GAMMA,
        help="value of gamma for exponential lr scheduler"
    )
    parser.add_argument(
        "-is", "--input_size",
        default=INPUT_SIZE,
        help="the size of the input to the model"
    )
    parser.add_argument(
        "-os", "--output_size",
        default=OUTPUT_SIZE,
        help="the size of the output to the model"
    )
    args = parser.parse_args(arguments)
    return args


class Config:
    def __init__(self):
        args = parse_args()
        self.device = DEVICE
        self.model = args.model
        self.dataset = args.dataset_name
        self.path_to_data = args.path_to_data
        self.dataset_folder = args.dataset_folder
        self.path_to_dataset = args.path_to_dataset
        self.models_path = args.path_to_models
        os.makedirs(self.models_path, exist_ok=True)
        self.reports_path = args.path_to_reports
        os.makedirs(self.reports_path, exist_ok=True)
        self.results_path = args.path_to_results
        os.makedirs(self.results_path, exist_ok=True)
        self.test_size = args.test_size
        self.lr = args.learning_rate
        self.momentum = args.momentum
        self.optimizer = args.optimizer
        self.loss = args.loss
        self.num_classes = args.num_classes
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.sampling_rate = args.sampling_rate
        self.scheduler_gamma = args.scheduler_gamma
        self.input_size = args.input_size
        self.output_size = args.output_size

        wandb.init(
            # set the wandb project where this run will be logged
            project="ecg_classification",

            #  set the run name
            name=''.join([self.dataset, '_', self.model, '_', str(self.batch_size),
                          '_', str(self.lr) + '_', self.loss, '_', self.optimizer]),

            # track hyperparameters and run metadata
            config={
                "model": self.model,
                "dataset": self.dataset,
                "epochs": self.num_epochs,
                "loss": self.loss,
                "lr": self.lr,
                "optimizer": self.optimizer,
                "batch size": self.batch_size
            }
        )



