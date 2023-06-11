from src.config import Config
from src.model import Model
from src.train import Train
from src.utils import prepare_dataset

if __name__ == '__main__':
    config = Config()
    train_loader, test_loader = prepare_dataset(config)
    model = Model(config)
    train = Train(config, model)
    train.train(train_loader, test_loader)


