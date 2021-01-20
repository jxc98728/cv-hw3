
import os
from datetime import datetime


class Settings:

    def __init__(self):
        # directory to save weights file
        self.CHECKPOINT_PATH = 'weights'

        # total training epoches
        self.EPOCH = 200
        self.MILESTONES = [60, 120, 160]

        # initial learning rate
        self.LEARNING_RATE = 0.1

        self.DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
        # time of we run the script
        self.TIME_NOW = datetime.now().strftime(self.DATE_FORMAT)

        # tensorboard log dir
        self.LOG_DIR = 'logs'

        # save weights file per SAVE_EPOCH epoch
        self.SAVE_EPOCH = 10

        self.DATASET_PATH = {
            'cifar10': "./data/cifar10", 
            'mnist': "./data/mnist", 
            'mycifar10': "./data/mycifar10",
            'mymnist': "./data/mymnist",
        }

        self.TRAIN_MEAN = {'cifar10': (0.49139968, 0.48215841, 0.44653091),
                           'mnist': (0.1306604762738429, 0.1306604762738429, 0.1306604762738429),
                           'mycifar10': (0.49272694, 0.48095958, 0.44546653),
                           'mymnist': (0.13581121523609466, 0.13581121523609466, 0.13581121523609466),
                           }
        self.TRAIN_STD = {'cifar10': (0.24703223, 0.24348513, 0.26158784),
                          'mnist': (0.3081078038564622, 0.3081078038564622, 0.3081078038564622),
                          'mycifar10': (0.24877609, 0.24505799, 0.26250601),
                          'mymnist': (0.3134455988907286, 0.3134455988907286, 0.3134455988907286),
                          }
        self.TEST_MEAN = {'cifar10': (0.49421428, 0.48513139, 0.45040909),
                          'mnist': (0.13251460584233699, 0.13251460584233699, 0.13251460584233699),
                          'mycifar10': (0.49582486, 0.48390737, 0.44934585),
                          'mymnist': (0.13777828731492592, 0.13777828731492592, 0.13777828731492592),
                          }
        self.TEST_STD = {'cifar10': (0.24665252, 0.24289226, 0.26159238),
                         'mnist': (0.3104802479305348, 0.3104802479305348, 0.3104802479305348),
                         'mycifar10': (0.2484043, 0.24451455, 0.26246481),
                         'mymnist': (0.3158870347433543, 0.3158870347433543, 0.3158870347433543),
                         }


settings = Settings()
