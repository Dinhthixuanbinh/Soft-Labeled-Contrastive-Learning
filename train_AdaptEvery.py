from trainer.Trainer_AdaptEvery import Trainer_AdaptEvery
from datetime import datetime


def main():
    trainer_adaptevery = Trainer_AdaptEvery()
    trainer_adaptevery.train()


if __name__ == '__main__':
    start_time = datetime.now()
    main()
    print('Time elapsed: {}'.format(datetime.now() - start_time))
    print('program finish')
