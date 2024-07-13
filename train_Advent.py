from trainer.Trainer_Advent import Trainer_Advent
from datetime import datetime


def main():
    trainer_advent = Trainer_Advent()
    trainer_advent.train()


if __name__ == '__main__':
    start_time = datetime.now()
    main()
    print('Time elapsed: {}'.format(datetime.now() - start_time))
    print('program finish')
