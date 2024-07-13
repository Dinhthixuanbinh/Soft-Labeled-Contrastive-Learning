from trainer.Trainer_MPSCL import Trainer_MPSCL
from datetime import datetime


def main():
    trainer_mpscl = Trainer_MPSCL()
    trainer_mpscl.train()


if __name__ == '__main__':
    start_time = datetime.now()
    main()
    print('Time elapsed: {}'.format(datetime.now() - start_time))
    print('program finish')
