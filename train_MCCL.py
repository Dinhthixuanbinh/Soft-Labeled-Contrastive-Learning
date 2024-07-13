from trainer.Trainer_MCCL import Trainer_MCCL
from datetime import datetime


def main():
    trainer_mccl = Trainer_MCCL()
    trainer_mccl.train()


if __name__ == '__main__':
    start_time = datetime.now()
    main()
    print('Time elapsed: {}'.format(datetime.now() - start_time))
    print('program finish')
