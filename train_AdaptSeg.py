from trainer.Trainer_AdaptSeg import Trainer_AdapSeg
from datetime import datetime


def main():
    trainer_adaptseg = Trainer_AdapSeg()
    trainer_adaptseg.train()


if __name__ == '__main__':
    start_time = datetime.now()
    main()
    print('Time elapsed: {}'.format(datetime.now() - start_time))
    print('program finish')
