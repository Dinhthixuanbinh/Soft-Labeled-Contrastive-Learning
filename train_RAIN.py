from trainer.Trainer_RAIN import Trainer_RAIN
from datetime import datetime


def main():
    trainer_adain = Trainer_RAIN()
    trainer_adain.train()


if __name__ == '__main__':
    start_time = datetime.now()
    main()
    print('Time elapsed: {}'.format(datetime.now() - start_time))
    print('program finish')
