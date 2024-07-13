from trainer.Trainer_baseline import Trainer_baseline
from datetime import datetime


def main():
    trainer_base = Trainer_baseline()
    trainer_base.train()


if __name__ == '__main__':
    start_time = datetime.now()
    main()
    print('Time elapsed: {}'.format(datetime.now() - start_time))
    print('program finish')
