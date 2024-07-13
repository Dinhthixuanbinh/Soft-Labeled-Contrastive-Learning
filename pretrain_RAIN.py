from trainer.Pretrainer_RAIN import Pretrainer_RAIN
from datetime import datetime


def main():
    trainer_adain = Pretrainer_RAIN()
    if trainer_adain.args.task == 'pretrain_RAIN':
        trainer_adain.train()
    elif trainer_adain.args.task == 'self_recon':
        trainer_adain.train_selfrecon()


if __name__ == '__main__':
    start_time = datetime.now()
    main()
    print('Time elapsed: {}'.format(datetime.now() - start_time))
    print('program finish')
