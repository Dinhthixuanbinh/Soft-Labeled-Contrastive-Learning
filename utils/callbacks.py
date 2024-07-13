import numpy as np
from torch import save
import os


class ModelCheckPointCallback:

    def __init__(self, mode="min", best_model_dir=None, save_best=False, entire_model=False,
                 save_last_model=False, model_name="../weights/model_checkpoint.pt", n_epochs=200,
                 save_every_epochs=None):
        """
        Save model checkpoints based on the parameters given
        Args:
            mode: 'min' or 'max' which decide the definition of a "better" / "best" score
            best_model_dir: directory to save the best model
            save_best: whether to save the best model
            entire_model: whether to save the entire model or just the state_dict
            save_last_model: whether to save the model of the last epoch
            model_name: the directory to save the model of the last epoch
            n_epochs: total epochs to run
        """
        self.start = True
        assert mode == "max" or mode == "min", "mode can only be /'min/' or /'max/'"
        self.mode = mode
        self.best_result = np.Inf if mode == 'min' else np.NINF
        self.model_name = model_name
        _, ext = os.path.splitext(model_name)
        if ext == '':
            self.model_name += '.pt'
        if best_model_dir is None:
            best_model_dir = model_name
        self.best_model_name_base, self.ext = os.path.splitext(best_model_dir)
        self.best_model_dir = best_model_dir
        self.best_model_save_dir = None
        if self.ext == '':
            self.ext = '.pt'
            self.best_model_dir += '.pt'
        self.entire_model = entire_model
        self.save_last_model = save_last_model
        self.n_epochs = n_epochs
        self.epoch = 0
        self._save_best = save_best
        self.save_every_epochs = save_every_epochs

    def step(self, monitor, model, epoch, optimizer=None, tobreak=False):
        """
        Check the monitor score with the previously saved best score and save the currently best model
        Args:
            monitor: the score used for monitoring
            model: the model to be saved
            epoch: the current epoch number
            optimizer: the optimizer of the model (only used to recover the training state when load the model)
            tobreak: whether reach the last epoch

        Returns:
        """
        model_name = None
        if self.entire_model:
            to_save = model
            opt_to_save = optimizer
        else:
            to_save = model.state_dict()
            opt_to_save = optimizer.state_dict() if optimizer is not None else None

        if (self.save_every_epochs is not None) and (epoch % self.save_every_epochs == 0):
            model_name = '{}.e{}.Scr{}{}'.format(self.best_model_name_base, epoch, np.around(self.best_result, 3), self.ext)
            save({'epoch': epoch,
                  'model_state_dict': to_save,
                  'optimizer_state_dict': opt_to_save}, model_name)
        if self._save_best:
            # check whether the current loss is lower than the previous best value.
            if self.mode == "max":
                better = monitor > self.best_result
            else:
                better = monitor < self.best_result
            if better or self.start:
                self.best_result = monitor
                self.epoch = epoch
                save({'epoch': epoch,
                      'model_state_dict': to_save,
                      'optimizer_state_dict': opt_to_save}, self.best_model_dir)
                if self.start:
                    print('****** best model saved for the first epoch ******')
                    self.start = False
            if (epoch == self.n_epochs) or tobreak:
                self.best_model_save_dir = '{}.e{}.Scr{}{}'.format(self.best_model_name_base, self.epoch, np.around(self.best_result, 3), self.ext)
                try:
                    os.rename(self.best_model_dir, self.best_model_save_dir)
                    print(f'best model renamed to: {self.best_model_save_dir}')
                except Exception as e:
                    print(e)
                    print(f'*******{self.best_model_save_dir} exists: {os.path.exists(self.best_model_save_dir)}*******')
        if self.save_last_model and ((epoch == self.n_epochs) or tobreak):
            save({'epoch': epoch,
                  'model_state_dict': to_save,
                  'optimizer_state_dict': opt_to_save}, self.model_name)
        return model_name


class EarlyStopCallback:
    def __init__(self, stop_criterion_len=200, mode='max'):
        self._stop_criterion_len = stop_criterion_len
        self._best_score = np.Inf if mode == 'min' else np.NINF
        self._mode = mode
        self._count = 0

    def step(self, monitor):
        if self._mode == 'max':
            if self._best_score >= monitor:
                self._count += 1
            else:
                self._best_score = monitor
                self._count = 0
        else:
            if self._best_score >= monitor:
                self._best_score = monitor
                self._count = 0
            else:
                self._count += 1
        if self._count == self._stop_criterion_len:
            self._count = 0
            return True
        else:
            return False


def get_model_dir(weight_dir: str, apdx: str, ext: str = '.pt', ifbest=False):
    dir = os.path.join(weight_dir, 'best_' + apdx + ext) if ifbest else os.path.join(weight_dir, apdx + ext)
    return dir


def get_model_checkpoint(n_epochs: int, save_best: bool, apdx: str, weight_dir: str, mode: str = 'max',
                         entire_model: bool = False, save_last_model: bool = True):
    if not os.path.exists(weight_dir):
        os.mkdir(weight_dir)
    model_dir = get_model_dir(weight_dir, apdx, '.pt')
    best_weight_dir = get_model_dir(weight_dir, apdx, '.pt', ifbest=True)
    """create the model check point"""
    modelcheckpoint_unet = ModelCheckPointCallback(n_epochs=n_epochs, save_best=save_best,
                                                   mode=mode,
                                                   best_model_dir=best_weight_dir,
                                                   save_last_model=save_last_model,
                                                   model_name=model_dir,
                                                   entire_model=entire_model)
    return modelcheckpoint_unet


if __name__ == '__main__':
    import os
    a = '../../train_pointnet.py'
    b, c = os.path.splitext(a)
    print('{}{}{}{}'.format(b, '.Scr', np.around(0.862323232, 2), c))
