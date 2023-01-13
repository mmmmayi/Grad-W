import math

class BaseClass:
    '''
    Base Class for learning rate scheduler
    '''

    def __init__(self,
                 optimizer,
                 num_epochs,
                 epoch_iter,
                 initial_lr,
                 final_lr,
                 warm_up_epoch=6,
                 scale_ratio=1.0,
                 warm_from_zero=False):
        '''
        warm_up_epoch: the first warm_up_epoch is the multiprocess warm-up stage
        scale_ratio: multiplied to the current lr in the multiprocess training
        process
        '''
        self.optimizer = optimizer
        self.max_iter = num_epochs * epoch_iter
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.scale_ratio = scale_ratio
        self.current_iter = 0
        self.warm_up_iter = warm_up_epoch * epoch_iter
        self.warm_from_zero = warm_from_zero

    def get_multi_process_coeff(self):
        lr_coeff = 1.0 * self.scale_ratio
        if self.current_iter < self.warm_up_iter:
            if self.warm_from_zero:
                lr_coeff = self.scale_ratio * self.current_iter / self.warm_up_iter
            elif self.scale_ratio > 1:
                lr_coeff = (self.scale_ratio -
                            1) * self.current_iter / self.warm_up_iter + 1.0

        return lr_coeff


    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def set_lr(self):
        current_lr = self.get_current_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr

    def step(self, current_iter=None):
        if current_iter is not None:
            self.current_iter = current_iter

        self.set_lr()
        self.current_iter += 1
    def step_return_lr(self, current_iter=None):
        if current_iter is not None:
            self.current_iter = current_iter

        current_lr = self.get_current_lr()
        self.current_iter += 1

        return current_lr

    def get_current_lr(self):
        '''
        This function should be implemented in the child class
        '''
        return 0.0


class ExponentialDecrease(BaseClass):

    def __init__(self,
                 optimizer,
                 num_epochs,
                 epoch_iter,
                 initial_lr,
                 final_lr,
                 warm_up_epoch=6,
                 scale_ratio=1.0,
                 warm_from_zero=False):
        super().__init__(optimizer, num_epochs, epoch_iter, initial_lr,
                         final_lr, warm_up_epoch, scale_ratio, warm_from_zero)

    def get_current_lr(self):
        lr_coeff = self.get_multi_process_coeff()
        current_lr = lr_coeff * self.initial_lr * math.exp(
            (self.current_iter / self.max_iter) *
            math.log(self.final_lr / self.initial_lr))
        print('current_lr',current_lr)
        return current_lr

def show_lr_curve(scheduler):
    import matplotlib.pyplot as plt

    lr_list = []
    for current_lr in range(0, scheduler.max_iter):
        lr_list.append(scheduler.step_return_lr(current_lr))
    data_index = list(range(1, len(lr_list) + 1))

    plt.plot(data_index, lr_list, '-o', markersize=1)
    plt.legend(loc='best')
    plt.xlabel("Iteration")
    plt.ylabel("LR")
    plt.savefig('scheduler.png')

    plt.show()

if __name__ == '__main__':
    optimizer = None
    num_epochs = 100
    epoch_iter = 1706
    initial_lr = 0.6
    final_lr = 0.1
    warm_up_epoch = 5
    scale_ratio = 1
    scheduler = ExponentialDecrease(optimizer, num_epochs, epoch_iter,
                                    initial_lr, final_lr, warm_up_epoch,
                                    scale_ratio)

    show_lr_curve(scheduler)
 
