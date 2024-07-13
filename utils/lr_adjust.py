def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def lr_warmup(base_lr, iter, warmup_iter):
    return base_lr * (float(iter) / warmup_iter)


def adjust_learning_rate(optimizer, epoch, lr, warmup_epochs, epochs, power=.9, end_lr=0):
    assert lr > end_lr
    if epoch < warmup_epochs:
        lr = lr - lr_warmup(lr, epoch, warmup_epochs)
    else:
        lr = lr_poly(lr - end_lr, epoch, epochs, power) + end_lr
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_custom(optimizer, lr, lr_decay, epoch):
    """Imitating the original implementation"""
    lr = lr / (1.0 + lr_decay * epoch)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_lr(optimizer, args, epoch):
    if args.lr_decay_method == 'poly':
        adjust_learning_rate(optimizer, epoch, args.lr, warmup_epochs=0, power=args.power,
                             epochs=args.epochs)
        # adjust_learning_rate(optimizer_dis, epoch, args.lr_dis, warmup_epochs=0, power=args.power,
        #                      epochs=args.epochs)
        # adjust_learning_rate(optimizer_dis1, epoch, args.lr_dis, warmup_epochs=0, power=args.power,
        #                      epochs=args.epochs)
    elif args.lr_decay_method == 'linear':
        adjust_learning_rate_custom(optimizer, args.lr, args.lr_decay, epoch)
    elif args.lr_decay_method is None:
        pass
    else:
        raise NotImplementedError
