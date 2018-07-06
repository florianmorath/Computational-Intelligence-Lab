from torch.autograd import Variable as V
from torch.nn.utils import clip_grad_norm
import numpy as np

def fit_model(model, epochs, optimizer, loss_func, callback, trainloader, validloader, clip=0):
    # start call back
    callback.on_train_begin()
    # number of train and valid batches
    nb_trn_batches = len(trainloader) 
    nb_vld_batches = len(validloader)
    for epoch in range(epochs):        
        # Iterate through training set
        model.train(True)
        train_loss = 0.0
        for x, y in trainloader:
            # unpack data
            userId, movieId, ratings = x[:,0],x[:,1],y
            # forward pass
            out = model(V(userId), V(movieId))
            # zero the parameter gradients
            optimizer.zero_grad()
            loss = loss_func(out, V(ratings))
            train_loss += loss.data[0]
            # backward pass
            loss.backward()
            # Gradient clipping
            if clip: 
                params = [p for p in model.parameters()]
                clip_grad_norm(params, clip)
            optimizer.step()
            # update learning rate
            callback.on_batch_end(loss.data[0])
        # Iterate through validation set
        model.train(False)
        valid_loss = 0.0
        for x, y in validloader:
            # unpack data
            userId, movieId, ratings = x[:,0],x[:,1],y
            # forward
            out = model(V(userId), V(movieId))
            loss = loss_func(out, V(ratings))
            # show validation loss
            valid_loss += loss.data[0]
        # Update callback
        callback.on_batch_end(loss.data[0])
        # Turn off training
        model.train(False)
        # Print training and validation loss
        print('Epoch %d: loss = %.6f, val = %.6f' % 
              (epoch+1, train_loss/nb_trn_batches, valid_loss/nb_vld_batches))
        
class ModelOptimizer():
    def __init__(self, opt_fn, model, lr, wd):
        self.model = model
        self.lr = lr
        self.wd = wd
        self.params = [{'params': list(self.model.parameters()), 
                        'lr':self.lr, 'weight_decay':self.wd}]
        self.opt_fn = opt_fn(self.params)
        
    def zero_grad(self):
        self.opt_fn.zero_grad()
        
    def step(self):
        self.opt_fn.step()
        
    def set_lr(self, lr):
        self.lr = lr
        self.opt_fn.param_groups[0]['lr'] = self.lr
        
class CosAnneal():
    def __init__(self, opt, num_batches):
        # number of batches
        self.nb = num_batches 
        # optimizer
        self.opt = opt
        # initial learning rate
        self.init_lr = np.array(opt.lr)
    
    
    def on_train_begin(self):
        """Set up counters needing during training"""
        # counts steps in this cycle
        self.cycle_iter = 0
        # count full cycles
        self.cycle_count = 0
        # Track batches trained on
        self.iteration = 0
        # Record losses and learning rates
        self.lrs = []
        self.losses = []
        # Track epoches trained on
        self.epoch = 0
        # Update learning rate
        self.update_lr()
        
    def on_batch_end(self, loss):
        """Record metric, step counters, 
        and update learning rate"""
        # Update counters 
        self.iteration += 1
        self.lrs.append(self.opt.lr)
        self.losses.append(loss)
        # Update learning rate
        self.update_lr()
    
    # EACH EPOCH
    def on_epoch_end(self, metrics):
        """Step epoch counter"""
        self.epoch += 1
    
    # HELPER FUNCTIONS
    def update_lr(self):
        """Update learning rate"""
        new_lrs = self.calc_lr(self.init_lr)
        # update optimizer with new learning rate
        self.opt.set_lr(new_lrs)

    def calc_lr(self, init_lr):
        """Calculate learning rate"""
        # Do this for the first 5% of batches
        # of batches on the first epoch only
        if self.iteration < self.nb/20:
            self.cycle_iter += 1 
            return init_lr/100
        # Do this for the rest of the batches
        cos_out = np.cos(np.pi*(self.cycle_iter/self.nb)) + 1
        self.cycle_iter += 1
        # Do this on the last batch of each epoch
        if self.cycle_iter == self.nb:
            # Reset steps in cycle counter
            self.cycle_iter = 0
            # Advance cycle count
            self.cycle_count += 1
        return (init_lr / 2) * cos_out