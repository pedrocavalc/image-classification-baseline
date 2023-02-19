import torch
import torch.nn as nn

@torch.no_grad()
def evaluate(model,val_loader):
  model.eval()
  outputs = [model.validation_step(batch) for batch in val_loader]
  return model.validation_epoch_end(outputs)

def get_lr(optimizer):
  for param_group in optimizer.param_groups:
    return param_group['lr']

def fit_one_cycle(epochs,max_lr,model,train_loader,val_loader,weight_decay =0, grad_clip = None, opt_func = torch.optim.SGD):
  """
  Function to make one cicle of training. The function utilizes learning rate schedule to spped up the training

  Args:
      epochs (_int_): Number of epochs for the train
      max_lr (_float_): Learning rate max that will be usedin training
      model (_Pytorch.Model_): A pytorch model architeture that will be training
      train_loader (_DataLoader_): Train DataLoader  that contains the image for train
      val_loader (_DataLoader_): Validation DataLoader  that contains the image for validation
      weight_decay (int, optional): Regularization technique that applyes a small penalty using the l2 penalty. Defaults to 0.
      grad_clip (_type_, optional): Regularization technique that limit the gradient values to avoid gradient explosion and 
      gradient vanishing. Defaults to None.
      opt_func (_Pytorch.optimize_, optional): Optimizer that will be used to update the weights of the model. Defaults to torch.optim.SGD.

  Returns:
      _list_: history with the metrics for each epoch
  """
  torch.cuda.empty_cache()
  history = []
  # set upt custom optimizer with weigth decay
  optimizer = opt_func(model.parameters(),max_lr,weight_decay=weight_decay)
  sched = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr,epochs=epochs,steps_per_epoch=len(train_loader))
  print(f'--------------------------------Start training ----------------------------')
  for epoch in range(epochs):
    model.train()
    train_losses = []
    lrs = []
    count = 0
    for batch in train_loader:
      count += 1
      if count % 10 == 0:
        print(f'batch {count}/{len(train_loader)}')
      loss = model.training_step(batch)
      train_losses.append(loss)
      loss.backward()
      # grad clip
      if grad_clip:
        nn.utils.clip_grad_value_(model.parameters(),grad_clip)
      
      optimizer.step()
      optimizer.zero_grad()

      lrs.append(get_lr(optimizer))
      sched.step()
  
    result = evaluate(model,val_loader)
    result['train_loss'] = torch.stack(train_losses).mean().item()
    result['lrs'] = lrs
    print(f'epoch {epoch} results{result}')
    model.epoch_end(epoch,result)
    history.append(result)
  return history