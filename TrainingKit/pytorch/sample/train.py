import os
import time
import argparse
from datetime import datetime
import numpy as np

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
from torchvision import datasets, transforms

from network import *
import numpy

def output_name_and_params(net):
    #for name, parameters in net.named_parameters():
    #    print(name, parameters.shape, parameters.requires_grad )
    for parameters in net.state_dict():
        print(parameters, net.state_dict()[parameters].shape)

# Training settings
parser = argparse.ArgumentParser(description='Quantized-Net pytorch implementation')

parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--data_dir', type=str, default='')
parser.add_argument('--log_name', type=str, default='')
parser.add_argument('--pretrain', action='store_true', default=False)
parser.add_argument('--pretrain_dir', type=str, default='')

parser.add_argument('--Wbits', type=int, default=8)
parser.add_argument('--Abits', type=int, default=8)

parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--wd', type=float, default=1e-4)

parser.add_argument('--train_batch_size', type=int, default=128)
parser.add_argument('--eval_batch_size', type=int, default=100)
parser.add_argument('--max_epochs', type=int, default=5)
parser.add_argument('--save_epoch', type=int, default=1)

parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--device_ids', type=str, default=[0,1,2,3])
parser.add_argument('--num_workers', type=int, default=5)

parser.add_argument('--cluster', action='store_true', default=False)
parser.add_argument('--summary', action='store_true', default=False)
parser.add_argument('--cuda', action='store_true', default=True)
parser.add_argument('--is_quantize', action='store_true', default=True)

cfg = parser.parse_args()
device_ids = cfg.device_ids

qtype = 'w%sa%s'%(cfg.Wbits, cfg.Abits)
cfg.log_name = 'net-'+ qtype
print(cfg)
cfg.log_dir = os.path.join(cfg.root_dir, 'logs/net/', cfg.log_name)
cfg.ckpt_dir = os.path.join(cfg.root_dir, 'ckpt', cfg.log_name)

if not os.path.exists(cfg.log_dir):
  os.makedirs(cfg.log_dir)
if not os.path.exists(cfg.ckpt_dir):
  os.makedirs(cfg.ckpt_dir)

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def main():
  print('training MNIST !')
  dataset = datasets.MNIST

  print('==> Preparing data ..')

  train_dataset = datasets.MNIST('./MNIST_data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                   ]))
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.train_batch_size*len(device_ids), shuffle=False, num_workers=cfg.num_workers)

  eval_dataset = datasets.MNIST('./MNIST_data', train=True, transform=transforms.Compose([
                       transforms.ToTensor(),
                   ]))
  eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=cfg.eval_batch_size*len(device_ids), shuffle=False, num_workers=cfg.num_workers)

  print("==> Building ResNet...is_quantize = ", cfg.is_quantize, "abits = ", cfg.Wbits, "abits = ", cfg.Abits)
  if cfg.cuda:
      model = Network_T40(is_quantize = cfg.is_quantize, BITA = cfg.Abits, BITW = cfg.Wbits).cuda()
  else:
      model = Network_T40(is_quantize = cfg.is_quantize, BITA = cfg.Abits, BITW = cfg.Wbits)

  # output_name_and_params(model)

  optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
  lr_schedu = optim.lr_scheduler.MultiStepLR(optimizer, [80, 120, 150], gamma=0.1)
  if cfg.cuda:
      criterion = torch.nn.CrossEntropyLoss().cuda()
  else:
      criterion = torch.nn.CrossEntropyLoss()
  if cfg.summary:
    summary_writer = SummaryWriter(cfg.log_dir)

  if cfg.pretrain:
    from collections import OrderedDict
    state_dict = torch.load(cfg.pretrain_dir)
    model.load_state_dict(state_dict)

  # Training
  def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()

    start_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
      shape = inputs.shape
      temp = np.zeros((shape[0], shape[1], shape[2] + 4, shape[3] + 4), np.float32)
      temp[:,:,2:30,2:30] = inputs
      temp = np.repeat(temp, 3, 1)
      inputs = torch.from_numpy(temp)
      if cfg.cuda:
          outputs = model(inputs.cuda())
          loss = criterion(outputs.cuda(), targets.cuda())
      else:
          outputs = model(inputs)
          loss = criterion(outputs, targets)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if batch_idx % cfg.log_interval == 0:
        step = len(train_loader) * epoch + batch_idx
        duration = time.time() - start_time

        print('lr: %.5f epoch: %d step: %d cls_loss= %.5f (%d samples/sec)' %
              (optimizer.state_dict()['param_groups'][0]['lr'], epoch, batch_idx, loss.item(),
               cfg.train_batch_size * cfg.log_interval / duration))

        start_time = time.time()
        if cfg.summary:
          summary_writer.add_scalar('cls_loss', loss.item(), step)
          summary_writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], step)

  def test(epoch):
    model.eval()
    correct = 0
    for batch_idx, (inputs, targets) in enumerate(eval_loader):
      shape = inputs.shape
      temp = np.zeros((shape[0], shape[1], shape[2] + 4, shape[3] + 4), np.float32)
      temp[:,:,2:30,2:30] = inputs
      temp = np.repeat(temp, 3, 1)
      inputs = torch.from_numpy(temp)
      if cfg.cuda:
          inputs, targets = inputs.cuda(), targets.cuda()

      outputs = model(inputs)
      _, predicted = torch.max(outputs.data, 1)
      correct += predicted.eq(targets.data).cpu().sum().item()

    acc = 100. * correct / len(eval_dataset)
    print('----------------------------------------------------------------- '
          'Precision@1: %.2f%% \n' % (acc))
    if cfg.summary:
      summary_writer.add_scalar('Precision@1', acc, global_step=epoch)

  for epoch in range(cfg.max_epochs):
    train(epoch)
    lr_schedu.step(epoch)
    test(epoch)
    if epoch % cfg.save_epoch == 0:
      torch.save(model.state_dict(), os.path.join(cfg.ckpt_dir, 'checkpoint.%s-%d'%(qtype, epoch)))

  if cfg.summary:
    summary_writer.close()


if __name__ == '__main__':
  main()
