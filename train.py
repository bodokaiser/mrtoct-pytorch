import os
import time
import torch
import argparse

from torch.nn import L1Loss
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from torchvision.utils import save_image

from mrtoct.network import Generator
from mrtoct.dataset import HDF5, Combined
from mrtoct.transform import Normalize, ToTensor


def timestamp():
  return time.strftime('%d%m%Y%H%M%S')


def save_model(model, checkpoint_path):
  filename = os.path.join(checkpoint_path, f'{timestamp()}.pt')

  torch.save(model.state_dict(), filename)


def save_results(tensors, checkpoint_path):
  filename = os.path.join(checkpoint_path, f'{timestamp()}.jpg')

  save_image(torch.stack([t[0].data for t in tensors]), filename)


def main(args):
  model = Generator()
  model.train()

  if args.cuda:
    model = model.cuda()
  if args.checkpoint:
    model.load_state_dict(torch.load(os.path.join(
        args.checkpoint_path, args.checkpoint)))

  if not os.path.exists(args.checkpoint_path):
    os.makedirs(args.checkpoint_path)

  inputs_dataset = HDF5(args.inputs_path, 'slices')
  inputs_transform = Compose([
      Normalize(inputs_dataset.meta['vmax'],
                inputs_dataset.meta['vmin']),
      ToTensor(),
  ])

  targets_dataset = HDF5(args.targets_path, 'slices')
  targets_transform = Compose([
      Normalize(targets_dataset.meta['vmax'],
                targets_dataset.meta['vmin']),
      ToTensor(),
  ])

  dataset = Combined(inputs_dataset, targets_dataset,
                     inputs_transform, targets_transform)
  loader = DataLoader(dataset, args.batch_size, shuffle=True)

  criterion = L1Loss()
  optimizer = Adam(model.parameters(), args.learn_rate)

  step = 0

  for epoch in range(1, args.num_epochs + 1):
    for inputs, targets in loader:
      if args.cuda:
        inputs = inputs.cuda()
        targets = targets.cuda()

      inputs = Variable(inputs, requires_grad=True)
      targets = Variable(targets)
      outputs = model(inputs)

      optimizer.zero_grad()
      loss = criterion(outputs, targets)
      loss.backward()
      optimizer.step()

      if step % args.log_steps == 0:
        print(f'step: {step}, loss: {loss.data[0]}')

        save_results([inputs, outputs, targets], args.checkpoint_path)
      if step % args.save_steps == 0:
        save_model(model, args.checkpoint_path)

      step += 1


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--cuda', action='store_true')
  parser.add_argument('--batch-size', default=16)
  parser.add_argument('--learn-rate', default=2e-4)
  parser.add_argument('--beta1-rate', default=5e-1)
  parser.add_argument('--num-epochs', default=300)
  parser.add_argument('--log-steps', default=100)
  parser.add_argument('--save-steps', default=2000)
  parser.add_argument('--checkpoint')
  parser.add_argument('--inputs-path', required=True)
  parser.add_argument('--targets-path', required=True)
  parser.add_argument('--checkpoint-path', required=True)

  main(parser.parse_args())
