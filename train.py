import os
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


def save_model(model, epoch, step, checkpoint_path):
  filename = os.path.join(checkpoint_path, f'model-{epoch}-{step}.pt')

  torch.save(model.state_dict(), filename)
  print(f'epoch: {epoch}, step: {step}, saved checkpoint')


def save_results(tensors, epoch, step, checkpoint_path):
  filename = os.path.join(checkpoint_path, f'results-{epoch}-{step}.jpg')

  save_image(torch.stack([t[0].data for t in tensors]), filename)
  print(f'epoch: {epoch}, step: {step}, saved results')


def restore_checkpoint(model, checkpoint_path):
  chkpts = [f for f in os.listdir(checkpoint_path) if f.endswith('.pt')]
  chkpts.sort()

  if len(chkpts) > 0:
    _, epoch, step = chkpts[0][:-3].split('-')[:3]

    model.load_state_dict(torch.load(os.path.join(
        checkpoint_path, chkpts[0])))
    print(f'epoch: {epoch}, step: {step}, restored checkpoint')
  else:
    epoch = step = 0

  return epoch, step


def main(args):
  model = Generator()
  model.train()

  if args.cuda:
    model = model.cuda()

  if not os.path.exists(args.checkpoint_path):
    os.makedirs(args.checkpoint_path)
  epoch, step = restore_checkpoint(model, args.checkpoint_path)

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

  while epoch < args.num_epochs:
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
        print(f'epoch: {epoch}, step: {step}, loss: {loss.data[0]}')

        save_results([inputs, outputs, targets],
                     epoch, step, args.checkpoint_path)
      if step % args.save_steps == 0:
        save_model(model, epoch, step, args.checkpoint_path)

      step += 1
    epoch += 1


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--cuda', action='store_true')
  parser.add_argument('--batch-size', default=16)
  parser.add_argument('--learn-rate', default=2e-4)
  parser.add_argument('--beta1-rate', default=5e-1)
  parser.add_argument('--num-epochs', default=300)
  parser.add_argument('--log-steps', default=100)
  parser.add_argument('--save-steps', default=2000)
  parser.add_argument('--inputs-path', required=True)
  parser.add_argument('--targets-path', required=True)
  parser.add_argument('--checkpoint-path', required=True)

  main(parser.parse_args())
