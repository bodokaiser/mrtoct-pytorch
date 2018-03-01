import argparse

from torch.nn import L1Loss
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from mrtoct.network import Generator
from mrtoct.dataset import HDF5, Combined
from mrtoct.transform import Normalize, ToTensor


def main(args):
  model = Generator()
  model.train()

  if args.cuda:
    model = model.cuda()

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

  for epoch in range(1, args.num_epochs + 1):
    for step, (inputs, targets) in enumerate(loader):
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

      if step % 100 == 0:
        print(f'step: {step}, loss: {loss.data[0]}')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--cuda', action='store_true')
  parser.add_argument('--batch-size', default=16)
  parser.add_argument('--learn-rate', default=2e-4)
  parser.add_argument('--beta1-rate', default=5e-1)
  parser.add_argument('--num-epochs', default=300)
  parser.add_argument('--inputs-path')
  parser.add_argument('--targets-path')
  parser.add_argument('--checkpoint-path')

  main(parser.parse_args())
