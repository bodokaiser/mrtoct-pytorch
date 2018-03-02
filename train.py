import os
import torch
import argparse

from torch.nn import L1Loss
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from mrtoct.network import Generator
from mrtoct.dataset import Patch
from mrtoct.transform import ToTensor


def log_msg(step, message):
  print(f'step: {step}, {message}')


def fmt_fname(prefix, ext, step, chckpt_path):
  return os.path.join(chckpt_path, f'{prefix}-{step:07d}{ext}')


def save_model(model, step, checkpoint_path):
  torch.save(model.state_dict(),
             fmt_fname('model', '.pt', step, checkpoint_path))
  log_msg(step, 'saved checkpoint')


def save_results(tensors, step, checkpoint_path):
  save_image(torch.stack([t[0].data for t in tensors]),
             fmt_fname('images', '.jpg', step, checkpoint_path))


def restore_checkpoint(model, checkpoint_path):
  chkpts = [f for f in os.listdir(checkpoint_path) if f.endswith('.pt')]
  chkpts.sort()

  chkpt = chkpts[-1]

  if len(chkpts) > 0:
    step = int(chkpt[:-3].split('-')[1])

    model.load_state_dict(torch.load(os.path.join(
        checkpoint_path, chkpt)))
    log_msg(step, f'restored checkpoint {chkpt}')
  else:
    step = 0

  return step


def main(args):
  model = Generator()
  model.train()

  if args.cuda:
    model = model.cuda()

  if not os.path.exists(args.checkpoint_path):
    os.makedirs(args.checkpoint_path)
  step = restore_checkpoint(model, args.checkpoint_path)

  dataset = Patch(args.data_path,
                  transform=ToTensor(),
                  target_transform=ToTensor())
  loader = DataLoader(dataset, args.batch_size, shuffle=True)

  criterion = L1Loss()
  optimizer = Adam(model.parameters(), args.learn_rate)

  while True:
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
        log_msg(step, f'loss: {loss.data[0]}')

        save_results([inputs, outputs, targets], step, args.checkpoint_path)
      if step % args.save_steps == 0 and step > 0:
        save_model(model, step, args.checkpoint_path)

      step += 1


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--cuda', action='store_true')
  parser.add_argument('--batch-size', default=16)
  parser.add_argument('--learn-rate', default=2e-4)
  parser.add_argument('--beta1-rate', default=5e-1)
  parser.add_argument('--log-steps', default=100)
  parser.add_argument('--save-steps', default=1000)
  parser.add_argument('--data-path', required=True)
  parser.add_argument('--checkpoint-path', required=True)

  main(parser.parse_args())
