import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


def weight_init(module):
  if any(isinstance(module, cl) for cl in [nn.Conv2d, nn.ConvTranspose2d]):
    init.xavier_uniform(module.weight.data)
    module.weight.data.normal_(.0, .01)

  if isinstance(module, nn.BatchNorm2d):
    module.weight.data.normal_(1.0, 0.02)
    module.bias.data.fill_(0)


class Encoder(nn.Module):

  def __init__(self, in_channels, out_channels, batch_norm=True):
    super().__init__()

    self.conv = nn.Conv2d(in_channels, out_channels, 4, 2, 1)
    self.norm = nn.BatchNorm2d(out_channels) if batch_norm else None

  def forward(self, x):
    x = self.conv(x)

    if self.norm is not None:
      x = self.norm(x)

    return F.leaky_relu(x)


class Decoder(nn.Module):

  def __init__(self, in_channels, out_channels, batch_norm=True,
               dropout=False):
    super().__init__()

    self.conv = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1)
    self.norm = nn.BatchNorm2d(out_channels) if batch_norm else None
    self.drop = nn.Dropout2d() if dropout else None

  def forward(self, x):
    x = self.conv(x)

    if self.norm is not None:
      x = self.norm(x)
    if self.drop is not None:
      x = self.drop(x)

    return F.relu(x)


class Generator(nn.Module):

  def __init__(self):
    super().__init__()

    self.enc1 = Encoder(1, 64, batch_norm=False)
    self.enc2 = Encoder(64, 128)
    self.enc3 = Encoder(128, 256)
    self.enc4 = Encoder(256, 512)
    self.enc5 = Encoder(512, 512)

    self.dec5 = Decoder(512, 512, dropout=True)
    self.dec4 = Decoder(1024, 256, dropout=True)
    self.dec3 = Decoder(512, 128)
    self.dec2 = Decoder(256, 64, dropout=True)
    self.dec1 = Decoder(128, 1)

    self.apply(weight_init)

  def forward(self, x):
    enc1 = self.enc1(x)
    enc2 = self.enc2(enc1)
    enc3 = self.enc3(enc2)
    enc4 = self.enc4(enc3)
    enc5 = self.enc5(enc4)

    dec5 = self.dec5(enc5)
    dec4 = self.dec4(torch.cat([dec5, enc4], 1))
    dec3 = self.dec3(torch.cat([dec4, enc3], 1))
    dec2 = self.dec2(torch.cat([dec3, enc2], 1))
    dec1 = self.dec1(torch.cat([dec2, enc1], 1))

    return F.tanh(dec1)
