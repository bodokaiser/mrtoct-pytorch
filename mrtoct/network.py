import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


def weight_init(module):
  if isinstance(module, nn.Conv2d):
    init.xavier_uniform(module.weight.data)
    init.xavier_uniform(module.bias.data)


class Encoder(nn.Module):

  def __init__(self, in_channels, out_channels, batch_norm=True):
    super().__init__()

    self.conv = nn.Conv2d(in_channels, out_channels, 4, 2)
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

    self.conv = nn.ConvTranspose2d(in_channels, out_channels, 4, 2)
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
    self.dec4 = Decoder(512, 512)
    self.dec3 = Decoder(512, 256)
    self.dec2 = Decoder(256, 128)
    self.dec1 = Decoder(128, 64)

    self.final = nn.ConvTranspose2d(64, 1, 3)
    self.apply(weight_init)

  def forward(self, x):
    enc1 = self.enc1(x)
    enc2 = self.enc2(enc1)
    enc3 = self.enc2(enc2)
    enc4 = self.enc2(enc3)
    enc5 = self.enc2(enc4)

    dec5 = self.dec5(enc5)
    dec4 = self.dec4(torch.cat([dec5, enc4], 1))
    dec3 = self.dec3(torch.cat([dec4, enc3], 1))
    dec2 = self.dec2(torch.cat([dec3, enc2], 1))
    dec1 = self.dec1(torch.cat([dec2, enc1], 1))

    final = self.final(dec1)

    return F.tanh(final)
