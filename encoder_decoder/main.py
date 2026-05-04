from train import (Decoder, Encoder, ImageDataset)
import torch
import matplotlib.pyplot as plt

encoder = Encoder()
decoder = Decoder()

encoder.load_state_dict(torch.load('encoder_4.pth'))
decoder.load_state_dict(torch.load('decoder_4.pth'))
encoder.eval()
decoder.eval()

dataset = ImageDataset(10, 256, 4)
image, _ = dataset[0]
with torch.no_grad():
    latent = encoder(image.unsqueeze(0))
    result = decoder(latent)

    plt.subplot(131)
    plt.imshow(image.squeeze().cpu().numpy())
    plt.subplot(132)
    plt.imshow(result.squeeze().cpu().detach().numpy())
    plt.subplot(133)
    plt.imshow(image.squeeze() - result.squeeze())
    plt.show()


