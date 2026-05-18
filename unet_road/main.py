from unet_road import ds, images_loader, model, device, model_path
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import torch

model.eval()

image, mask = ds[0]

input_tensor = image.unsqueeze(0).to(device)

with torch.no_grad():
    prediction_logits = model(input_tensor)
    prediction = (torch.sigmoid(prediction_logits) > 0.5).float()

pred_numpy = prediction.squeeze().cpu().numpy()
orig_image_numpy = image.permute(1, 2, 0).numpy()
true_mask_numpy = mask.squeeze().numpy()
difference = true_mask_numpy - pred_numpy

fig, ax = plt.subplots(1, 3)
ax[0].set_title("Original")
ax[0].imshow(orig_image_numpy)
ax[1].set_title("Prediction")
ax[1].imshow(pred_numpy)
ax[2].set_title('Difference')
ax[2].imshow(difference)
plt.show()
