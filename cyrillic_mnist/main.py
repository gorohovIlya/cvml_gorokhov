from train_model import model, cmd_test, test_loader, device
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

classes = cmd_test.classes
model.eval()

true_vals = 0
all_vals = 0

mistakes_dir = Path("./mistakes")
mistakes_dir.mkdir(exist_ok=True)

for images, labels in test_loader:
    current_batch_size = images.size(0)
    
    for image_idx in range(current_batch_size):
        image = images[image_idx].unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
        
        true_val = classes[labels[image_idx]]
        pred_val = classes[predicted.cpu().item()]
        
        if true_val == pred_val:
            true_vals += 1
        else:
            print(f"True - {true_val}")
            print(f"Pred - {pred_val}")
            img_to_show = images[image_idx].cpu().numpy()
            img_to_show = np.transpose(img_to_show, (1, 2, 0))
                
            plt.figure(figsize=(4, 4))
            plt.imshow(img_to_show)
                
            plt.title(f"True: {true_val}, Pred: {pred_val}")
            plt.axis('off')
                
            filename = mistakes_dir / f"mistake_{true_val}_as_{pred_val}_{all_vals}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
        all_vals += 1

test_accuracy = 100 * true_vals / all_vals
print(f"Test accuracy: {test_accuracy:.3f} ({true_vals}/{all_vals})")