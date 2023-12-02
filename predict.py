from modules.CASENet import CASENet_resnet101
from utils import utils

import torch
from torchvision import transforms
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt


def main():
    image_path = "predict/input/blood_vesels.png"
    img = Image.open(image_path)

    transforms_list = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(336, 336))
    ])

    img_tensor = transforms_list(img.convert(mode="RGB"))
    img_tensor = torch.unsqueeze(img_tensor, dim=0)

    weights_path = "model_casenet.pth.tar"
    model = CASENet_resnet101(pretrained=False, num_classes=19)
    utils.load_pretrained_model(model=model, pretrained_model_path=weights_path)

    # Output: score_feats1, cropped_score_feats2, cropped_score_feats3, cropped_score_feats5, fused_feats
    output_tensors = model.forward(x=img_tensor, for_vis=True)

    # Step 1: Convert Tensor to NumPy Array
    output_arrays = [tensor.cpu().detach().numpy() for tensor in output_tensors]

    # Step 2: Post-process the Output Tensors (Example: Normalize)
    normalized_output = [output_array / np.max(output_array) for output_array in output_arrays]

    # Step 3: Overlay the Result on the Input Image
    fig, axes = plt.subplots(1, len(output_tensors) + 1, figsize=(15, 5))

    # Display the original image
    axes[0].imshow(np.array(img))
    axes[0].set_title('Input Image')

    # Create a binary mask for each output tensor
    binary_masks = [normalized_output[i][0, 0, :, :] > 0.5 for i in range(len(output_tensors))]
    merged_edges = np.zeros_like(binary_masks[0], dtype=np.uint8)

    # Display the processed output tensors
    for i in range(len(output_tensors)):
        axes[i + 1].imshow(normalized_output[i][0, 0, :, :])  # Assuming a single-channel output
        axes[i + 1].set_title(f'Output {i + 1}')

        # Save the output images
        output_image_path = f"predict/output/output_{i + 1}.png"
        plt.imsave(output_image_path, normalized_output[i][0, 0, :, :], cmap='gray')

        # Save the output images without background (binary mask)
        binary_mask = normalized_output[i][0, 0, :, :] > 0.5  # Adjust the threshold as needed
        output_no_background_path = f"predict/output/output_{i + 1}_no_background.png"
        plt.imsave(output_no_background_path, binary_mask, cmap='gray')

        # Merge the binary mask into the merged edges
        merged_edges[binary_masks[i]] = 255  # Set pixel to white (255) where there is an edge

    # Save the merged edges image
    merged_edges_image = Image.fromarray(merged_edges, 'L')  # 'L' mode for grayscale
    merged_edges_image.save("predict/output/merged_edges.png")

    # Show the merged edges
    axes[len(output_tensors)].imshow(merged_edges, cmap='gray')
    axes[len(output_tensors)].set_title('Merged Edges')

    plt.show()


if __name__ == '__main__':
    main()
