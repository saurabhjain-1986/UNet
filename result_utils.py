import matplotlib.pyplot as plt


def save_fig(input_image, ground_truth, predicted_mask, output_file):
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(1, 3, 1)
    # showing image
    plt.imshow(input_image)
    plt.axis('off')
    plt.title("Input")
    # Adds a subplot at the 2nd position
    fig.add_subplot(1, 3, 2)
    # showing image
    plt.imshow(ground_truth)
    plt.axis('off')
    plt.title("Ground Truth Mask")
    # Adds a subplot at the 3rd position
    fig.add_subplot(1, 3, 3)
    # showing image
    plt.imshow(predicted_mask)
    plt.axis('off')
    plt.title("Predicted Mask")
    plt.savefig(output_file)
    plt.close()
