import torch
from torchviz import make_dot

def create_pytorch_model_diagram(model, input_size, file_path='model_diagram.png'):
    """
    Generates a diagram of a PyTorch model.

    Args:
    model (torch.nn.Module): The PyTorch model to visualize.
    input_size (tuple): The size of the input tensor (excluding batch size).
    file_path (str): Path to save the generated diagram.

    Returns:
    None; saves the diagram to the specified file path.
    """

    # Create a dummy variable with the input size
    dummy_input = torch.randn(1, *input_size, requires_grad=True)

    # Perform a forward pass to get the output
    output = model(dummy_input)

    # Create a dot graph from the model
    dot = make_dot(output, params=dict(list(model.named_parameters()) + [('input', dummy_input)]))

    # Save the diagram
    dot.render(file_path, format='png', cleanup=True)

    print(f'Model diagram saved to {file_path}')

import models.autoencoder as ae
model = ae.Autoencoder()
create_pytorch_model_diagram(model, (476,), 'autoencoder_diagram')

import models.vae as vae
model = vae.VAE(476, 32)
create_pytorch_model_diagram(model, (476,), 'vae_diagram')

import models.cvae as cvae
model = cvae.CVAE(476, 128, 32)
create_pytorch_model_diagram(model, (476,), 'cvae_diagram')