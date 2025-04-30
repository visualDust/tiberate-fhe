import torch

from tiberate.compiler import tiberate_compiler


def test_tiberate_compiler():
    # Define a simple model
    class SimpleModel(torch.nn.Module):
        def forward(self, x):
            return torch.addmm(x, x, x)

    # Create an instance of the model
    model = SimpleModel()

    compiled_model = torch.compile(
        model,
        backend=tiberate_compiler,
    )
    # Create example input
    example_input = torch.randn(3, 3)
    # Run the compiled model
    output = compiled_model(example_input)
