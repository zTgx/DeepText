import torch
import torch.nn as nn
from activation import GELU

class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]),
            GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]),
            GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]),
            GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]),
            GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]),
            GELU())
        ])

    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output

        return x
        
def print_gradients(model, x):
    output = model(x)

    target = torch.tensor([[0.]])
    
    loss = nn.MSELoss()
    loss = loss(output, target)

    # .backward() method is a convenient method in PyTorch that computes loss gradients,
    loss.backward()

    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")


### Example usage
layer_sizes = [3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([[1., 0., -1.]])
torch.manual_seed(123)
model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)

print_gradients(model_without_shortcut, sample_input)

# layers.0.0.weight has gradient mean of 0.00020173584925942123
# layers.1.0.weight has gradient mean of 0.00012011159560643137
# layers.2.0.weight has gradient mean of 0.0007152040489017963
# layers.3.0.weight has gradient mean of 0.0013988736318424344
# layers.4.0.weight has gradient mean of 0.005049645435065031

torch.manual_seed(123)
model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)

print_gradients(model_without_shortcut, sample_input)

# layers.0.0.weight has gradient mean of 0.22169791162014008
# layers.1.0.weight has gradient mean of 0.20694105327129364
# layers.2.0.weight has gradient mean of 0.32896995544433594
# layers.3.0.weight has gradient mean of 0.2665732204914093
# layers.4.0.weight has gradient mean of 1.3258540630340576

