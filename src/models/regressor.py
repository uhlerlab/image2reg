from torch import nn, Tensor


class FeatureDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim, loss_fct=None):
        super().__init__()
        self.modules = [
            nn.Sequential(
                nn.Linear(latent_dim, hidden_dims[0]),
                nn.BatchNorm1d(hidden_dims[0]),
                nn.PReLU(),
            )
        ]
        for i in range(1, len(hidden_dims)):
            self.modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i - 1], hidden_dims[i]),
                    nn.BatchNorm1d(hidden_dims[i]),
                    nn.PReLU(),
                )
            )
        self.modules.append(nn.Linear(hidden_dims[-1], output_dim))
        self.model = nn.Sequential(*self.modules)
        if loss_fct is None:
            self.loss_fct = nn.MSELoss()
        else:
            self.loss_fct = loss_fct

    def forward(self, latents: Tensor):
        outputs = self.model(latents)
        return outputs

    def loss(self, inputs: Tensor, latents: Tensor):
        preds = self.forward(latents)
        return self.loss_fct(preds, inputs)

    def reset_parameters(self):
        for layer in self.model.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
