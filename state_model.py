import torch
import torch.nn as nn


class CropStateModel(nn.Module):

    def __init__(
        self,
        state_dim=4,
        forcing_dim=3,
        hidden_dim=32
    ):
        super().__init__()

        self.dynamics = nn.Sequential(
            nn.Linear(
                state_dim + forcing_dim,
                hidden_dim
            ),
            nn.Tanh(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),
            nn.Tanh(),

            nn.Linear(
                hidden_dim,
                state_dim
            )
        )

    def forward(self, state, forcing):

        x = torch.cat(
            [state, forcing],
            dim=-1
        )

        delta = self.dynamics(x)

        # Small residual update.
        # This prevents the state from jumping
        # unrealistically between consecutive days.
        delta = 0.05 * torch.tanh(delta)

        next_state = state + delta

        # Keep the latent state bounded.
        next_state = 3.0 * torch.tanh(
            next_state / 3.0
        )

        return next_state


if __name__ == "__main__":

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Using device:", device)

    model = CropStateModel().to(device)

    state = torch.tensor(
        [[0.5, 0.5, 0.5, 0.0]],
        dtype=torch.float32,
        device=device
    )

    forcing = torch.tensor(
        [[0.0, 0.0, 0.0]],
        dtype=torch.float32,
        device=device
    )

    next_state = model(
        state,
        forcing
    )

    print("Input state:", state)
    print("Next state:", next_state)

    print(
        "Trainable parameters:",
        sum(
            p.numel()
            for p in model.parameters()
        )
    )
