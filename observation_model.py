import torch
import torch.nn as nn


class S2ObservationModel(nn.Module):
    """
    Maps latent crop state to Sentinel-2 observations.

    State:
        [LAI, Biomass, Water, Phenology]

    Output:
        [NDVI, EVI, FPAR, NDWI, SVI]
    """

    def __init__(self, state_dim=4, output_dim=5):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, 16),
            nn.Tanh(),
            nn.Linear(16, output_dim)
        )

    def forward(self, state):
        return self.network(state)


class S1ObservationModel(nn.Module):
    """
    Maps latent crop state to Sentinel-1 observations.

    Output:
        [VV, VH, VV/VH]
    """

    def __init__(self, state_dim=4, output_dim=3):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, 16),
            nn.Tanh(),
            nn.Linear(16, output_dim)
        )

    def forward(self, state):
        return self.network(state)


if __name__ == "__main__":

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Using device:", device)

    state = torch.tensor(
        [[0.5, 0.5, 0.5, 0.0]],
        dtype=torch.float32,
        device=device
    )

    s2_model = S2ObservationModel().to(device)
    s1_model = S1ObservationModel().to(device)

    s2_prediction = s2_model(state)
    s1_prediction = s1_model(state)

    print("S2 prediction:", s2_prediction)
    print("S1 prediction:", s1_prediction)
