import numpy as np


class EnsembleKalmanFilter:
    """
    Ensemble Kalman Filter for the crop digital twin.

    State:
        [LAI, Biomass, Water, Phenology]

    The filter supports asynchronous observations:
        - Sentinel-1 only
        - Sentinel-2 only
        - Sentinel-1 + Sentinel-2
    """

    def __init__(
        self,
        state_dim=4,
        ensemble_size=50,
        process_noise=0.05,
        observation_noise_s1=0.15,
        observation_noise_s2=0.15,
        seed=42,
    ):

        self.state_dim = state_dim
        self.ensemble_size = ensemble_size

        rng = np.random.default_rng(seed)

        # Initial ensemble around a neutral state.
        self.ensemble = rng.normal(
            loc=0.0,
            scale=0.5,
            size=(ensemble_size, state_dim)
        )

        self.process_noise = process_noise

        # Observation uncertainty.
        self.R_s1 = (
            observation_noise_s1 ** 2
        ) * np.eye(3)

        self.R_s2 = (
            observation_noise_s2 ** 2
        ) * np.eye(5)

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    def predict(self, transition_function, forcing):

        new_ensemble = []

        for state in self.ensemble:

            next_state = transition_function(
                state,
                forcing
            )

            new_ensemble.append(next_state)

        new_ensemble = np.asarray(new_ensemble)

        noise = np.random.normal(
            0.0,
            self.process_noise,
            size=new_ensemble.shape
        )

        self.ensemble = (
            new_ensemble + noise
        )

    # --------------------------------------------------
    # Generic EnKF update
    # --------------------------------------------------

    def update(
        self,
        observation,
        observation_function,
        observation_covariance
    ):

        X = self.ensemble

        # Predicted observations for every ensemble member
        Y = np.asarray([
            observation_function(x)
            for x in X
        ])

        # Ensemble means
        x_mean = X.mean(axis=0)
        y_mean = Y.mean(axis=0)

        # Anomalies
        X_anom = X - x_mean
        Y_anom = Y - y_mean

        # Covariances
        P_xy = (
            X_anom.T @ Y_anom
            / (self.ensemble_size - 1)
        )

        P_yy = (
            Y_anom.T @ Y_anom
            / (self.ensemble_size - 1)
        )

        P_yy = (
            P_yy
            + observation_covariance
            + 1e-6 * np.eye(len(y_mean))
        )

        # Kalman gain
        K = P_xy @ np.linalg.inv(P_yy)

        # Perturbed observations
        observation_noise = np.random.multivariate_normal(
            mean=np.zeros(len(observation)),
            cov=observation_covariance,
            size=self.ensemble_size
        )

        perturbed_obs = (
            observation
            + observation_noise
        )

        innovations = (
            perturbed_obs - Y
        )

        # State update
        self.ensemble = (
            X + innovations @ K.T
        )

    # --------------------------------------------------
    # Current state estimate
    # --------------------------------------------------

    def state_mean(self):

        return self.ensemble.mean(axis=0)

    def state_std(self):

        return self.ensemble.std(axis=0)


if __name__ == "__main__":

    print("Testing EnKF...")

    enkf = EnsembleKalmanFilter(
        state_dim=4,
        ensemble_size=50
    )

    print(
        "Initial state:",
        enkf.state_mean()
    )

    print(
        "Initial uncertainty:",
        enkf.state_std()
    )

    # Simple test transition
    def transition(state, forcing):
        return state + 0.01 * forcing.mean()

    # Simple observation model
    def observation_function(state):
        return state[:2]

    enkf.predict(
        transition,
        np.array([0.2, 0.1, 0.3])
    )

    observation = np.array([
        0.5,
        0.2
    ])

    enkf.update(
        observation,
        observation_function,
        0.1 * np.eye(2)
    )

    print(
        "Updated state:",
        enkf.state_mean()
    )

    print(
        "Updated uncertainty:",
        enkf.state_std()
    )

    print("\nEnKF test successful.")
