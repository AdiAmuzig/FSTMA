from ast import Str
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np


class belief:
    def __init__(self, mu: np.matrix, Sigma: np.matrix) -> None:
        """Generate the Gaussian belief values in 2D

        Args:
            mu (np.matrix): Mean values for the 2D Gaussian distribution
            Sigma (np.matrix): Variance values for the 2D Gaussian distribution
        """
        self.mu = mu
        self.Sigma = Sigma


class KalmanFilter:
    def __init__(self, A: np.matrix, B: np.matrix, C: np.matrix, R: np.matrix, Q: np.matrix, mu_0: np.matrix, Sigma_0: np.matrix, x_0: np.matrix, k: float) -> None:
        """Creating a Kalman Filater variable containing all necesary values
        for it to be able to calculate location uncertainty.

        Args:
            A (np.matrix): State transition model matrix
            B (np.matrix): Control input model matrix
            C (np.matrix): Observation model matrix
            R (np.matrix): State transition noise covariance matrix
            Q (np.matrix): Measurment noise covariance matrix
            mu_0 (np.matrix): Belief position mean matrix
            Sigma_0 (np.matrix): Belief position covariance matrix
            x_0 (np.matrix): Initial location
            k (float): Variable determining amount of standard diviations
        """
        self.A = A
        self.B = B
        self.C = C
        self.R = R
        self.Q = Q
        self.x_0 = x_0
        self.belief_0 = belief(mu_0, Sigma_0)
        self.k = k

    def plotTrajectory(self, X: List[np.matrix], color: Str = 'r') -> None:
        """Creates (but does not display) a plot for the real location of the agent over time.

        Args:
            X (List[np.matrix]): A list conataining the real locations of the agent along its path across consecotive time steps
            color (str, optional): The color of the plot. Defaults to 'r'.
        """
        t = len(X)
        x_location, y_location = fromMat2LocationVal(X, t)

        plt.title('Executed Path')
        plt.plot(x_location, y_location, color, label="Location")
        # plt.legend()
        plt.xlabel('x - axis')
        plt.ylabel('y - axis')
        plt.grid(color='lightgray', linestyle='--')
        plt.autoscale()

    def PropagateUpdateBelief(self, belief_minus_1: belief, u_t: np.matrix, z_t: np.matrix = np.matrix('0,0'), with_observations: bool = False) -> belief:
        """Propogate new belief state acording to Kalman Filter and the previos belief state

        Args:
            belief_minus_1 (belief): Previos belief state
            u_t (np.matrix): Motion model
            z_t (np.matrix, optional): observation model. Defaults to np.matrix('0,0').
            with_observations (bool, optional): does the agent have observation abilities. Defaults to False.

        Returns:
            belief: Next belief state
        """
        mu_bar_t = self.A @ belief_minus_1.mu + self.B @ u_t
        Sigma_bar_t = self.A @ belief_minus_1.Sigma @ self.A.T + self.R

        if with_observations == True:
            K = Sigma_bar_t @ self.C.T @ (self.C @
                                          Sigma_bar_t @ self.C.T + self.Q).I
            mu_t = mu_bar_t + K @ (z_t - self.C @ mu_bar_t)
            Sigma_t = (np.matrix(np.eye(len(K @ self.C))) -
                       K @ self.C) @ Sigma_bar_t

        else:
            mu_t = mu_bar_t
            Sigma_t = Sigma_bar_t

        belief_t = belief(mu_t, Sigma_t)
        return belief_t

    def SampleMotionModel(self, x: np.matrix, u: np.matrix) -> np.matrix:
        """Generate a new location for an agent with Gaussian location uncertainy

        Args:
            x (np.matrix): Current real location of the agent
            u (np.matrix): Given motion

        Returns:
            np.matrix: next position given by the motion model
        """
        epsilon_mean = np.transpose(np.zeros(len(x)))
        epsilon = np.matrix(
            np.random.multivariate_normal(epsilon_mean, self.R, 1))
        epsilon = epsilon.T
        x_next = self.A @ x + self.B @ u + epsilon
        return x_next

    def runSim(self, actions: List[np.matrix]) -> List[np.matrix]:
        """Given a set of actions, run a simulation of the agent's movements

        Args:
            actions (List[np.matrix]): A set of actions for the agent to perform

        Returns:
            List[np.matrix]: The locations of the agent along its path
        """
        X = [self.x_0]
        for i in range(len(actions)):
            x_new = self.SampleMotionModel(X[-1], actions[i])
            X.append(x_new)
        return X


def fromMat2LocationVal(X: List[np.matrix], t: int) -> Tuple[List[float], List[float]]:
    """Transfer from a list of matricies (each a length of 2) to 2 lists of values 

    Args:
        X (List[np.matrix]): List of matricies each the size of 2
        t (int): Length of the list of matricies

    Returns:
        Tuple[List[float], List[float]]: 
            List[float] - x_location - A list of all the first values in each matrix 
            List[float] - y_location - A list of all the second values in each matrix 
    """
    x_location = list()
    y_location = list()

    for i in range(t):
        X_t = (X[i]).A
        x_location.extend(X_t[0])
        y_location.extend(X_t[1])

    return x_location, y_location
