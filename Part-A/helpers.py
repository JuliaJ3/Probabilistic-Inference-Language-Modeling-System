from IPython.display import Image, display
from pgmpy.models import BayesianNetwork
from textworld_express import TextWorldExpressEnv
import numpy as np
import random
import re
import tempfile


def visualize_bayesian_network(model: BayesianNetwork) -> None:
    """
    Visualizes the given Bayesian network by generating a graph
    representation and displaying it as an image.

    Creates a temporary PNG file of the Bayesian network's graph using
    Graphviz and displays it in the IPython environment. The graph is
    generated from the provided BayesianNetwork model.

    Parameters:
        model (BayesianNetwork): The Bayesian network model to
        visualize.

    Returns:
        None
    """

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
        model_graph = model.to_graphviz()
        model_graph.draw(temp.name, prog="dot")
        display(Image(filename=temp.name))


class GhostTextWorldExpressEnv(TextWorldExpressEnv):

    def __init__(self, serverPath=None, envStepLimit=100):
        # Call the super constructor
        super().__init__(serverPath, envStepLimit)
        self.__ghost = None  # Ghost location (x, y) initially none
        self.__player = (0, 0)  # player location (x, y)
        self.__max_dist = 5  # Range the ghost is allowed to be in. Default is 5
        self.__noise_values = (
            None  # keep track of the +/- actual distance values, relative to the ghost
        )
        self.__noise_counts = (
            None  # Probability (non-normalized) of noise values, relative to the ghost
        )

    ### Override for the environment load function
    def load(self, gameName, gameParams):
        # Call the super method
        super().load(gameName, gameParams)
        # Figure out how many locations are in the gameParams. This will set max_dist
        m = re.search(r"numLocations\=([0-9]+)", gameParams)
        if m is not None and m.lastindex >= 1:
            # numLocations found in gameParams
            self.__max_dist = int(m[1]) + (1 if int(m[1]) % 2 == 0 else 0)
        else:
            # Relying on defaults
            self.__max_dist = 5
        # Initialize noise values and noise counts
        self.__noise_values = np.array(
            [i - self.__max_dist for i in range((self.__max_dist * 2) + 1)]
        )
        self.__noise_counts = np.array(
            list(
                map(
                    lambda x: int(x),
                    [2 ** (self.__max_dist - abs(v)) for v in self.__noise_values],
                )
            )
        )
        self.__ghost = None
        self.__player = (0, 0)

    ### Override for the environment reset function
    def reset(
        self,
        seed=None,
        gameFold=None,
        gameName=None,
        gameParams=None,
        generateGoldPath=False,
    ):
        # Call the super method
        obs, infos = super().reset(
            seed, gameFold, gameName, gameParams, generateGoldPath
        )
        # Randomly choose the ghost's location
        while self.__ghost is None or self.__ghost == (0, 0):
            self.__ghost = (
                random.choice(
                    list(map(lambda x: -1 * x, list(range(1, self.__max_dist + 1))))
                    + list(range(self.__max_dist + 1))
                ),
                random.choice(
                    list(map(lambda x: -1 * x, list(range(1, self.__max_dist + 1))))
                    + list(range(self.__max_dist + 1))
                ),
            )
        # reset the player
        self.__player = (0, 0)
        # Compute noisy distance to ghost
        dist = abs(self.__player[0] - self.__ghost[0]) + abs(
            self.__ghost[1] - self.__player[1]
        )  # Manhattan distance
        noisy_distance = int(
            max(
                0,
                dist
                + np.random.choice(
                    self.__noise_values,
                    p=self.__noise_counts / self.__noise_counts.sum(),
                ),
            )
        )
        noisy_distance = min(4 * self.__max_dist, noisy_distance)

        x_dist = abs(0 - self.__ghost[0])  # X distance
        noisy_distance_x = int(
            max(
                0,
                x_dist
                + np.random.choice(
                    self.__noise_values,
                    p=self.__noise_counts / self.__noise_counts.sum(),
                ),
            )
        )
        noisy_distance_x = min(4 * self.__max_dist, noisy_distance_x)

        # Make distribution
        infos["distribution"] = self.__make_distribution(noisy_distance)
        infos["x_dist_distribution"] = self.__make_distribution(noisy_distance_x)

        # Add noisy ghost distance to infos
        infos["ghost"] = noisy_distance
        # Add player location to infos
        infos["player"] = self.__player
        # add the 'report' action to the valid actions
        infos["validActions"] = infos["validActions"]
        # Add distance info to observation
        if obs == infos["look"]:
            infos["look"] = self.__make_ghost_obs(infos["look"], noisy_distance)
            infos["observation"] = infos["look"]
            obs = infos["look"]
        else:
            infos["look"] = self.__make_ghost_obs(infos["look"], noisy_distance)
        print("self.__ghost: ", self.__ghost)
        return obs, infos

    def get_max_dist(self):
        return self.__max_dist

    def step(self, action: str):
        # If ghost location is none, then the player cannot perform any actions
        # Check to see if the action is a 'report'
        if action[0:6] == "report":
            # Player is reporting
            words = action.strip().split()
            if len(words) >= 3:
                x = int(words[1])  # x position being guessed
                y = int(words[2])  # y position being guessed
                if (x, y) == self.__ghost:
                    # Guess is correct, report True and terminate game
                    self.__ghost = None
                    return "True", 1.0, True, {}
                else:
                    # Guess is incorrect, report False and terminate game
                    self.__ghost = None
                    return "False", -1.0, True, {}
        # ASSERT: not reporting (or report is ill-formatted)
        if self.__ghost is not None:
            # Call the super method
            observation, reward, isCompleted, infos = super().step(action)
            if observation == infos["look"]:
                # when the observation is the same as infos[look], it is because of a location change
                # Change player location (if at all)
                self.__player = self.__change_coordinates(self.__player, action)
            # Compute true distance and noisy distance
            dist = abs(self.__player[0] - self.__ghost[0]) + abs(
                self.__ghost[1] - self.__player[1]
            )  # Manhattan distance
            noisy_distance = int(
                max(
                    0,
                    dist
                    + np.random.choice(
                        self.__noise_values,
                        p=self.__noise_counts / self.__noise_counts.sum(),
                    ),
                )
            )
            noisy_distance = min(4 * self.__max_dist, noisy_distance)

            x_dist = abs(0 - self.__ghost[0])  # X distance
            noisy_distance_x = int(
                max(
                    0,
                    x_dist
                    + np.random.choice(
                        self.__noise_values,
                        p=self.__noise_counts / self.__noise_counts.sum(),
                    ),
                )
            )
            noisy_distance_x = min(4 * self.__max_dist, noisy_distance_x)

            # Make distribution
            infos["distribution"] = self.__make_distribution(noisy_distance)
            infos["x_dist_distribution"] = self.__make_distribution(noisy_distance_x)
            # Add noisy distance to ghost to infos
            infos["ghost"] = noisy_distance
            # Add player location to infos
            infos["player"] = self.__player
            # Add 'report' to valid actions
            infos["validActions"] = infos["validActions"]
            infos["observation"] = infos["look"]
            if observation == infos["look"]:
                # when the observation is the same as infos[look], it is because of a location change
                infos["look"] = self.__make_ghost_obs(infos["look"], noisy_distance)
                observation = infos["look"]
            else:
                infos["look"] = self.__make_ghost_obs(infos["look"], noisy_distance)
            # Return all the information
            return observation, reward, isCompleted, infos
        # ASSERT: ghost is not active, don't allow any action to be executed
        return "The game has ended.", 0.0, True, {}

    ### Make a distribution to share with agent
    def __make_distribution(self, noisy_distance):
        distribution = np.zeros(4 * self.__max_dist + 1)
        probs = self.__noise_counts / self.__noise_counts.sum()
        shifted = probs[
            max(0, self.__max_dist - noisy_distance) : min(
                len(probs), 5 * self.__max_dist - noisy_distance + 1
            )
        ]
        distribution[
            max(0, noisy_distance - self.__max_dist) : len(shifted)
            + max(0, noisy_distance - self.__max_dist)
        ] = shifted
        return distribution

    ### Make text about ghost distance to add to observation
    def __make_ghost_obs(self, obs, distance):
        return obs + "\nYou hear a ghost " + format(distance, ".2f") + " rooms away."

    ### Helper function to figure out how player's location changes
    def __change_coordinates(self, coordinates, action):
        if "move" in action:
            dir = action.split()[1]
            if dir == "north":
                return (coordinates[0], coordinates[1] + 1)
            elif dir == "south":
                return (coordinates[0], coordinates[1] - 1)
            elif dir == "east":
                return (coordinates[0] + 1, coordinates[1])
            elif dir == "west":
                return (coordinates[0] - 1, coordinates[1])
        return coordinates
