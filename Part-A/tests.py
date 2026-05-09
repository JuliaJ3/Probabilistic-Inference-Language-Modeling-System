from pgmpy.models import BayesianNetwork
from helpers import GhostTextWorldExpressEnv


def test_lab_safety_network_nodes_and_edges(lab_safety_model: BayesianNetwork) -> None:
    """
    Tests the nodes and edges of the given Bayesian network model.

    Checks that the model has the expected nodes and edges. The expected
    nodes and edges are defined in the test case.

    Parameters:
        lab_safety_model (BayesianNetwork): The Bayesian network model to test.

    Returns:
        None
    """

    assert (
        len(lab_safety_model.nodes) == 7
    ), "Your model does not have the expected number of nodes. Be sure to review the scenario and the nodes that you need to include in the model."
    assert (
        len(lab_safety_model.edges) == 6
    ), "Your model does not have the expected number of edges. Be sure to review the scenario and the relationships between the events that you need to include in the model."

    print("Your Bayesian Network has the expected number of nodes and edges!")


def check_lab_safety_model(lab_safety_model: BayesianNetwork) -> None:
    """
    Checks the nodes and edges of the given Bayesian network model.

    Calls the test_lab_safety_network_nodes_and_edges function to check the
    nodes and edges of the given Bayesian network model.

    Parameters:
        lab_safety_model (BayesianNetwork): The Bayesian network model to check.

    Returns:
        None
    """

    assert (
        lab_safety_model.check_model()
    ), "Your model is not valid. Check that the CPDs all sum to 1 and that the CPDs associated with nodes are consistent with their parents and the model structure."

    print(
        "Your Bayesian Network has all probabilities summing to 1, and the CPDs are consistent with your defined model structure!"
    )

test_seeds = list(range(0, 20))
test_environments = [GhostTextWorldExpressEnv(envStepLimit=100)]
test_games = {
    "coin": [
        "numLocations=5,includeDoors=1,numDistractorItems=0",
        "numLocations=6,includeDoors=1,numDistractorItems=0",
        "numLocations=7,includeDoors=1,numDistractorItems=0",
        "numLocations=10,includeDoors=1,numDistractorItems=0",
        "numLocations=11,includeDoors=1,numDistractorItems=0",
    ],
    "mapreader": [
        "numLocations=5,maxDistanceApart=3,includeDoors=0,maxDistractorItemsPerLocation=0",
        "numLocations=8,maxDistanceApart=4,includeDoors=0,maxDistractorItemsPerLocation=0",
        "numLocations=11,maxDistanceApart=5,includeDoors=0,maxDistractorItemsPerLocation=0",
        "numLocations=15,maxDistanceApart=8,includeDoors=0,maxDistractorItemsPerLocation=0",
        "numLocations=20,maxDistanceApart=8,includeDoors=0,maxDistractorItemsPerLocation=0",
    ],
}


def test_agent(env, guess_x, guess_y):
    obs, _, _, _ = env.step("report " + str(guess_x) + " " + str(guess_y))
    print("Guess: (", guess_x, ", ", guess_y, ")")
    return obs == "True"


def run_all(
    run_agent: callable,
    environments=test_environments,
    games=test_games,
    seeds=test_seeds,
    alpha_param=0.90,
) -> None:
    # Results will contain a key (env type, game type, game params, seed) and values will be plans and total_rewards
    num_tests_ran = 0
    num_game_params = sum([len(params) for params in games.values()])
    num_tests_ran = len(environments) * num_game_params * len(seeds)
    results = {}
    # Iterate through all environments given
    for env in environments:
        # set global environment
        ENV = env
        # Iterate through all game types, the keys of the games dict
        for game_type in games:
            # Set the global game type
            GAME_TYPE = game_type
            # Iterate through all game parameters for the given game type in game dict
            for params in games[game_type]:
                # set the global game params
                GAME_PARAMS = params
                # load the environment
                ENV.load(gameName=GAME_TYPE, gameParams=GAME_PARAMS)

                # Iterate through all seeds
                for seed in seeds:
                    try:
                        # set the seed
                        print(GAME_TYPE, GAME_PARAMS, seed)
                        obs, infos = ENV.reset(seed)
                        x, y = run_agent(ENV, obs, infos, alpha=alpha_param)

                        # Store the score in the results
                        results[(type(env), game_type, params, seed)] = test_agent(
                            env, x, y
                        )
                    except Exception as e:
                        if str(e).startswith(
                            "An error occurred while calling o2.generateNewGameJSON."
                        ):
                            print(f"Skipping {type(ENV)} {seed} {params} {game_type}")
                            num_tests_ran -= 1
                        else:
                            print("Student code raises an error")
                            print("or you are not exporting your cells correctly")
                            print(e)
    return results, num_tests_ran


def test_all_configs(run_agent: callable, alpha_param=0.90) -> None:
    final_results, num_tests_ran = run_all(
        run_agent, test_environments, test_games, test_seeds, alpha_param
    )
    results = list(final_results.values())
    accuracy = results.count(True) / num_tests_ran

    # print(results)
    print("Number of tests ran: ", num_tests_ran)
    print("Number of tests passed: ", results.count(True))
    print("Agent accuracy: ", accuracy)
    assert accuracy >= alpha_param
    print("Passed! Accuracy greater than ", alpha_param)
