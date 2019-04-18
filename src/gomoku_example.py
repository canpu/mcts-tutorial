from gomoku import *
from mcts import *


def gomoku_example() -> GomokuState:
    """ We create an example game; in this game, the black has taken great
        advantage, and we want to use MCTS to help black to turn advantage
        into victory
    """
    initial_state = GomokuState(use_default_heuristics=True, reward_player=0)
    initial_state.go((3, 5)).go((3, 6)).go((4, 6)).go((5, 7)).go((5, 5)) \
        .go((6, 4)).go((4, 5)).go((6, 5)).go((4, 4)).go((4, 7)).go((6, 6)) \
        .go((7, 7)).go((2, 5)).go((1, 5)).go((3, 3)).go((2, 2)).go((6, 7)) \
        .go((3, 4))
    return initial_state


def gomoku_example_solution() -> GomokuState:
    """ This is suggested, but not unique, strategy for black to win in the
        example
    """
    suggested_state = gomoku_example().__copy__()
    suggested_state.go((4, 3)).go((4, 2)).go((5, 3)).go((6, 3)).go((6, 2)) \
        .go((2, 6)).go((7, 1))
    return suggested_state


def gomoku_example_simulate(select_policy, expand_policy,
                            simulate_policy, backpropagate_policy,
                            black_heuristics: bool = False,
                            white_heuristics: bool = True,
                            samples_per_step: int = 1000,
                            random_seed: int = 0) -> GomokuState:
    black_state = gomoku_example().__copy__()
    black_state.heuristics = black_heuristics
    black_state.side = 0
    white_state = gomoku_example().__copy__()
    white_state.heuristics = white_heuristics
    white_state.side = 1

    black_mcts = MonteCarloSearchTree(black_state, samples=samples_per_step,
                                      max_tree_depth=15,
                                      tree_select_policy=select_policy,
                                      tree_expand_policy=expand_policy,
                                      rollout_policy=simulate_policy,
                                      backpropagate_method=backpropagate_policy)
    white_mcts = MonteCarloSearchTree(white_state, samples=samples_per_step,
                                      max_tree_depth=15,
                                      tree_select_policy=select_policy,
                                      tree_expand_policy=expand_policy,
                                      rollout_policy=simulate_policy,
                                      backpropagate_method=backpropagate_policy)
    while not black_state.is_terminal:
        # Black goes
        black_action = black_mcts.search_for_actions(search_depth=1,
                                                     random_seed=random_seed)[0]
        random_seed += 1
        print(black_action)
        black_mcts.update_root(black_action)
        white_mcts.update_root(black_action)
        black_state.go(black_action.position)
        white_state.go(black_action.position)

        if not black_state.is_terminal:
            # White goes
            white_action = white_mcts.search_for_actions(
                search_depth=1, random_seed=random_seed)[0]
            random_seed += 1
            print(white_action)
            black_mcts.update_root(white_action)
            white_mcts.update_root(white_action)
            black_state.go(white_action.position)
            white_state.go(white_action.position)

    return black_state


def simulate_with_black_sample_arbitrarily(select_policy, expand_policy,
                                           simulate_policy,
                                           backpropagate_policy,
                                           random_seed: int = 1000,
                                           num_iters: int = 1000) -> None:
    """ The black player estimates possible locations with uniform distribution
        anywhere on the board
    """
    (gomoku_example_simulate(black_heuristics=False, white_heuristics=True,
                             random_seed=random_seed,
                             select_policy=select_policy,
                             expand_policy=expand_policy,
                             simulate_policy=simulate_policy,
                             backpropagate_policy=backpropagate_policy,
                             samples_per_step=num_iters)
     .visualize())


def simulate_with_black_sample_neighborhood(select_policy, expand_policy,
                                            simulate_policy,
                                            backpropagate_policy,
                                            random_seed: int = 0,
                                            num_iters: int = 1000) -> None:
    """ The black player estimates possible locations only if a position's
        neighbor is not totally unoccupied
    """
    (gomoku_example_simulate(black_heuristics=True, white_heuristics=True,
                             random_seed=random_seed,
                             select_policy=select_policy,
                             expand_policy=expand_policy,
                             simulate_policy=simulate_policy,
                             backpropagate_policy=backpropagate_policy,
                             samples_per_step=num_iters)
     .visualize())


if __name__ == '__main__':
    print("===== Start of Simulation 1 =====")
    simulate_with_black_sample_arbitrarily(select, expand,
                                           default_rollout_policy,
                                           backpropagate,
                                           random_seed=1000,
                                           num_iters=1000)

    print("\n===== Start of Simulation 2 =====")
    simulate_with_black_sample_neighborhood(select, expand,
                                            default_rollout_policy,
                                            backpropagate,
                                            random_seed=1000,
                                            num_iters=1000)
