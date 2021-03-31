import asyncio
import time
import numpy as np
# import os
# os.environ['KERAS_BACKEND'] = 'theano'
# import tensorflow.keras
# Using Theano backend.
import nest_asyncio
nest_asyncio.apply()

from poke_env.player_configuration import PlayerConfiguration
from poke_env.server_configuration import ServerConfiguration
from poke_env.player.random_player import RandomPlayer
from poke_env.player.env_player import Gen8EnvSinglePlayer
from poke_env.player.player import Player
from tabulate import tabulate
from poke_env.player.utils import cross_evaluate
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from tensorflow.keras.optimizers import Adam



# Demo class
class YourFirstAgent(Player):
    """
    Simple agent class to play Pokemon showdown
    """
    def choose_move(self, battle):
        for move in battle.available_moves:
            if move.base_power > 90:
                # A powerful move! Let's use it
                return self.create_order(move)

        # No available move? Let's switch then!
        for switch in battle.available_switches:
            if switch.current_hp_fraction > battle.active_pokemon.current_hp_fraction:
                # This other pokemon has more HP left... Let's switch it in?
                return self.create_order(switch)

        # Not sure what to do?
        return self.choose_random_move(battle)

# This will work on servers that do not require authentication, which is the
# case of the server launched in our 'Getting Started' section
my_player_config = PlayerConfiguration("my_username", None)


# This object can be used with a player connecting to a server using authentication
# The user 'my_username' must exist and have 'super-secret-password' as his password
my_player_config = PlayerConfiguration("my_username", "super-secret-password")


# If your server is accessible at my.custom.host:5432, and your authentication
# endpoint is authentication-endpoint.com/action.php?
my_server_config= ServerConfiguration(
    "my.custom.host:5432",
    "authentication-endpoint.com/action.php?"
)

# You can now use my_server_config with a Player object

# Create a MaxDamagePlayer
class MaxDamagePlayer(Player):
    """
    Player that aims to achieve max damage
    """
    def choose_move(self, battle):
        # If the player can attack, it will
        if battle.available_moves:
            # Finds the best move among available ones
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)
        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)

class SimpleRLPlayer(Gen8EnvSinglePlayer):
    def embed_battle(self, battle):
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = move.base_power / 100 # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                )

        # We count how many pokemons have not fainted in each team
        remaining_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        remaining_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # Final vector with 10 components
        return np.concatenate(
            [moves_base_power, moves_dmg_multiplier, [remaining_mon_team, remaining_mon_opponent]]
        )

    def compute_reward(self, battle) -> float:
        return self.reward_computing_helper(
            battle,
            fainted_value=2,
            hp_value=1,
            victory_value=30,
        )


async def evaluate_random(n_players=3):
    """
    Evaluate random players playing against each other

    :return:
        None
    """
    # We create three random players
    players = [
        RandomPlayer(max_concurrent_battles=10) for _ in range(n_players)
    ]

    cross_evaluation = await cross_evaluate(players, n_challenges=20)

    # Defines a header for displaying results
    table = [["-"] + [p.username for p in players]]

    # Adds one line per player with corresponding results
    for p_1, results in cross_evaluation.items():
        table.append([p_1] + [cross_evaluation[p_1][p_2] for p_2 in results])

    # Displays results in a nicely formatted table.
    print(tabulate(table))


async def evaluate_max_damage():
    """
    Evaluate performance of max damage player against regular player
    :return:
        None
    """
    start = time.time()

    # We create two players.
    random_player = RandomPlayer(
        battle_format="gen8randombattle",
    )
    max_damage_player = MaxDamagePlayer(
        battle_format="gen8randombattle",
    )

    # Now, let's evaluate our player
    await max_damage_player.battle_against(random_player, n_battles=100)

    print(
        "Max damage player won %d / 100 battles [this took %f seconds]"
        % (
            max_damage_player.n_won_battles, time.time() - start
        )
    )


def evaluate_dqn():
    """
    Evaluate the performance of a DQN
    :return:
    """
    env_player = SimpleRLPlayer(battle_format="gen8randombattle")

    # define opponents
    opponent = RandomPlayer(battle_format="gen8randombattle")
    second_opponent = MaxDamagePlayer(battle_format="gen8randombattle")

    # Output dimension
    n_action = len(env_player.action_space)

    #TODO: manually change to 18 but should be 22
    n_action = 18

    model = Sequential()
    model.add(Dense(128, activation="elu", input_shape=(1, 10,)))

    # Our embedding have shape (1, 10), which affects our hidden layer dimension and output dimension
    # Flattening resolve potential issues that would arise otherwise
    model.add(Flatten())
    model.add(Dense(64, activation="elu"))
    model.add(Dense(n_action, activation="linear"))

    memory = SequentialMemory(limit=10000, window_length=1)

    print(model.summary())

    # Simple epsilon greedy
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.05,
        value_test=0,
        nb_steps=10000,
    )

    # Defining our DQN
    dqn = DQNAgent(
        model=model,
        nb_actions=18,
        policy=policy,
        memory=memory,
        nb_steps_warmup=1000,
        gamma=0.5,
        target_model_update=1,
        delta_clip=0.01,
        enable_double_dqn=True,
    )

    dqn.compile(Adam(lr=0.00025), metrics=["mae"])

    # await env_player.play_against(
    #     env_algorithm=dqn_training,
    #     opponent=opponent,
    #     env_algorithm_kwargs={"dqn": dqn, "nb_steps": 100000},
    # )

    # Training
    env_player.play_against(
        env_algorithm=dqn_training,
        opponent=opponent,
        env_algorithm_kwargs={"dqn": dqn, "nb_steps": 100000},
    )

    # Evaluation
    print("Results against random player:")
    env_player.play_against(
        env_algorithm=dqn_evaluation,
        opponent=opponent,
        env_algorithm_kwargs={"dqn": dqn, "nb_episodes": 100},
    )

    print("\nResults against max player:")
    env_player.play_against(
        env_algorithm=dqn_evaluation,
        opponent=second_opponent,
        env_algorithm_kwargs={"dqn": dqn, "nb_episodes": 100},
    )


def dqn_training(player, dqn, nb_steps):
    dqn.fit(player, nb_steps=nb_steps)

    # This call will finished eventual unfinshed battles before returning
    player.complete_current_battle()


def dqn_evaluation(player, dqn, nb_episodes):
    # Reset battle statistics
    player.reset_battles()
    dqn.test(player, nb_episodes=nb_episodes, visualize=False, verbose=False)

    print(
        "DQN Evaluation: %d victories out of %d episodes"
        % (player.n_won_battles, nb_episodes)
    )


async def main():
# def main():

    ###############################################################
    #           Cross Evaluate Random Players
    ###############################################################

    # await evaluate_random()

    ###############################################################
    #           MaxDamagePlayer vs RandomPlayer
    ###############################################################

    # await evaluate_max_damage()

    ###################################################################
    #           DQN with keras-rl
    ###################################################################

    evaluate_dqn()


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
