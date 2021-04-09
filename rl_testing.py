# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import pickle

from poke_env.player.env_player import Gen8EnvSinglePlayer
from poke_env.player.random_player import RandomPlayer
from poke_env.player.player import Player

from poke_env.player_configuration import PlayerConfiguration
from poke_env.server_configuration import ShowdownServerConfiguration

from rl.agents.dqn import DQNAgent
from rl.agents.sarsa import SARSAAgent
from rl.agents.cem import CEMAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy, BoltzmannQPolicy, SoftmaxPolicy
from rl.memory import SequentialMemory, EpisodeParameterMemory
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json


# graph = tf.compat.v1.get_default_graph()
import sys

# class DevNull:
#     def write(self, msg):
#         pass
#
# sys.stderr = DevNull()

# adjustment
import nest_asyncio
nest_asyncio.apply()


# We define our RL player
# It needs a state embedder and a reward computer, hence these two methods
class SimpleRLPlayer(Gen8EnvSinglePlayer):
    def embed_battle(self, battle):
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                )

        # We count how many pokemons have not fainted in each team
        remaining_mon_team = (
            len([mon for mon in battle.team.values() if mon.fainted]) / 6
        )
        remaining_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # Final vector with 10 components
        return np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [remaining_mon_team, remaining_mon_opponent],
            ]
        )

    def compute_reward(self, battle) -> float:
        return self.reward_computing_helper(
            battle, fainted_value=2, hp_value=1, victory_value=30
        )

# class TrainedRLPlayer(Player):
#     def __init__(self, model, *args, **kwargs):
#         Player.__init__(self, *args, **kwargs)
#         self.model = model
#
#     def choose_move(self, battle):
#         # print('battle', battle)
#         state = SimpleRLPlayer().embed_battle(battle=battle)
#         state = np.array(state).reshape((1,1,-1))
#         predictions = self.model.predict([state])[0]
#         action = np.argmax(predictions)
#         return SimpleRLPlayer()._action_to_move(action, battle)
class TrainedRLPlayer(Player):
    def __init__(self, model, *args, **kwargs):
        Player.__init__(self, *args, **kwargs)
        self.model = model

    def choose_move(self, battle):
        state = SimpleRLPlayer(player_configuration=PlayerConfiguration("mynameisgillian", "beanscool"),
        server_configuration=ShowdownServerConfiguration).embed_battle(battle=battle)
        state = np.array(state).reshape((1,1,-1))
        predictions = self.model.predict([state])[0]
        action = np.argmax(predictions)
        return SimpleRLPlayer()._action_to_move(action, battle)


class MaxDamagePlayer(RandomPlayer):
    def choose_move(self, battle):
        # If the player can attack, it will
        if battle.available_moves:
            # Finds the best move among available ones
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)


class RLPlayer(RandomPlayer):
    def choose_move(self, battle):
        # If the player can attack, it will
        if battle.available_moves:
            # Finds the best move among available ones
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)


NB_TRAINING_STEPS = 5000
NB_EVALUATION_EPISODES = 100

tf.random.set_seed(0)
np.random.seed(0)


# This is the function that will be used to train the dqn
def dqn_training(player, dqn, nb_steps):
    dqn.fit(player, nb_steps=nb_steps, verbose=2)
    player.complete_current_battle()
    print(
        "DQN Evaluation: %d victories out of %d episodes"
        % (player.n_won_battles, player.n_finished_battles)
    )


def dqn_evaluation(player, dqn, nb_episodes):
    # Reset battle statistics
    player.reset_battles()
    dqn.test(player, nb_episodes=nb_episodes, visualize=False, verbose=False)

    print(
        "DQN Evaluation: %d victories out of %d episodes"
        % (player.n_won_battles, nb_episodes)
    )


def test():

    # fine-tune gamma
    params1 = {
        'name': 'params1',
        'model_type': 'dqn_agent',  # 'dqn_agent', 'sarsa_agent', 'other'
        'l1_out': 128,
        'l2_out': 64,
        'gamma': 0.5,
        'target_model_update': 1,
        'delta_clip': 0.01,
        'nb_steps_warmup': 1000,
        'enable_dueling_dqn__': False,
        'enable_double_dqn__': True,
        'dueling_type__': 'avg',  # one of 'avg', 'max', 'naive'
    }
    params2 = {
        'name': 'params2',
        'model_type': 'dqn_agent',  # 'dqn_agent', 'sarsa_agent', 'other'
        'l1_out': 128,
        'l2_out': 64,
        'gamma': 0.7,
        'target_model_update': 1,
        'delta_clip': 0.01,
        'nb_steps_warmup': 1000,
        'enable_dueling_dqn__': False,
        'enable_double_dqn__': True,
        'dueling_type__': 'avg',  # one of 'avg', 'max', 'naive'
    }
    params3 = {
        'name': 'params3',
        'model_type': 'dqn_agent',  # 'dqn_agent', 'sarsa_agent', 'other'
        'l1_out': 128,
        'l2_out': 64,
        'gamma': 0.9,
        'target_model_update': 1,
        'delta_clip': 0.01,
        'nb_steps_warmup': 1000,
        'enable_dueling_dqn__': False,
        'enable_double_dqn__': True,
        'dueling_type__': 'avg',  # one of 'avg', 'max', 'naive'
    }
    # check deuling ability
    params4 = {
        'name': 'params4',
        'model_type': 'dqn_agent',  # 'dqn_agent', 'sarsa_agent', 'other'
        'l1_out': 128,
        'l2_out': 64,
        'gamma': 0.5,
        'target_model_update': 1,
        'delta_clip': 0.01,
        'nb_steps_warmup': 1000,
        'enable_dueling_dqn__': True,
        'enable_double_dqn__': True,
        'dueling_type__': 'avg',  # one of 'avg', 'max', 'naive'
    }

    # check delta clipping
    params5 = {
        'name': 'params5',
        'model_type': 'dqn_agent',  # 'dqn_agent', 'sarsa_agent', 'other'
        'l1_out': 128,
        'l2_out': 64,
        'gamma': 0.5,
        'target_model_update': 1,
        'delta_clip': 0.05,  # modifying delta clip causes problems
        'nb_steps_warmup': 1000,
        'enable_dueling_dqn__': False,
        'enable_double_dqn__': True,
        'dueling_type__': 'avg',  # one of 'avg', 'max', 'naive'
    }

    # check target_model_update
    params6 = {
        'name': 'params6',
        'model_type': 'dqn_agent',  # 'dqn_agent', 'sarsa_agent', 'other'
        'l1_out': 128,
        'l2_out': 64,
        'gamma': 0.5,
        'target_model_update': 0.01,
        'delta_clip': 0.01,
        'nb_steps_warmup': 1000,
        'enable_dueling_dqn__': False,
        'enable_double_dqn__': True,
        'dueling_type__': 'avg',  # one of 'avg', 'max', 'naive'
    }
    # also target_model update
    params7 = {
        'name': 'params7',
        'model_type': 'dqn_agent',  # 'dqn_agent', 'sarsa_agent', 'other'
        'l1_out': 128,
        'l2_out': 64,
        'gamma': 0.5,
        'target_model_update': 100,
        'delta_clip': 0.01,
        'nb_steps_warmup': 1000,
        'enable_dueling_dqn__': False,
        'enable_double_dqn__': True,
        'dueling_type__': 'avg',  # one of 'avg', 'max', 'naive'
    }

    # compare with sarsa agent
    params8 = {
        'name': 'params8',
        'model_type': 'sarsa_agent',  # 'dqn_agent', 'sarsa_agent', 'other'
        'l1_out': 128,
        'l2_out': 64,
        'gamma': 0.5,
        'target_model_update': 1,
        'delta_clip': 0.01,
        'nb_steps_warmup': 1000,
        'enable_dueling_dqn__': False,
        'enable_double_dqn__': True,
        'dueling_type__': 'avg',  # one of 'avg', 'max', 'naive'
    }

    # CEM agent
    params9 = {
        'name': 'params9',
        'model_type': 'other',  # 'dqn_agent', 'sarsa_agent', 'other'
        'l1_out': 128,
        'l2_out': 64,
        'gamma': 0.5,
        'target_model_update': 1,
        'delta_clip': 0.01,
        'nb_steps_warmup': 1000,
        'enable_dueling_dqn__': False,
        'enable_double_dqn__': True,
        'dueling_type__': 'avg',  # one of 'avg', 'max', 'naive'
    }

    # dueling no double max
    params10 = {
        'name': 'params10',
        'model_type': 'dqn_agent',  # 'dqn_agent', 'sarsa_agent', 'other'
        'l1_out': 128,
        'l2_out': 64,
        'gamma': 0.5,
        'target_model_update': 1,
        'delta_clip': 0.01,
        'nb_steps_warmup': 1000,
        'enable_dueling_dqn__': True,
        'enable_double_dqn__': True,
        'dueling_type__': 'avg',  # one of 'avg', 'max', 'naive'
    }
    # dueling max
    params11 = {
        'name': 'params11',
        'model_type': 'dqn_agent',  # 'dqn_agent', 'sarsa_agent', 'other'
        'l1_out': 128,
        'l2_out': 64,
        'l3_out': 32,
        'gamma': 0.5,
        'target_model_update': 1,
        'delta_clip': 0.01,
        'nb_steps_warmup': 1000,
        'enable_dueling_dqn__': True,
        'enable_double_dqn__': True,
        'dueling_type__': 'max',  # one of 'avg', 'max', 'naive'
    }
    # dueling naive
    params12 = {
        'name': 'params12',
        'model_type': 'dqn_agent',  # 'dqn_agent', 'sarsa_agent', 'other'
        'l1_out': 128,
        'l2_out': 64,
        'gamma': 0.5,
        'target_model_update': 1,
        'delta_clip': 0.0,
        'nb_steps_warmup': 1000,
        'enable_dueling_dqn__': True,
        'enable_double_dqn__': True,
        'dueling_type__': "naive" # one of 'avg', 'max', 'naive'
    }
    params13 = {
        'name': 'params13',
        'model_type': 'dqn_agent',  # 'dqn_agent', 'sarsa_agent', 'other'
        'l1_out': 128,
        'l2_out': 64,
        'L3_out': 32,
        'gamma': 0.5,
        'target_model_update': 1,
        'delta_clip': 0.01,
        'nb_steps_warmup': 1000,
        'enable_dueling_dqn__': False,
        'enable_double_dqn__': True,
        'dueling_type__': 'avg',  # one of 'avg', 'max', 'naive'
    }

    grid = [#params1, params2, params3, params4,
            #params5, params6, params7, params8,
            params9, params10, params11, params12,
            params13]
    # grid = [params1]

    n_trials = 1

    for i, params in enumerate(grid):
        print("params {}".format(i))
        print(params)
        for j in range(n_trials):
            print('trial {}'.format(j))
            try:
                main(params)
            except Exception as e:
                print(e)
                print('ERROR in test() iteration, going to next set of params')

def get_model(params=None):
    """

    :param params:
        - 'l1_out'
        - 'l2_out'
        - 'n_actions'

    :return:
    model - keras sequential model
    """

    if params is None:
        print('NEED params for model')
        return -1

    n_actions = params['n_actions']
    l1_out = params['l1_out']
    l2_out = params['l2_out']

    model = Sequential()
    model.add(Dense(l1_out, activation="elu", input_shape=(1, 10)))

    # Our embedding have shape (1, 10), which affects our hidden layer
    # dimension and output dimension
    # Flattening resolve potential issues that would arise otherwise
    model.add(Flatten())
    model.add(Dense(l2_out, activation="elu"))

    if 'l3_out' in params:
        model.add(Dense(params['l3_out'], activation="elu"))

    if params['model_type'] in {'dqn_agent', 'sarsa_agent'}:
        model.add(Dense(n_actions, activation="linear"))
    else:
        model.add(Dense(n_actions, activation="softmax"))
    return model


async def main_ladder(model):
    # We create a random player

    # model = load_model('pokemon_project/model_25000')
    # model = load_model('model_25000')

    player = TrainedRLPlayer(model,
                             player_configuration=PlayerConfiguration("mynameisgillian", "beanscool"),
                             server_configuration=ShowdownServerConfiguration,
                             )

    # Sending challenges to 'your_username'
    await player.send_challenges("UW_Brock", n_challenges=1)

    # Accepting one challenge from any user
    # await player.accept_challenges(None, 1)

    # Accepting three challenges from 'your_username'
    # await player.accept_challenges('your_username', 3)

    # Playing 5 games on the ladder
    # await player.ladder(5)

    # Print the rating of the player and its opponent after each battle
    for battle in player.battles.values():
        print(battle.rating, battle.opponent_rating)


def main(params=None):
    """
    main function
    :return:
    None
    """
    if params is None:
        params = {
            'model_type': 'dqn_agent',
            'l1_out': 128,
            'l2_out': 64,
            'gamma': 0.5,
            'target_model_update': 1,
            'delta_clip': 0.01,
            'nb_steps_warmup': 1000
        }



    model_type = 'dqn_agent'
    env_player = SimpleRLPlayer(battle_format="gen8randombattle")
    # print('env_player',env_player)
    # print('help', help(env_player))
    env_player2 = SimpleRLPlayer(battle_format="gen8randombattle")

    opponent = RandomPlayer(battle_format="gen8randombattle")
    second_opponent = MaxDamagePlayer(battle_format="gen8randombattle")


    # Output dimension
    n_action = len(env_player.action_space)

    # model_params = {
    #     'n_actions': n_action,
    #     'l1_out': 128,
    #     'l2_out': 64,
    #     'model_type': params['model_type']
    # }
    model_params = params
    model_params['n_actions'] = n_action

    model = get_model(model_params)

    # print('first model summary')
    # print(model.summary())
    # model = Sequential()
    # model.add(Dense(128, activation="elu", input_shape=(1, 10)))
    #
    # # Our embedding have shape (1, 10), which affects our hidden layer
    # # dimension and output dimension
    # # Flattening resolve potential issues that would arise otherwise
    # model.add(Flatten())
    # model.add(Dense(64, activation="elu"))
    # model.add(Dense(n_action, activation="linear"))

    # elu activation is similar to relu
    # https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#elu

    # determine memory type
    if params['model_type'] in {'dqn_agent', 'sarsa_agent'}:
        # memory = SequentialMemory(limit=10000, window_length=1)
        memory = SequentialMemory(limit=NB_TRAINING_STEPS, window_length=1)
    else:
        memory = EpisodeParameterMemory(limit=10000, window_length=1)

    # Ssimple epsilon greedy
    # What is linear annealed policy?
    # - this policy gives gradually decreasing thresholds for the epsilon greedy policy
    # - it acts as a wrapper around epsilon greedy to feed in a custom threshold
    pol_steps = NB_TRAINING_STEPS
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.05,
        value_test=0,
        nb_steps=pol_steps,
    )
    # pol_steps = NB_TRAINING_STEPS
    policy_boltz = BoltzmannQPolicy(tau=1)
    # policy = LinearAnnealedPolicy(
    #     BoltzmannQPolicy(),
    #     attr="tau",
    #     value_max=1.0,
    #     value_min=0.05,
    #     value_test=0,
    #     nb_steps=pol_steps,
    # )
    # policy = policy_boltz

    # Defining our DQN
    # model = tf.keras.models.load_model('dqn_v_dqn')
    # print('second model summary')
    # print(model.summary())

    # load json and create model
    # json_file = open('dqn_alt_10k.json', 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # model = model_from_json(loaded_model_json)
    # # load weights into new model
    # model.load_weights("dqn_alt_10k.h5")
    # print("Loaded model from disk")

    if params['model_type'] == 'dqn_agent':
        dqn = DQNAgent(
            model=model,
            nb_actions=len(env_player.action_space),
            policy=policy,
            memory=memory,
            nb_steps_warmup=params['nb_steps_warmup'],
            gamma=params['gamma'],
            target_model_update=params['target_model_update'],
            # delta_clip=0.01,
            delta_clip=params['delta_clip'],
            enable_double_dqn=params['enable_double_dqn__'],
            enable_dueling_network=params['enable_double_dqn__'],
            dueling_type=params['dueling_type__']
        )
        dqn.compile(Adam(lr=0.00025), metrics=["mae"])

    elif params['model_type'] == 'sarsa_agent':
        dqn = SARSAAgent(
            model=model,
            nb_actions=len(env_player.action_space),
            policy=policy,
            nb_steps_warmup=params['nb_steps_warmup'],
            gamma=params['gamma'],
            delta_clip=params['delta_clip']
        )
        dqn.compile(Adam(lr=0.00025), metrics=["mae"])
    else:
        # CEMAgent
        # https://towardsdatascience.com/cross-entropy-method-for-reinforcement-learning-2b6de2a4f3a0
        dqn = CEMAgent(
            model=model,
            nb_actions=len(env_player.action_space),
            memory=memory,
            nb_steps_warmup=params['nb_steps_warmup']
        )
        # different compile function
        dqn.compile()

    # dqn.compile(Adam(lr=0.00025), metrics=["mae"])
    # opponent dqn
    dqn_opponent = DQNAgent(
        model=model,
        nb_actions=len(env_player.action_space),
        policy=policy,
        memory=memory,
        nb_steps_warmup=params['nb_steps_warmup'],
        gamma=params['gamma'],
        target_model_update=params['target_model_update'],
        # delta_clip=0.01,
        delta_clip=params['delta_clip'],
        enable_double_dqn=params['enable_double_dqn__'],
        enable_dueling_network=params['enable_double_dqn__'],
        dueling_type=params['dueling_type__']
    )
    dqn_opponent.compile(Adam(lr=0.00025), metrics=["mae"])
    # NB_TRAINING_STEPS = NB_TRAINING_STEPS

    # rl_opponent = TrainedRLPlayer(model)
    # Training
    rounds = 4
    n_steps = NB_TRAINING_STEPS // rounds

    # practice pickleing model
    # pickle.dumps(model)

    for k in range(rounds):
        env_player.play_against(
            env_algorithm=dqn_training,
            opponent=opponent,
            env_algorithm_kwargs={"dqn": dqn, "nb_steps": n_steps},
        )
        env_player.play_against(
            env_algorithm=dqn_training,
            opponent=second_opponent,
            env_algorithm_kwargs={"dqn": dqn, "nb_steps": n_steps},
        )





        # pickle model

        # update opponent
        # rl_opponent = TrainedRLPlayer(model)
        # env_player.play_against(
        #     env_algorithm=dqn_training,
        #     opponent=second_opponent,
        #     env_algorithm_kwargs={"dqn": dqn, "nb_steps": n_steps},
        # )
    #
    #
    #     # save model with h5 and json
    #     model_json = model.to_json()
    #     with open("dqn_alt_10k.json", "w") as json_file:
    #         json_file.write(model_json)
    #     # serialize weights to HDF5
    #     model.save_weights("dqn_alt_10k.h5")
    #     print("Saved model to disk")

        # model.save("dqn_new_%d" % NB_TRAINING_STEPS * (k + 1))
    model.save(params["name"] + "_model" + "_ladder")



    # with open('model.pkl', 'wb') as f:
    #     pickle.dump(model, f)
    #
    # loaded_model = pickle.load(open("model.pkl", 'rb'))

    # for w1, w2 in zip(model.get_weights(), loaded_model.get_weights()):
    #     tf.debugging.assert_equal(w1, w2)

    # Evaluation
    print("Results against random player:")
    env_player.play_against(
        env_algorithm=dqn_evaluation,
        opponent=opponent,
        env_algorithm_kwargs={"dqn": dqn, "nb_episodes": NB_EVALUATION_EPISODES},
    )

    print("\nResults against max player:")
    env_player.play_against(
        env_algorithm=dqn_evaluation,
        opponent=second_opponent,
        env_algorithm_kwargs={"dqn": dqn, "nb_episodes": NB_EVALUATION_EPISODES},
    )
    return model


# main_ladder(model)



if __name__ == "__main__":

    test()
    # model_type = 'other'
    # env_player = SimpleRLPlayer(battle_format="gen8randombattle")
    #
    # opponent = RandomPlayer(battle_format="gen8randombattle")
    # second_opponent = MaxDamagePlayer(battle_format="gen8randombattle")
    #
    # # Output dimension
    # n_action = len(env_player.action_space)
    #
    # model = Sequential()
    # model.add(Dense(128, activation="elu", input_shape=(1, 10)))
    #
    # # Our embedding have shape (1, 10), which affects our hidden layer
    # # dimension and output dimension
    # # Flattening resolve potential issues that would arise otherwise
    # model.add(Flatten())
    # model.add(Dense(64, activation="elu"))
    # model.add(Dense(n_action, activation="linear"))
    #
    # # elu activation is similar to relu
    # # https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#elu
    #
    # memory = SequentialMemory(limit=10000, window_length=1)
    #
    # # Ssimple epsilon greedy
    # policy = LinearAnnealedPolicy(
    #     EpsGreedyQPolicy(),
    #     attr="eps",
    #     value_max=1.0,
    #     value_min=0.05,
    #     value_test=0,
    #     nb_steps=10000,
    # )
    #
    # # Defining our DQN
    # if model_type == 'dqn_agent':
    #     dqn = DQNAgent(
    #         model=model,
    #         nb_actions=len(env_player.action_space),
    #         policy=policy,
    #         memory=memory,
    #         nb_steps_warmup=1000,
    #         gamma=0.5,
    #         target_model_update=1,
    #         delta_clip=0.01,
    #         enable_double_dqn=True,
    #     )
    # elif model_type == 'sarsa_agent':
    #     dqn = SARSAAgent(
    #         model=model,
    #         nb_actions=len(env_player.action_space),
    #         policy=policy,
    #         nb_steps_warmup=1000,
    #         gamma=0.5,
    #         delta_clip=0.01
    #     )
    # else:
    #     # CEMAgent
    #     dqn = CEMAgent(
    #         model=model,
    #         nb_actions=len(env_player.action_space),
    #         memory=memory,
    #         nb_steps_warmup=1000,
    #         gamma=0.5,
    #         delta_clip=0.01
    #     )
    #
    # dqn.compile(Adam(lr=0.00025), metrics=["mae"])
    #
    # # Training
    # env_player.play_against(
    #     env_algorithm=dqn_training,
    #     opponent=opponent,
    #     env_algorithm_kwargs={"dqn": dqn, "nb_steps": NB_TRAINING_STEPS},
    # )
    # model.save("model_%d" % NB_TRAINING_STEPS)
    #
    # # Evaluation
    # print("Results against random player:")
    # env_player.play_against(
    #     env_algorithm=dqn_evaluation,
    #     opponent=opponent,
    #     env_algorithm_kwargs={"dqn": dqn, "nb_episodes": NB_EVALUATION_EPISODES},
    # )
    #
    # print("\nResults against max player:")
    # env_player.play_against(
    #     env_algorithm=dqn_evaluation,
    #     opponent=second_opponent,
    #     env_algorithm_kwargs={"dqn": dqn, "nb_episodes": NB_EVALUATION_EPISODES},
    # )
