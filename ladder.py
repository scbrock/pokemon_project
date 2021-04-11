import asyncio
import nest_asyncio
import numpy as np

nest_asyncio.apply()

from poke_env.player.player import Player
from poke_env.player.random_player import RandomPlayer
from poke_env.player.env_player import Gen8EnvSinglePlayer

from poke_env.player_configuration import PlayerConfiguration
from poke_env.server_configuration import ShowdownServerConfiguration

from tensorflow.keras.models import load_model


LOGIN_USERNAME = ''
LOGIN_PASSWORD = ''

# Unused for now - TrainedRLPlayer and SimpleRLPlayer represent model-dependent RL agents
class TrainedRLPlayer(Player):
    def __init__(self, model, *args, **kwargs):
        Player.__init__(self, *args, **kwargs)
        self.model = model

    def choose_move(self, battle):
        state = SimpleRLPlayer(player_configuration=PlayerConfiguration(LOGIN_USERNAME, LOGIN_PASSWORD),
        server_configuration=ShowdownServerConfiguration).embed_battle(battle=battle)
        state = np.array(state).reshape((1,1,-1))
        predictions = self.model.predict([state])[0]
        action = np.argmax(predictions)
        return SimpleRLPlayer()._action_to_move(action, battle)
      
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

# Class used for the rule-based Max Damage agent to play online
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
    
async def main():
    
    player = MaxDamagePlayer(player_configuration=PlayerConfiguration(LOGIN_USERNAME, LOGIN_PASSWORD),
                             server_configuration=ShowdownServerConfiguration)
    

    # Sending challenges to 'your_username'
    # await player.send_challenges("UW_Brock", n_challenges=1)

    # Playing X games on the ladder
    await player.ladder(40)

    # Print the rating of the player and its opponent after each battle
    for battle in player.battles.values():
        print(battle.rating, battle.opponent_rating)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())