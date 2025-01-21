'''
This implements vanilla counterfactual regret for Kuhn poker.

Code from https://justinsermeno.com/posts/cfr/
'''

import numpy as np

# Number of actions a player can take at a decision node.
_N_ACTIONS = 2
_N_CARDS = 3


def main():
    """
    Run iterations of counterfactual regret minimization algorithm.
    """
    i_map = {}  # map of information sets
    n_iterations = 10000
    expected_game_value = 0

    players = ('Alice', 'Bob')

    for _ in range(n_iterations):
        expected_game_value += cfr(i_map, players)
        for _, v in i_map.items():
            v.next_strategy()

    expected_game_value /= n_iterations

    display_results(expected_game_value, i_map, players)


def cfr(i_map, players, history="", cards=None, prs=(1,1), pr_c=1):
    """
    Counterfactual regret minimization algorithm.

    Parameters
    ----------

    i_map: dict
        Dictionary of all information sets.
    history : [{'r', 'c', 'b'}], str
        A string representation of the game tree path we have taken.
        Each character of the string represents a single action:

        'r': random chance action
        'c': check/fold action
        'b': bet/call action

    cards : a tuple containing integers representing each player's cards

    pr_1 : (0, 1.0), float
        The probability that player A reaches `history`.
    pr_2 : (0, 1.0), float
        The probability that player B reaches `history`.
    pr_c: (0, 1.0), float
        The probability contribution of chance events to reach `history`.
    """
    if is_chance_node(history):
        return chance_util(i_map, players)

    if is_terminal(history):
        return terminal_util(history, cards)

    n = len(history)
    player_to_act_index = n % len(players)
    is_player_1 = player_to_act_index == 0
    info_set = get_info_set(i_map, cards[player_to_act_index], history)

    pr_1, pr_2 = prs

    strategy = info_set.strategy
    info_set.reach_pr += prs[player_to_act_index]

    # Counterfactual utility per action.
    action_utils = np.zeros(_N_ACTIONS)

    for i, action in enumerate(["c", "b"]):
        next_history = history + action
        if is_player_1:
            action_utils[i] = -1 * cfr(i_map, players, next_history,
                                       cards,
                                       (pr_1 * strategy[i], pr_2), pr_c)
        else:
            action_utils[i] = -1 * cfr(i_map, players, next_history,
                                       cards,
                                       (pr_1, pr_2 * strategy[i]), pr_c)

    # Utility of information set.
    util = sum(action_utils * strategy)
    regrets = action_utils - util
    if is_player_1:
        info_set.regret_sum += prs[1] * pr_c * regrets
    else:
        info_set.regret_sum += prs[0] * pr_c * regrets

    return util


def is_chance_node(history):
    """
    Determine if we are at a chance node based on tree history.
    """
    return history == ""


def chance_util(i_map, players):
    expected_value = 0
    n_possibilities = 6
    for i in range(_N_CARDS):
        for j in range(_N_CARDS):
            if i != j:
                expected_value += cfr(i_map, players, "rr", (i,j), (1,1), 1/n_possibilities)
    return expected_value/n_possibilities


def is_terminal(history):
    """
    Returns true if the history is a terminal history.
    """
    possibilities = {"rrcc": True, "rrcbc": True,
                     "rrcbb": True, "rrbc": True, "rrbb": True}
    return history in possibilities


def terminal_util(history, cards):
    """
    Returns the utility of a terminal history.
    """
    n = len(history)
    card_player = cards[0] if n % 2 == 0 else cards[1]
    card_opponent = cards[1] if n % 2 == 0 else cards[0]

    if history == "rrcbc" or history == "rrbc":
        # Last player folded. The current player wins.
        return 1
    elif history == "rrcc":
        # Showdown with no bets
        return 1 if card_player > card_opponent else -1

    # Showdown with 1 bet
    assert(history == "rrcbb" or history == "rrbb")
    return 2 if card_player > card_opponent else -2


def card_str(card):
    return 'QKA'[card]


def get_info_set(i_map, card, history):
    """
    Retrieve information set from dictionary.
    """
    key = card_str(card) + " " + history
    info_set = None

    if key not in i_map:
        info_set = InformationSet(key)
        i_map[key] = info_set
        return info_set

    return i_map[key]


class InformationSet():
    def __init__(self, key):
        self.key = key
        self.regret_sum = np.zeros(_N_ACTIONS)
        self.strategy_sum = np.zeros(_N_ACTIONS)
        self.strategy = np.repeat(1/_N_ACTIONS, _N_ACTIONS)
        self.reach_pr = 0
        self.reach_pr_sum = 0

    def next_strategy(self):
        self.strategy_sum += self.reach_pr * self.strategy
        self.strategy = self.calc_strategy()
        self.reach_pr_sum += self.reach_pr
        self.reach_pr = 0

    def calc_strategy(self):
        """
        Calculate current strategy from the sum of regret.
        """
        strategy = self.make_positive(self.regret_sum)
        total = sum(strategy)
        if total > 0:
            strategy = strategy / total
        else:
            n = _N_ACTIONS
            strategy = np.repeat(1/n, n)

        return strategy

    def get_average_strategy(self):
        """
        Calculate average strategy over all iterations. This is the
        Nash equilibrium strategy.
        """
        strategy = self.strategy_sum / self.reach_pr_sum

        # Purify to remove actions that are likely a mistake
        strategy = np.where(strategy < 0.001, 0, strategy)

        # Re-normalize
        total = sum(strategy)
        strategy /= total

        return strategy

    def make_positive(self, x):
        return np.where(x > 0, x, 0)

    def __str__(self):
        strategies = ['{:03.2f}'.format(x)
                      for x in self.get_average_strategy()]
        return '{} {}'.format(self.key.ljust(6), strategies)


def display_results(ev, i_map, players):
    print('player 1 expected value: {}'.format(ev))
    print('player 2 expected value: {}'.format(-1 * ev))

    sorted_items = sorted(i_map.items(), key=lambda x: x[0])

    for player_index, player in enumerate(players):
        print()
        print('{} strategies:'.format(player))

        for _, v in filter(lambda x: len(x[0]) % len(players) == player_index, sorted_items):
            print(v)


if __name__ == "__main__":
    main()
