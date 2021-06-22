"""main to run RSP game"""

from multiple_games import MultipleGames
from serial_player import SerialPlayer
from random_player import RandomPlayer
from most_common_player import MostCommonPlayer
from historian_player import HistorianPlayer

PLAYER1_TYPE = ''
PLAYER2_TYPE = ''

while True:
    while PLAYER1_TYPE not in ('R', 'S', 'M', 'H'):
        PLAYER1_TYPE = input(
            "[R] - Random\n[S] - Serial\n[M] - Most Common\n[H] - Historian\n"
            "To start a game create a new player. Which type of player do you want to create: ")
    PLAYER1_NAME = input("Give this player a name: ")
    if PLAYER1_TYPE == 'R':
        PLAYER1 = RandomPlayer(PLAYER1_NAME)
    if PLAYER1_TYPE == 'S':
        PLAYER1 = SerialPlayer(PLAYER1_NAME)
    if PLAYER1_TYPE == 'M':
        PLAYER1 = MostCommonPlayer(PLAYER1_NAME)
    if PLAYER1_TYPE == 'H':
        PLAYER1_MEMORY = input("Give your historian player a memory value: ")
        PLAYER1 = HistorianPlayer(PLAYER1_NAME, int(PLAYER1_MEMORY))
    while PLAYER2_TYPE not in ('R', 'S', 'M', 'H'):
        PLAYER2_TYPE = input(
            "[R] - Random\n[S] - Serial\n[M] - Most Common\n[H] - "
            "Historian\nCreate an opponent. Which type of opponent player do you want to create: ")
    PLAYER2_NAME = input("Give this player a name: ")
    if PLAYER2_TYPE == 'R':
        PLAYER2 = RandomPlayer(PLAYER2_NAME)
    if PLAYER2_TYPE == 'S':
        PLAYER2 = SerialPlayer(PLAYER2_NAME)
    if PLAYER2_TYPE == 'M':
        PLAYER2 = MostCommonPlayer(PLAYER2_NAME)
    if PLAYER2_TYPE == 'H':
        PLAYER2_MEMORY = input("Give your historian player a memory value: ")
        PLAYER2 = HistorianPlayer(PLAYER2_NAME, int(PLAYER2_MEMORY))
    NR_OF_GAMES = input("How many games do you want the players to execute? ")
    GAMES = MultipleGames(PLAYER1, PLAYER2, int(NR_OF_GAMES))
    GAMES.run_games()
    while True:
        NEW_ROUND = input("Do you want to create a new game[Y] or "
                          "run another game with the same players[N]? ")
        if NEW_ROUND == 'Y':
            PLAYER1_TYPE = ''
            PLAYER2_TYPE = ''
            break
        NR_OF_GAMES = input(
            "How many games do you want the players to execute? ")
        GAMES = MultipleGames(PLAYER1, PLAYER2, int(NR_OF_GAMES))
        GAMES.run_games()
