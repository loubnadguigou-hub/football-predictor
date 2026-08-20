import os
import pandas as pd

# Ensure data folder exists
os.makedirs("data", exist_ok=True)

# Dataset containing Premier League squad players
players_data = [
    # Arsenal
    {"team": "Arsenal", "player": "Bukayo Saka", "pos": "FW", "xg90": 0.45, "xsot90": 1.10, "goals": 16, "xg": 15.2, "minutes": 2800},
    {"team": "Arsenal", "player": "Kai Havertz", "pos": "FW", "xg90": 0.42, "xsot90": 0.95, "goals": 13, "xg": 12.8, "minutes": 2600},
    {"team": "Arsenal", "player": "Gabriel Martinelli", "pos": "FW", "xg90": 0.32, "xsot90": 0.85, "goals": 8, "xg": 8.1, "minutes": 2100},
    {"team": "Arsenal", "player": "Martin Ødegaard", "pos": "MF", "xg90": 0.28, "xsot90": 0.75, "goals": 8, "xg": 7.5, "minutes": 2900},
    {"team": "Arsenal", "player": "Declan Rice", "pos": "MF", "xg90": 0.12, "xsot90": 0.35, "goals": 7, "xg": 4.2, "minutes": 3200},
    {"team": "Arsenal", "player": "Gabriel Magalhães", "pos": "DF", "xg90": 0.09, "xsot90": 0.25, "goals": 4, "xg": 3.1, "minutes": 3100},
    {"team": "Arsenal", "player": "William Saliba", "pos": "DF", "xg90": 0.04, "xsot90": 0.10, "goals": 2, "xg": 1.5, "minutes": 3420},

    # Manchester City
    {"team": "Manchester City", "player": "Erling Haaland", "pos": "FW", "xg90": 0.82, "xsot90": 1.75, "goals": 27, "xg": 26.1, "minutes": 2550},
    {"team": "Manchester City", "player": "Phil Foden", "pos": "FW", "xg90": 0.44, "xsot90": 1.20, "goals": 19, "xg": 14.5, "minutes": 2860},
    {"team": "Manchester City", "player": "Julian Alvarez", "pos": "FW", "xg90": 0.38, "xsot90": 0.98, "goals": 11, "xg": 11.2, "minutes": 2650},
    {"team": "Manchester City", "player": "Kevin De Bruyne", "pos": "MF", "xg90": 0.29, "xsot90": 0.88, "goals": 4, "xg": 4.5, "minutes": 1220},
    {"team": "Manchester City", "player": "Rodri", "pos": "MF", "xg90": 0.18, "xsot90": 0.52, "goals": 8, "xg": 5.8, "minutes": 2930},
    {"team": "Manchester City", "player": "Bernardo Silva", "pos": "MF", "xg90": 0.21, "xsot90": 0.60, "goals": 6, "xg": 5.2, "minutes": 2580},
    {"team": "Manchester City", "player": "Josko Gvardiol", "pos": "DF", "xg90": 0.11, "xsot90": 0.30, "goals": 4, "xg": 3.4, "minutes": 2320},

    # Liverpool
    {"team": "Liverpool", "player": "Mohamed Salah", "pos": "FW", "xg90": 0.65, "xsot90": 1.50, "goals": 18, "xg": 18.4, "minutes": 2540},
    {"team": "Liverpool", "player": "Darwin Núñez", "pos": "FW", "xg90": 0.68, "xsot90": 1.62, "goals": 11, "xg": 16.2, "minutes": 2040},
    {"team": "Liverpool", "player": "Cody Gakpo", "pos": "FW", "xg90": 0.41, "xsot90": 1.02, "goals": 8, "xg": 8.6, "minutes": 1620},
    {"team": "Liverpool", "player": "Diogo Jota", "pos": "FW", "xg90": 0.55, "xsot90": 1.30, "goals": 10, "xg": 8.1, "minutes": 1160},
    {"team": "Liverpool", "player": "Luis Díaz", "pos": "FW", "xg90": 0.38, "xsot90": 0.95, "goals": 8, "xg": 10.1, "minutes": 2640},
    {"team": "Liverpool", "player": "Alexis Mac Allister", "pos": "MF", "xg90": 0.16, "xsot90": 0.45, "goals": 5, "xg": 4.8, "minutes": 2600},
    {"team": "Liverpool", "player": "Virgil van Dijk", "pos": "DF", "xg90": 0.08, "xsot90": 0.22, "goals": 2, "xg": 2.8, "minutes": 3150},

    # Chelsea
    {"team": "Chelsea", "player": "Cole Palmer", "pos": "FW", "xg90": 0.58, "xsot90": 1.35, "goals": 22, "xg": 18.2, "minutes": 2770},
    {"team": "Chelsea", "player": "Nicolas Jackson", "pos": "FW", "xg90": 0.52, "xsot90": 1.25, "goals": 14, "xg": 14.8, "minutes": 2800},
    {"team": "Chelsea", "player": "Noni Madueke", "pos": "FW", "xg90": 0.35, "xsot90": 0.90, "goals": 5, "xg": 4.9, "minutes": 1060},
    {"team": "Chelsea", "player": "Conor Gallagher", "pos": "MF", "xg90": 0.19, "xsot90": 0.50, "goals": 5, "xg": 5.6, "minutes": 3130},
    {"team": "Chelsea", "player": "Enzo Fernández", "pos": "MF", "xg90": 0.15, "xsot90": 0.40, "goals": 3, "xg": 3.8, "minutes": 2180},
    {"team": "Chelsea", "player": "Thiago Silva", "pos": "DF", "xg90": 0.06, "xsot90": 0.18, "goals": 3, "xg": 2.1, "minutes": 2620},

    # Tottenham
    {"team": "Tottenham", "player": "Son Heung-Min", "pos": "FW", "xg90": 0.41, "xsot90": 1.15, "goals": 17, "xg": 12.1, "minutes": 2940},
    {"team": "Tottenham", "player": "Richarlison", "pos": "FW", "xg90": 0.54, "xsot90": 1.20, "goals": 11, "xg": 9.8, "minutes": 1500},
    {"team": "Tottenham", "player": "Dejan Kulusevski", "pos": "FW", "xg90": 0.25, "xsot90": 0.70, "goals": 8, "xg": 6.8, "minutes": 2770},
    {"team": "Tottenham", "player": "James Maddison", "pos": "MF", "xg90": 0.24, "xsot90": 0.72, "goals": 4, "xg": 4.9, "minutes": 2100},
    {"team": "Tottenham", "player": "Cristian Romero", "pos": "DF", "xg90": 0.12, "xsot90": 0.30, "goals": 5, "xg": 3.6, "minutes": 2800},

    # Manchester United
    {"team": "Manchester United", "player": "Rasmus Højlund", "pos": "FW", "xg90": 0.38, "xsot90": 0.85, "goals": 10, "xg": 9.2, "minutes": 2170},
    {"team": "Manchester United", "player": "Marcus Rashford", "pos": "FW", "xg90": 0.32, "xsot90": 0.80, "goals": 7, "xg": 8.1, "minutes": 2260},
    {"team": "Manchester United", "player": "Alejandro Garnacho", "pos": "FW", "xg90": 0.30, "xsot90": 0.88, "goals": 7, "xg": 8.5, "minutes": 2570},
    {"team": "Manchester United", "player": "Bruno Fernandes", "pos": "MF", "xg90": 0.31, "xsot90": 0.82, "goals": 10, "xg": 10.8, "minutes": 3120},
    {"team": "Manchester United", "player": "Scott McTominay", "pos": "MF", "xg90": 0.29, "xsot90": 0.65, "goals": 7, "xg": 6.1, "minutes": 1800},
    {"team": "Manchester United", "player": "Harry Maguire", "pos": "DF", "xg90": 0.09, "xsot90": 0.25, "goals": 2, "xg": 2.2, "minutes": 1650},
]

df = pd.DataFrame(players_data)
csv_path = "data/players_database.csv"
df.to_csv(csv_path, index=False)
print(f"🎉 Created dataset with {len(df)} players at {csv_path}!")