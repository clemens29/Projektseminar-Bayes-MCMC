import csv

def count_germany_matches_and_wins(csv_file):
    germany_matches = 0
    germany_wins = 0

    germany = ['Germany', 'West Germany', 'Germany DR']

    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            home_team = row['home_team']
            away_team = row['away_team']
            home_score = int(row['home_score'])
            away_score = int(row['away_score'])
            home_penalty = int(row['home_penalty'] if row['home_penalty'] else 0)
            away_penalty = int(row['away_penalty'] if row['away_penalty'] else 0)

            if home_team in germany or away_team in germany:
                germany_matches += 1
                
                # Check if Germany won
                if home_team in germany and home_score > away_score:
                    germany_wins += 1
                elif away_team in germany and away_score > home_score:
                    germany_wins += 1
                elif home_team in germany and home_score == away_score and home_penalty > away_penalty:
                    germany_wins += 1
                elif away_team in germany and home_score == away_score and away_penalty > home_penalty:
                    germany_wins += 1

    return germany_matches, germany_wins

# Pfad zur CSV-Datei anpassen
csv_file_path = 'matches_1930_2022.csv'  # Datei entsprechend benennen
germany_matches, germany_wins = count_germany_matches_and_wins(csv_file_path)

print(f"Deutschland hat {germany_matches} Spiele gespielt und davon {germany_wins} gewonnen.")
