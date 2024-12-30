import csv

def count_germany_matches_by_opponent_region(csv_file):
    europe_teams = {
        'Albania', 'Andorra', 'Armenia', 'Austria', 'Azerbaijan', 'Belarus', 'Belgium', 'Bosnia and Herzegovina', 
        'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic', 'Czechoslovakia', 'Denmark', 'England', 'Estonia', 
        'Faroe Islands', 'Finland', 'France', 'Georgia', 'Germany', 'Gibraltar', 'Greece', 'Hungary', 'Iceland', 
        'Ireland', 'Israel', 'Italy', 'Kazakhstan', 'Kosovo', 'Latvia', 'Liechtenstein', 'Lithuania', 'Luxembourg', 
        'Malta', 'Moldova', 'Monaco', 'Montenegro', 'Netherlands', 'North Macedonia', 'Northern Ireland', 
        'Norway', 'Poland', 'Portugal', 'Romania', 'Russia', 'San Marino', 'Scotland', 'Serbia', 'Slovakia', 
        'Slovenia', 'Soviet Union', 'Spain', 'Sweden', 'Switzerland', 'Turkey', 'Ukraine', 'Wales', 'West Germany', 
        'Yugoslavia', 'Republic of Ireland', 'FR Yugoslavia', 'Germany DR', 'Türkiye'
    }
    
    germany_matches = {"europe": 0, "non_europe": 0}
    germany_wins = {"europe": 0, "non_europe": 0}
    germany = ['Germany', 'West Germany']

    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            home_team = row['home_team']
            away_team = row['away_team']
            home_score = int(row['home_score'])
            away_score = int(row['away_score'])
            home_penalty = int(row['home_penalty'] if row['home_penalty'] else 0)
            away_penalty = int(row['away_penalty'] if row['away_penalty'] else 0)

            # Prüfen, ob Deutschland beteiligt ist
            if home_team in germany or away_team in germany:
                # Gegner bestimmen
                opponent = away_team if home_team in germany else home_team
                is_european = opponent in europe_teams

                # Kategorie zuordnen
                if is_european:
                    germany_matches["europe"] += 1
                else:
                    germany_matches["non_europe"] += 1

                # Sieg prüfen
                if (home_team in germany and home_score > away_score) or \
                   (away_team in germany and away_score > home_score) or \
                   (home_team in germany and home_score == away_score and home_penalty > away_penalty) or \
                   (away_team in germany and home_score == away_score and away_penalty > home_penalty):
                    if is_european:
                        germany_wins["europe"] += 1
                    else:
                        germany_wins["non_europe"] += 1

    return germany_matches, germany_wins

# Datei-Pfad zur CSV anpassen
csv_file_path = 'data/matches_1930_2022.csv'
germany_matches, germany_wins = count_germany_matches_by_opponent_region(csv_file_path)
total_matches = germany_matches["europe"] + germany_matches["non_europe"]
total_wins = germany_wins["europe"] + germany_wins["non_europe"]

# Ergebnisse anzeigen
print(f"Gegen europäische Teams: {germany_matches['europe']} Spiele, {germany_wins['europe']} Siege.")
print(f"Gegen nicht-europäische Teams: {germany_matches['non_europe']} Spiele, {germany_wins['non_europe']} Siege.")
print(f"Gesamt: {total_matches} Spiele, {total_wins} Siege.")
