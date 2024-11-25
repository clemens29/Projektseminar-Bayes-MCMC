
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import data
#import kagglehub
#Letzte Version des Datensatzes herunterladen
#dataset = "piterfm/fifa-football-world-cup"
#path = kagglehub.dataset_download(dataset)
#print("Path to dataset files:", path)
# Import data

#path = r"C:\Users\clemi\.cache\kagglehub\datasets\piterfm\fifa-football-world-cup\versions\24"

# Load data
matches = pd.read_csv(path + "/matches_1930_2022.csv")
winners = pd.read_csv(path + "/world_cup.csv")

# Show first rows
print(matches.head())
print(winners.head())
