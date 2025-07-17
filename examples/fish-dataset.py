import torch
import pandas as pd

url = "https://raw.githubusercontent.com/kittenpub/database-repository/main/Fish_Dataset_Pytorch.csv"
fish_data = pd.read_csv(url)
print(fish_data.head())

#   Species  Weight  Length1  Length2  Length3   Height   Width
# 0   Bream   242.0     23.2     25.4     30.0  11.5200  4.0200
# 1   Bream   290.0     24.0     26.3     31.2  12.4800  4.3056
# 2   Bream   340.0     23.9     26.5     31.1  12.3778  4.6961
# 3   Bream   363.0     26.3     29.0     33.5  12.7300  4.4555
# 4   Bream   430.0     26.5     29.0     34.0  12.4440  5.1340

features = fish_data.drop('Species', axis=1).values
labels = fish_data['Species'].values

features_tensor = torch.tensor(features, dtype=torch.float32)
print(f"Shape of the fish tensor: {features_tensor.shape}")

labels_tensor = torch.tensor(labels)
print(f"Shape of the labels tensor: {labels_tensor.shape}")