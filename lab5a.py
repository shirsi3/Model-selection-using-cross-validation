lsimport pandas as pd
import random


original_data = pd.read_csv("dataset.csv")

#30 random samples of benignware
benign_samples = original_data[original_data['label'] == 'benignware']
random_benign_samples = random.sample(list(benign_samples.index), 30)
benign_sample_dataframe = original_data.loc[random_benign_samples]

#30 random samples of malware
malware_samples = original_data[original_data['label'] == 'malware']
random_malware_samples = random.sample(list(malware_samples.index), 30)
malware_sample_dataframe = original_data.loc[random_malware_samples]

#combine the two sets 
combined_samples_dataframe = pd.concat([benign_sample_dataframe, malware_sample_dataframe])

#remove original data frame
reduced_data = original_data.drop(index=combined_samples_dataframe.index)

#reduced data to a new CSV file
reduced_data.to_csv("reduced_dataset.csv", index=False)

# Write the data 
samples_csv = combined_samples_dataframe[['MD5', 'label']]
samples_csv.to_csv("samples.csv", index=False)
