import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVC
import pickle
import json
import datetime
import subprocess
import os
import hashlib


data = pd.read_csv("reduceddataset.csv")

data['target'] = (data['label'] == 'malware').astype(int)


features = data.drop(['MD5', 'label', 'target'], axis=1)

kbest = SelectKBest(score_func=chi2, k=15)
selected_features = kbest.fit_transform(features, data['target'])


classifier = SVC(kernel='linear')
classifier.fit(selected_features, data['target'])

with open("saved_detector.pkl", "wb") as f:
    pickle.dump((classifier, kbest), f)

samples_folder = "samples"
if os.path.exists(samples_folder):
    detector_results = []

    for root, dirs, files in os.walk(samples_folder):
        for file in files:
            file_path = os.path.join(root, file)

            
            with open(file_path, 'rb') as f:
                md5_hash = hashlib.md5(f.read()).hexdigest()

            
            current_time = datetime.datetime.now()
            timestamp = current_time.isoformat()
            
            log_message = {
                "timestamp": timestamp,
                "MD5 hash": md5_hash,
            }

            try:
                
                result = subprocess.run(["./capa", "-v", file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if result.returncode == 0:
                    features = []
                    lines = result.stdout.split("\n\n")
                    for line in lines[1:]:
                        features.append(line.split("\n")[0])
                    
                    
                    selected_features = kbest.transform([features])

                    
                    classification = classifier.predict(selected_features)
                    classification_label = "malware" if classification[0] == 1 else "benignware"
                    log_message["classification"] = classification_label

                else:
                    log_message["classification"] = "ND"  

            except Exception as e:
                log_message["classification"] = "ND" 
            
            detector_results.append(log_message)

    
    with open("detector.log", "w") as log_file:
        for result in detector_results:
            log_file.write(json.dumps(result) + "\n")
