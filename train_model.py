import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# ✅ Make sure model directory exists
os.makedirs('model', exist_ok=True)

# Expanded dataset with a wider variety of interests, skills, and career suggestions
data = {
    'interest': ['tech', 'tech', 'bio', 'art', 'math', 'bio', 'tech', 'art', 'science', 'ai', 'medicine', 'music', 'math',
                 'design', 'history', 'education', 'economics', 'engineering', 'business', 'politics'],
    'skill': ['coding', 'networking', 'research', 'drawing', 'statistics', 'lab', 'ai', 'design', 'experiment', 'machine learning',
              'surgery', 'music production', 'data analysis', 'graphic design', 'teaching', 'writing', 'marketing', 'robotics', 'management', 'negotiation'],
    'career': ['Software Engineer', 'Network Admin', 'Biologist', 'Graphic Designer', 'Data Analyst', 'Pharmacist', 'ML Engineer', 'UX Designer',
               'Scientist', 'AI Specialist', 'Surgeon', 'Music Producer', 'Data Scientist', 'Creative Director', 'Teacher', 'Professor',
               'Economist', 'Mechanical Engineer', 'Business Analyst', 'Politician']
}

df = pd.DataFrame(data)

# Prepare feature set (X) and labels (y)
X = pd.get_dummies(df[['interest', 'skill']])
y = df['career']

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier (to improve the model's suggestion capabilities)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model with cross-validation to ensure it generalizes well
model.fit(X_train, y_train)

# Evaluate the model's performance on the test set
accuracy = model.score(X_test, y_test)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Save the model as a pickle file
with open('model/recommender.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model training completed and saved as recommender.pkl")
