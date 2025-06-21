import pandas as pd
import matplotlib.pyplot as plt

# Load training data
df = pd.read_csv("archive (38)/training.csv")

# Emotion label mapping
emotion_labels = {
    0: "Sadness",
    1: "Joy",
    2: "Anger",
    3: "Fear",
    4: "Love",
    5: "Surprise"
}

# Count and rename labels
label_counts = df['label'].value_counts().sort_index()
label_counts.index = label_counts.index.map(emotion_labels)

# Plot
label_counts.plot(kind='bar', color='skyblue')
plt.title("Emotion Label Distribution")
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.grid(True)
plt.show()

