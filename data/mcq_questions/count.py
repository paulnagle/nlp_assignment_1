import pandas as pd
df = pd.read_csv('data/mcq_questions/mc_questions_file-1.csv')
print(f"Total questions: {len(df)}")
print(f"Questions by country:\n{df['country'].value_counts()}")