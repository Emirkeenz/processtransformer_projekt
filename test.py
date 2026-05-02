from src.data_loader import load_event_log
from src.preprocessor import create_prefixes

df = load_event_log("data/BPI_Challenge_2012.xes.gz")
prefixes = create_prefixes(df)

print("Shape:", prefixes.shape)
print(prefixes.head())
print("\nPrefix lengths distribution:")
print(prefixes['prefix_length'].value_counts().sort_index().head(10))
print("\nRemaining time sample:")
print(prefixes['remaining_time'].describe())