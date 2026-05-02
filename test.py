import warnings
warnings.filterwarnings("ignore")

from src.data_loader import load_event_log
from src.preprocessor import create_prefixes
from src.baseline import MostFrequentClassBaseline
from src.baseline import MeanRemainingTimeBaseline
from src.evaluate import accuracy, mae, evaluate_by_prefix_length

# 1. Загрузка данных
df = load_event_log("data/BPI_Challenge_2012.xes.gz")
print("=== Data Loader ===")
print("Shape:", df.shape)
print(df.head())
print("Columns:", df.columns.tolist())

# 2. Создание префиксов
prefixes = create_prefixes(df)
print("\n=== Preprocessor ===")
print("Shape:", prefixes.shape)
print(prefixes.head())
print("\nPrefix lengths distribution:")
print(prefixes['prefix_length'].value_counts().sort_index().head(10))
print("\nRemaining time sample:")
print(prefixes['remaining_time'].describe())

# 3. Most Frequent Class baseline
baseline_mfc = MostFrequentClassBaseline()
baseline_mfc.fit(prefixes)
predictions = baseline_mfc.predict(prefixes)
print("\n=== MFC Baseline ===")
print("Most frequent activity:", baseline_mfc.most_frequent)
print("Predictions sample:", predictions[:5])
print("Total predictions:", len(predictions))

# 4. Mean baseline для remaining time
baseline_mean = MeanRemainingTimeBaseline()
baseline_mean.fit(prefixes)
time_predictions = baseline_mean.predict(prefixes)

print("\n=== Mean Remaining Time Baseline ===")
print("Mean remaining time (seconds):", baseline_mean.mean_time)
print("Mean remaining time (days):", round(baseline_mean.mean_time / 86400, 2))
print("Predictions sample:", time_predictions[:3])

# 5. Metrics
print("\n=== Evaluate ===")

# Тест accuracy
acc = accuracy(prefixes['next_activity'].tolist(), predictions)
print("MFC Accuracy:", round(acc, 4))

# Тест MAE
mae_score = mae(prefixes['remaining_time'].tolist(), time_predictions)
print("Mean Baseline MAE (seconds):", round(mae_score, 2))
print("Mean Baseline MAE (days):", round(mae_score / 86400, 2))

# Тест evaluate_by_prefix_length
results_df = evaluate_by_prefix_length(prefixes, predictions, time_predictions)
print("\nResults by prefix length:")
print(results_df.head(10))