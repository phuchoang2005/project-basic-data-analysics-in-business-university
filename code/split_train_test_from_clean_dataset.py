INPUT_FILE = "quantumn_clean.csv"
HAS_HEADER = True
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
TEST_SIZE = 0.3
RANDOM_STATE = 50

from pandas import read_csv
from sklearn.model_selection import train_test_split
if HAS_HEADER:
    df = read_csv(INPUT_FILE)
else:
    df = read_csv(INPUT_FILE, header=None)

print(f"[INFO] Đọc {len(df)} dòng từ {INPUT_FILE}")

# Chia 7/3
train_df, test_df = train_test_split(
    df, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
)

train_df.to_csv(TRAIN_FILE, index=False, header=HAS_HEADER)
test_df.to_csv(TEST_FILE, index=False, header=HAS_HEADER)
print(f"[OK] Train set: {len(train_df)} dòng -> {TRAIN_FILE}")
print(f"[OK] Test set: {len(test_df)} dòng -> {TEST_FILE}")