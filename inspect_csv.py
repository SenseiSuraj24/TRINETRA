import pandas as pd

path = r"C:\Users\SURAJ\Desktop\NEXJEM\CSV's\MachineLearningCVE\Monday-WorkingHours.pcap_ISCX.csv"
df = pd.read_csv(path, nrows=3)
print("=== COLUMNS ===")
for i, c in enumerate(df.columns):
    print(f"  [{i}] '{c}'")
print("\n=== DTYPES ===")
print(df.dtypes.to_string())

path2 = r"C:\Users\SURAJ\Desktop\NEXJEM\CSV's\MachineLearningCVE\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
df2 = pd.read_csv(path2, nrows=500)
print("\n=== LABELS ===")
print(df2.iloc[:, -1].unique())
print("\n=== FIRST ROW SAMPLE ===")
print(df.iloc[0].to_string())
