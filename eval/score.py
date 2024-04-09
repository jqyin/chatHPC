import pandas as pd
from evaltool_groundtruth import *
from evaltool_nli import *
import sys,os

# Check if the correct number of arguments is provided
if len(sys.argv) != 2:
    print("Usage: python main.py <jsonl_file>")
    sys.exit(1)

jl_file = sys.argv[1]
df = pd.read_json(jl_file, lines=True)

x = df['chosen'].tolist()
y = df['chathpc1'].tolist()

df = calc_rougescore(y, x, df)
df = calc_bertscore(y, x, df)
df = calc_bartscore(y, x, df)
df = calc_nli_score(df, 5)

basename = os.path.splitext(os.path.basename(jl_file))[0]
df.to_csv(f"{basename}-scores.csv", index=False)

