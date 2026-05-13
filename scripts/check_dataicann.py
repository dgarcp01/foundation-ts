import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from foundation_ts.data import load_dataicann, make_dataicann_windows

plt.ion()

df_icann = load_dataicann()

plt.figure("Exp 0")
plt.clf()
plt.plot(df_icann["is"].iloc[:400])

X = make_dataicann_windows(df_icann, window_size=400, stride=400, signal_columns=["ac","ax","ay","ir","is"])