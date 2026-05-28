# Data format

Raw clinical recordings are not included.

The expected private preprocessed file has shape:

[context, trial, channel, time]

Contexts:
0 = Wait
1 = React
2 = Reach
3 = Return

Each context window is 400 ms after preprocessing. Sliding windows use:
T_in = 100
T_out = 20
stride = 20

Splits are performed at the trial level to prevent leakage between overlapping windows.

Normalization statistics are computed from training trials only and applied to validation/test trials.
