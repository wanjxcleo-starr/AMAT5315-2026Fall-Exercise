# Boltzmann energy plot

From the repository root, with Matplotlib installed, draw the figure with:

```sh
python3 week3/evidence/draw_boltzmann.py
```

The script reads `week3/runs/T3.0/series.jsonl` and `week3/runs/T3.1/series.jsonl` and writes `week3/evidence/boltzmann.png`. It uses common 40-unit bins for `4096 * E`. The lower panel plots the log ratio of normalized bin counts where both counts are at least five. Its comparison line has slope `1/3.0 - 1/3.1`, with only the vertical offset fitted to the plotted bins.
