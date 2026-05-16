# SMC Engine Research Notes

## Observation 1
Range detection works during compression phases.

## Observation 2
Bias correctly flips bearish after displacement candle.

## Observation 3
Bearish setup remains False because:
- range condition disappears during expansion phase

## Research Insight
Current architecture lacks state persistence.

Need:
- temporal memory
- recent range tracking
- multi-step setup lifecycle