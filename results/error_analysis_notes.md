# Error analysis: B4 classical model

Model: word+character TF-IDF + calibrated LinearSVC  
Evaluation split: official test set  
Test size: 5,303  
Errors: 1,166  
Accuracy: 0.780  
Macro-F1: 0.781

## Confusion matrix

Rows are gold labels and columns are predictions. Label order is positive, neutral, negative.

```text
[[1481, 176,  98],
 [ 115, 1372, 302],
 [ 123, 352, 1284]]
```

The largest error directions are:

- Negative predicted as neutral: 352
- Neutral predicted as negative: 302
- Positive predicted as neutral: 176

This shows that the main difficulty is separating neutral or weakly expressed sentiment from negative sentiment.

## Successful examples

These are high-confidence correct predictions from the test set.

| Gold label | Text | Model confidence | Interpretation |
|---|---|---:|---|
| Positive | `masha allah allah yakara basira` | 0.999 | Clear religious praise or goodwill expression. |
| Positive | `masha allah abu yayi kyau fatan alkairi allah ya kara daukaka da basira and` | 0.999 | Contains praise and a positive good-wishes phrase. |
| Negative | `mtsww karyar banza` | 0.997 | Short insult or accusation of worthless lying. |
| Negative | `bbc shegiyar karya` | 0.996 | Strong negative wording directed at the subject. |
| Neutral | `saura nawa nima` | 0.963 | Short informational/question-like statement with limited sentiment. |
| Neutral | `hmn kudi ake rabawa ne` | 0.951 | Question or observation without a strong polarity cue. |

## Failed examples

These are high-confidence errors from the test set. The interpretations below are hypotheses for discussion, not additional ground-truth annotations.

| Gold | Prediction | Text | Confidence | Likely difficulty |
|---|---|---|---:|---|
| Positive | Negative | `ride on hajiya kin chi uwarsu walahi ba kare mai haushi` | 0.949 | Code-switching and context-dependent praise/taunting; isolated negative-looking words can mislead the model. |
| Positive | Negative | `da zaman banza ay gara aikin kishiya` | 0.943 | Ambiguous sarcasm or contrast; literal negative-looking wording may not match the intended sentiment. |
| Negative | Neutral | `wasan gaba a watsa mai kaya kawai` | 0.935 | Short context and implied criticism without a highly distinctive polarity token. |
| Negative | Positive | `fatan rashin nasara allah yasa abasu kaya` | 0.931 | Negation and a wish for failure; religious vocabulary can overlap with positive examples. |
| Negative | Positive | `allah ubangiji yasine mata` | 0.927 | Religious expression whose sentiment depends heavily on context. |
| Neutral | Negative | `nabi abincin kaji da gudu ba` | 0.907 | Very short, informal text with unclear context and possible negation. |

## Discussion

The model performs best on explicit lexical cues, especially common praise expressions and direct insults. Character n-grams help with spelling variation and informal Hausa text, but they cannot reliably resolve sarcasm, negation scope, code-switching, or context-dependent religious expressions. Neutral examples are especially difficult because they may be factual, rhetorical, or weakly evaluative. The error analysis supports retaining the classical model for deployment while documenting these limitations and considering contextual pretrained models as future work.

Generated prediction files:

- `error_analysis_correct_examples.csv`
- `error_analysis_error_examples.csv`
- `error_analysis_all_test_predictions.csv`
- `error_analysis_summary.json`
