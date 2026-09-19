# Cohere ASR service

Copy this directory, replace `Recognizer` in `model/model.py`, and change
`MODEL_ID`. Do not change the message shapes: both lanes must emit identical
Protocol 1 frames, which is what makes the hour-5 promotion a config flip.

Confidence must come from `contract/confidence.py`, never from NeMo's packaged
confidence utility, which raises `IndexError` on TDT models.
