Scores a predicted JSON against a ground truth using biomedical embeddings. Outputs a markdown report with three comparison tables (structure, values, leaf).
pip install torch transformers numpy scipy
python embed_eval.py --gt ground_truth.json --pred prediction.json
Key flags: --preset (medcpt/biobert/biogpt), --report-out, --no-list-matching, --batch-size.
All scores are 0–1, higher is better.
