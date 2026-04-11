1. Export what you need

```
export API_KEY=Your api key
export BASE_URL=Your base url
```

For hle,

2. Run this command

```
python evaluate_hle_official.py --input_fp your_input_folder --model_path your_qwen_model_path
```

For other benchmarks,

2. Run this command

```
python evaluate_deepsearch_official.py --input_folder your_input_folder --dataset your_evaluated_dataset --debug

python evaluate_deepsearch_official.py --input_folder your_input_folder --dataset your_evaluated_dataset
```
