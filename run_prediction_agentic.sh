# export DEEPSEEK_TOKENIZER_PATH="xxx"
export OPENAI_API_KEY="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"
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"

# DATASET_PATH=feabench-data/FEA-Bench-v1.0-Oracle
DATASET_PATH=PGCodeLLM/FEA-Bench-v1.0-Standard
MODEL_NAME="qwen/qwen3-coder-30b-a3b-instruct"
RESULTS_ROOT_DIR=scripts/experiments/results_qwen3-coder-30b-a3b-instruct

# PROMPT_MODE=natural-detailed
PROMPT_MODE=problem_statement
python -m feabench.run_prediction \
    --dataset_name_or_path $DATASET_PATH \
    --model_type openai \
    --model_name_or_path $MODEL_NAME \
    --input_text $PROMPT_MODE \
    --output_dir $RESULTS_ROOT_DIR/$PROMPT_MODE \
    --num_proc 16 \
    --work_mode agentic \
    --agent_name iflow-cli

chmod 777 -R .
