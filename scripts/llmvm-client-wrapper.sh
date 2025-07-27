#!/bin/bash
source /home/llmvm/.bashrc
cd /home/llmvm/llmvm
conda activate llmvm

# Model selection
echo "Select a model:"
echo "1) claude-sonnet-4-20250514 (default)"
echo "2) gpt-4.1"
echo "3) o4-mini"
echo "4) o3"
echo "5) gemini-2.5-pro"
echo ""
read -p "Enter selection (1-5) or model name [default: 1]: " selection

# Default to claude-3-7-sonnet-latest if no input
if [ -z "$selection" ]; then
  selection="1"
fi

# Map selection to model name
case "$selection" in
1)
  model="claude-sonnet-4-20250514"
  executor="anthropic"
  ;;
2)
  model="gpt-4.1"
  executor="openai"
  ;;
3)
  model="o4-mini"
  executor="openai"
  ;;
4)
  model="o3"
  executor="openai"
  ;;
5)
  model="gemini-2.5-pro"
  executor="gemini"
  ;;
*)
  # Assume user entered a model name directly
  model="$selection"
  # Ask for executor when custom model is entered
  echo ""
  echo "Select executor for $model:"
  echo "1) anthropic"
  echo "2) openai"
  echo "3) gemini"
  read -p "Enter selection (1-3): " executor_selection

  case "$executor_selection" in
  1)
    executor="anthropic"
    ;;
  2)
    executor="openai"
    ;;
  3)
    executor="gemini"
    ;;
  *)
    echo "Invalid executor selection. Defaulting to openai."
    executor="openai"
    ;;
  esac
  ;;
esac

echo "Using model: $model"
echo "Using executor: $executor"

export LLMVM_FULL_PROCESSING="true"
export LLMVM_PROFILING="true"
export LLMVM_MODEL="$model"
export LLMVM_EXECUTOR="$executor"
exec python -m llmvm.client
