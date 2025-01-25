#!/bin/zsh

original_dir=$(pwd)

source "$HOME/.zshrc"
export PATH="$PATH:{os.environ.get('PATH', '')}"

cd $original_dir

eval "$@"

