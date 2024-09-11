#!/bin/zsh

source "$HOME/.zshrc"
export PATH="$PATH:{os.environ.get('PATH', '')}"
eval "$@"
