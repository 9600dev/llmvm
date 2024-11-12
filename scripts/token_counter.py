import asyncio
import os
import pickle
import random
from typing import Dict, Generator, List, Tuple, cast

import click
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from anthropic import Anthropic
from anthropic.types.message import Message as AnthropicMessage
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from llmvm.client.client import execute_llm_call, execute_llm_call_direct, llm
from llmvm.common.anthropic_executor import AnthropicExecutor
from llmvm.common.objects import (Assistant, Content, Message, MessageModel,
                                  SessionThreadModel, User)


class TokenizerModel(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout):
        super(TokenizerModel, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.conv1 = nn.Conv1d(embedding_size, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        embedded = embedded.permute(0, 2, 1)  # Reshape for conv1d
        conv1_out = nn.functional.relu(self.conv1(embedded))
        conv2_out = nn.functional.relu(self.conv2(conv1_out))
        pooled_out = self.pool(conv2_out)
        pooled_out = pooled_out.permute(0, 2, 1)  # Reshape for LSTM
        packed_embedded = nn.utils.rnn.pack_padded_sequence(pooled_out, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed_embedded)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        avg_pool_out = torch.mean(lstm_out, dim=1)
        fc1_out = nn.functional.relu(self.fc1(avg_pool_out))
        fc1_out = self.dropout(fc1_out)
        out = self.fc2(fc1_out)
        return out


def train_pytorch_tokenizer(test_lines, vocab, epochs=200, batch_size=128, learning_rate=0.001):
    # Prepare the dataset
    data = []
    for count, text in test_lines:
        # Convert characters to indices
        indices = [vocab[char] for char in text]
        data.append((count, indices))

    # Sort the data by length (longest to shortest)
    data.sort(key=lambda x: len(x[1]), reverse=True)

    # Separate counts and text indices
    counts = [count for count, _ in data]
    text_indices = [indices for _, indices in data]

    # Convert data to PyTorch tensors
    counts_tensor = torch.tensor(counts, dtype=torch.float32).unsqueeze(1)
    text_tensor = nn.utils.rnn.pad_sequence([torch.tensor(indices, dtype=torch.long) for indices in text_indices], batch_first=True)
    lengths_tensor = torch.tensor([len(indices) for indices in text_indices], dtype=torch.long)

    # Move tensors to GPU (except lengths_tensor)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    counts_tensor = counts_tensor.to(device)
    text_tensor = text_tensor.to(device)

    # Create a dataset and data loader
    dataset = TensorDataset(text_tensor, counts_tensor, lengths_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model
    input_size = len(vocab)
    embedding_size = 128
    hidden_size = 256
    num_layers = 2
    dropout = 0.2
    model = TokenizerModel(input_size, embedding_size, hidden_size, num_layers, dropout).to(device)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_text, batch_counts, batch_lengths in dataloader:
            # Move batch_lengths to CPU
            batch_lengths = batch_lengths.cpu()

            # Forward pass
            outputs = model(batch_text, batch_lengths)
            loss = criterion(outputs, batch_counts)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        # Print the average loss for every epoch
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    return model


def get_language(filename: str) -> Generator:
    with open(filename, 'r') as file:
        for line in file:
            yield line.split()[1]


def generate_sentences(language: str) -> List[str]:
    prompt = f"""
    I want you to generate 50 sentences that each have nice poetry in the {language} language.
    In those sentences, be very liberal with punctuation and symbols used in the {language} language.

    You can emit between 2 and 200 words in each sentence. Do not repeat sentences and try not to reuse words.

    Only emit in the {language} language. You must generate 50 sentences.
    Use line breaks to separate the sentences.
    """

    result = asyncio.run(execute_llm_call(
        api_endpoint='',
        id=-1,
        message=User(Content(prompt)),
        executor='anthropic',
        model='claude-3-haiku-20240307',
        mode='direct',
    ))
    return MessageModel.to_message(result.messages[-1]).message.get_str().splitlines()


def generate_test_set(language: str, filename_output: str):
    sentences = generate_sentences(language)
    client = Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])

    with open(filename_output, 'a') as file:
        for line in [s for s in sentences if len(s) > 0]:
            # I want to for over the current line and a split line by space that is randomly chosen
            # find all indexes for ' '
            # then randomly choose a number between 0 and len(indexes)
            indexes = [i for i, c in enumerate(line) if c == ' ']
            sub_lines = [line]
            if len(indexes) > 0:
                random_index = random.choice(indexes)
                sub_lines.append(line[:random_index])
                sub_lines.append(line[random_index:])

            for sub_line in sub_lines:
                if len(sub_line) == 0 or sub_line.isspace():
                    continue
                try:
                    message = client.messages.create(
                        max_tokens=10,
                        messages=[
                            {
                                "role": "user",
                                "content": sub_line
                            }
                        ],
                        model='claude-3-haiku-20240307',
                    )
                    token_count = message.usage.input_tokens
                    file.write(f"{token_count},{sub_line}\n")
                    print(f"Token count {token_count}: {sub_line}")
                    file.flush()
                except Exception as e:
                    print(f"Error: {e}")

def tokenize_test(count: int, text: str, x: bool = False) -> Dict:
    char_count = len(text)
    word_count = len(text.split())
    avg_word_length = char_count / word_count
    spaces = text.count(' ')
    special_char_count = sum([1 for c in text if not c.isalnum()])
    # how many contiguous chunks of alpha characters more than 1
    contiguous_alpha = sum(1 for i in range(len(text)) if text[i].isalpha() and (i == 0 or not text[i-1].isalpha()))
    contiguous_spaces = sum(1 for i in range(len(text)) if text[i].isspace() and (i == 0 or not text[i-1].isspace()))

    result = {
        'token_count': count,
        'char_count': char_count,
        'word_count': word_count,
        'avg_word_length': avg_word_length,
        'special_char_count': special_char_count,
        'spaces': spaces,
        'contiguous_alpha': contiguous_alpha,
        'contiguous_spaces': contiguous_spaces,
    }
    if x:
        # unset the token count
        result.pop('token_count')
    return result

def build_model(test_directory):
    test_lines = []
    for filename in os.listdir(test_directory):
        with open(test_directory + '/' + filename, 'r') as file:
            for line in file.readlines():
                count = int(line.split(',')[0])
                text = ''.join(line.split(',')[1:])
                test_lines.append((count, text))

    tests = []
    for test in test_lines:
        count, text = test
        tests.append(tokenize_test(count, text))

    print(f'Number of tests: {len(tests)}')
    df = pd.DataFrame(tests)
    X = df[['char_count', 'word_count', 'avg_word_length', 'special_char_count', 'spaces', 'contiguous_alpha', 'contiguous_spaces']]
    y = df['token_count']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    # model = LinearRegression()
    # model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2:.2f}")

    print("Saving model")
    with open('tokenizer_model.pkl', 'wb') as file:
        pickle.dump(model, file)


def build_pytorch_model(test_directory):
    def build_vocab(test_lines):
        # Create a set of unique characters
        chars = set()
        for _, text in test_lines:
            chars.update(text)

        # Create a dictionary mapping characters to indices
        vocab = {char: i for i, char in enumerate(chars)}
        return vocab

    test_lines = []
    for filename in os.listdir(test_directory):
        with open(test_directory + '/' + filename, 'r') as file:
            for line in file.readlines():
                count = int(line.split(',')[0])
                text = ''.join(line.split(',')[1:])
                test_lines.append((count, text))
    print('building vocab')
    vocab = build_vocab(test_lines)
    print('calling training method')
    model = train_pytorch_tokenizer(test_lines, vocab)


def tokenize_string(text: str) -> int:
    with open('tokenizer_model.pkl', 'rb') as file:
        model = pickle.load(file)
        test = tokenize_test(0, text, True)
        print(test)
        df = pd.DataFrame([test])
        return int(model.predict(df)[0])


def run_test(text: str):
    client = Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
    message = client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": text
            }
        ],
        model='claude-3-haiku-20240307',
    )
    print(f'Prediction: {tokenize_string(text)}')
    print(message.content[0].text)
    print('Actual:"')
    print(f'  Input token count: {message.usage.input_tokens}')
    print(f'  Output token count: {message.usage.output_tokens}')

@click.command()
@click.option('--tokenize', required=False, help='Tokenize a string')
@click.option('--test', required=False, help='test a string against the actual model')
@click.option('--model', is_flag=True, required=True, default=False, help='Build the model')
@click.option('--filename', required=False, help='Filename that contains languages, or model output file')
@click.option('--directory', required=False, help='Directory to save test sets')
def cli(tokenize, test, model, filename: str, directory: str):
    if test:
        run_test(test)
        return

    if model:
        # build_model(directory)
        build_pytorch_model(directory)
        return

    if tokenize:
        tokens = tokenize_string(tokenize)
        print(f"Token count: {tokens}")
        return

    languages = list(get_language(filename))
    random.shuffle(languages)
    for language in languages:
        print(f"Generating test set for {language}")
        filename_output = f"{directory}/{language}.txt"
        generate_test_set(language, filename_output)


if __name__ == '__main__':
    cli()
