import json
from importlib import resources
from typing import Optional


class TokenPriceCalculator():
    def __init__(
        self,
        price_file: str = 'model_prices_and_context_window.json',
    ):
        self.price_file = resources.files('llmvm.common') / price_file
        self.prices = self.__load_prices()

    def __load_prices(self):
        with open(self.price_file, 'r') as f:  # type: ignore
            json_prices = json.load(f)
            return json_prices

    def prompt_price(
        self,
        model: str,
        executor: Optional[str] = None
    ):
        if model in self.prices:
            return self.prices[model]['output_cost_per_token']
        elif executor and f'{executor}/{model}' in self.prices:
            return self.prices[f'{executor}/{model}']['output_cost_per_token']
        else:
            return 0.0

    def sample_price(
        self,
        model: str,
        executor: Optional[str] = None
    ):
        if model in self.prices:
            return self.prices[model]['output_cost_per_token']
        elif executor and f'{executor}/{model}' in self.prices:
            return self.prices[f'{executor}/{model}']['output_cost_per_token']
        else:
            return 0.0
