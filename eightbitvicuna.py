from guidance.llms._transformers import Transformers
from transformers import AutoModelForCausalLM, LlamaTokenizer


class VicunaEightBit(Transformers):
    def __init__(self, model, tokenizer=None, device_map=None, **kwargs):
        if isinstance(model, str):
            if tokenizer is None:
                tokenizer = LlamaTokenizer.from_pretrained(model, device_map=device_map, load_in_8bit=True)

            model = AutoModelForCausalLM.from_pretrained(model, device_map=device_map, load_in_8bit=True)
        super().__init__(model=model, tokenizer=tokenizer, device_map=device_map, **kwargs)

    default_system_prompt = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."""

    @staticmethod
    def role_start(role):
        if role == 'user':
            return 'USER: '
        elif role == 'assistant':
            return 'ASSISTANT: '
        else:
            return ''

    @staticmethod
    def role_end(role):
        if role == 'user':
            return ''
        elif role == 'assistant':
            return '</s>'
        else:
            return ''
