import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import LogitsProcessorList, PhrasalConstraint
from utils import stopping_criteria
from tqdm import tqdm

from utils import clean_output


class CustomConstraintLogitsProcessor:
    def __init__(self, tokenizer, phrases):
        self.tokenizer = tokenizer
        self.phrase_ids = [tokenizer.encode(phrase, add_special_tokens=False) for phrase in phrases]

    def __call__(self, input_ids, scores):
        for phrase_ids in self.phrase_ids:
            if phrase_ids[0] in scores[0]:
                scores[0, phrase_ids[0]] += 100  # Increase probability of the first token
        return scores

class TextGenerator:
    def __init__(self):
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate_text(self, prompt, max_length=100):
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            no_repeat_ngram_size=3,
            do_sample=True
        )
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text  

    # def generate_text(self, prompt, max_length=100, temperature=0.7, top_k=50, repetition_penalty=1.2):
    #     model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
    #     tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    #     input_ids = tokenizer.encode(prompt, return_tensors="pt")
    #     input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
    #     stopping_criteria = StoppingCriteriaList([self.RepetitionCriteria(self.tokenizer)])
    #     custom_constraint = CustomConstraintLogitsProcessor(self.tokenizer, [" Facebook", " Twitter"])
    #     logits_processor = LogitsProcessorList([custom_constraint])

        generated = self.model.generate(
            input_ids,
            max_length=300,
            num_return_sequences=1,
            temperature=0.8,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            #input_ids,
            #max_length=max_length,
            #temperature=temperature,
            #top_k=top_k,
            #no_repeat_ngram_size=3,
            repetition_penalty=repetition_penalty,
            #num_return_sequences=1,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor
        )

        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        cleaned_text = clean_output(generated_text)
        return cleaned_text
        generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return self.remove_duplicates(generated_text)

    def remove_duplicates(self, text):
        sentences = text.split('.')
        unique_sentences = list(dict.fromkeys(sentences))
        return '. '.join(unique_sentences).strip()

    class RepetitionCriteria(StoppingCriteria):
        def __init__(self, tokenizer, window_size=5, threshold=0.8):
            self.tokenizer = tokenizer
            self.window_size = window_size
            self.threshold = threshold

        def __call__(self, input_ids, scores, **kwargs):
            if len(input_ids[0]) < self.window_size:
                return False
            last_window = input_ids[0][-self.window_size:]
            last_window_text = self.tokenizer.decode(last_window)
            previous_text = self.tokenizer.decode(input_ids[0][:-self.window_size])
            return last_window_text in previous_text

    def generate_stream(self, prompt, max_length=None, temperature=None, top_k=None, top_p=None):
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        max_length = max_length or Config.MAX_LENGTH
        temperature = temperature or Config.TEMPERATURE
        top_k = top_k or Config.TOP_K
        top_p = top_p or Config.TOP_P

        for i in tqdm(range(inputs.shape[1], max_length)):
            with torch.no_grad():
                outputs = self.model(inputs)
            next_token_logits = outputs.logits[:, -1, :]

            # Apply temperature
            next_token_logits = next_token_logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                top_k = min(top_k, next_token_logits.size(-1))  # Safety check
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = -float('Inf')

            # Sample from the filtered distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break  # Stop generation if EOS token is generated

            inputs = torch.cat([inputs, next_token], dim=-1)

            yield self.tokenizer.decode(inputs[0], skip_special_tokens=True)

        return self.tokenizer.decode(inputs[0], skip_special_tokens=True)

class Config:
    MODEL_NAME = "gpt2-medium"
    MAX_LENGTH = 200
    TEMPERATURE = 0.7
    TOP_K = 50
    TOP_P = 0.95
