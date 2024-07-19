import re
from transformers import StoppingCriteria, StoppingCriteriaList

def clean_text(text):
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Capitalize the first letter of sentences
    text = '. '.join(s.capitalize() for s in text.split('. '))
    return text

def get_prompt_template(prompt_type):
    templates = {
        "Story": "Write a {genre} story about {protagonist} in {setting}. {additional_info}",
        "Article": "Write a {style} article about {topic}. {additional_info}",
        "Poem": "Write a {form} poem about {theme}. {additional_info}"
    }
    return templates.get(prompt_type, "")
    from transformers import StoppingCriteria, StoppingCriteriaList

class RepetitionPenaltyLogitsProcessor(StoppingCriteria):
    def __init__(self, threshold=3):
        self.threshold = threshold
        self.generated_tokens = []

    def __call__(self, input_ids, scores):
        last_token = input_ids[0][-1].item()
        self.generated_tokens.append(last_token)
        
        if len(self.generated_tokens) >= self.threshold:
            recent_tokens = self.generated_tokens[-self.threshold:]
            if len(set(recent_tokens)) == 1:
                return True
        return False

stopping_criteria = StoppingCriteriaList([RepetitionPenaltyLogitsProcessor()])

def clean_output(text):
    import re
    text = re.sub(r'\b(.+?)\s+\1\b', r'\1', text)
    text = ' '.join(text.split())
    return text