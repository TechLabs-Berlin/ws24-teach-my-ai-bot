import torch
import nltk
import os
from transformers import T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained("t5-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

current_dir = os.path.dirname(__file__)
model_path_a = os.path.join(current_dir, "..", "models", "adult_tuned2e.pth")
model_path_b = os.path.join(current_dir, "..", "models", "base4epoch.pth")

model_a = torch.load(model_path_a, map_location=torch.device("cpu"))
model_q = torch.load(model_path_b, map_location=torch.device("cpu"))


######   SEGMENT THE INPUT TEXT    ######

##   Changed chunker to no longer decode tokenized inputs

def chunker(text, tokenizer, min_tokens=140, max_tokens=200):

    sentences = nltk.tokenize.sent_tokenize(text)

    segments = []
    current_segment = []
    current_token_count = 0

    for sentence in sentences:
        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
        sentence_token_count = len(sentence_tokens)

        if current_token_count + sentence_token_count > max_tokens:
            segments.append(current_segment)
            current_segment = sentence_tokens
            current_token_count = sentence_token_count
        
        else:
            current_segment.extend(sentence_tokens)
            current_token_count += sentence_token_count

        if min_tokens <= current_token_count <= max_tokens:
            segments.append(current_segment)
            current_segment = []
            current_token_count = 0

    if current_segment:
        segments.append(current_segment)

    return segments


######   ANSWER GENERATOR   ######

##   Updated, no longer re-tokenized inputer from chunker

def gen_a(tokenized_text):

    model_a.eval()
    input_ids = torch.tensor([tokenized_text]).to(device)
    
    with torch.no_grad():
        outputs = model_a.generate(input_ids, max_length=50)
   
    generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_answer


######   HIGHLIGHTING SEGMENTS   ######


def highlight_answer(context, answer, tokenizer):

    context = tokenizer.decode(context, skip_special_tokens=True)
    highlighted_context = context.replace(answer, f"[ANSWER] {answer} [/ANSWER]")
    return highlighted_context


######   GENERATE QUESTION   #####

def gen_q(text):
    model_q.eval()
    input_text = f"generate question: {text} </s>"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model_q.generate(input_ids)

    generated_question = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_question