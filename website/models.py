import torch
import nltk
import os
from transformers import T5Tokenizer, pipeline, AutoTokenizer
from sentence_transformers import SentenceTransformer, util

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


######   Answer Evaluation   #####

# Initialize the models
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
answer_assessment_model = pipeline("text-classification", model="Giyaseddin/distilroberta-base-finetuned-short-answer-assessment", return_all_scores=True)

def evaluate_answer(context, question, ref_answer, student_answer):
    # Initialize evaluation and feedback
    evaluation_result = {"evaluation": None, "feedback": None}

    # Direct Comparison
    if student_answer.lower().strip() == ref_answer.lower().strip():
        evaluation_result["evaluation"] = "correct"
        evaluation_result["feedback"] = "Your answer is exactly correct."
        return evaluation_result

    # Apply thresholds for semantic similarity
    if similarity > 0.8:
        evaluation_result["evaluation"] = "correct"
        evaluation_result["feedback"] = "Your understanding of the concept is on point."
        return evaluation_result
    elif similarity < 0.4:
        evaluation_result["evaluation"] = "incorrect"
        evaluation_result["feedback"] = "There seems to be a misunderstanding of the concept."
        return evaluation_result

    # DistilRoBERTa Evaluation for nuanced assessment
    body = " [SEP] ".join([context, question, ref_answer, student_answer])
    raw_results = answer_assessment_model([body])
    best_result = max(raw_results[0], key=lambda x: x['score'])
    distilroberta_label = int(best_result['label'][-1])

    if distilroberta_label == 0:  # Correct
        evaluation_result["evaluation"] = "correct"
        evaluation_result["feedback"] = "Your answer aligns well with the core concept."
    elif distilroberta_label == 1:  # Correct but Incomplete
        evaluation_result["evaluation"] = "not_exactly"
        evaluation_result["feedback"] = "You're on the right track, but some details are missing."
    elif distilroberta_label == 2:  # Contradictory
        evaluation_result["evaluation"] = "incorrect"
        evaluation_result["feedback"] = "Your answer seems to contradict the key concept."
    else:  # Incorrect
        evaluation_result["evaluation"] = "incorrect"
        evaluation_result["feedback"] = "The provided answer doesn't match the expected concept."

    return evaluation_result

######   Generate User Feedback   #####

# Initialize the summarization pipeline
summarizer_pipeline = pipeline("summarization", model="Oulaa/teachMy_sum")
def generate_user_feedback(evaluation_result, context):
    """
    Generates tailored user feedback by summarizing the context or feedback and context
    based on the evaluation of the user's answer.

    Parameters:
    - evaluation_result (dict): The result from the evaluate_answer function, containing 'evaluation' and 'feedback'.
    - context (str): The context related to the question and answer.

    Returns:
    - str: Tailored user feedback based on the summarization of the context or feedback with context.
    """

    separator = "\n\n"  # Double newline for clear separation

    # Tokenize the context to count the tokens
    context_tokens = tokenizer.tokenize(context)
    num_context_tokens = len(context_tokens)

    # Define a threshold for when the context is considered too long in terms of tokens
    token_threshold = 512  # Adjust based on model's capacity and performance

    # Determine the maximum length for summarization
    max_length_output = num_context_tokens  # Use the context token count unless it exceeds the threshold
    if num_context_tokens > token_threshold:
        max_length_output = int(num_context_tokens * 0.75)  # Summarize to 75% of the context token count

    # Summarize the context
    summarized_context = \
    summarizer_pipeline(context, max_length=max_length_output, min_length=int(max_length_output / 2), do_sample=False)[
        0]['summary_text']

    # Prepend the feedback to the summarized content with clear separation
    final_feedback = f"{evaluation_result['feedback']}{separator}{summarized_context}"

    return final_feedback
