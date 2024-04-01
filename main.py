import os
import PyPDF2
import fitz
import nltk
import logging
from collections import Counter
from nltk.corpus import stopwords
import string
from haystack.document_stores import InMemoryDocumentStore
from haystack.pipelines.standard_pipelines import TextIndexingPipeline
from haystack.nodes import BM25Retriever
from haystack.nodes import FARMReader
from haystack.pipelines import ExtractiveQAPipeline
from haystack.utils import print_answers 


logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

document_store = InMemoryDocumentStore(use_bm25=True)

nltk.download('stopwords') #where?

#Fernanda's part
def extract_text(file_path):
    # Using PyMuPDF to extract text
    text = ""
    with fitz.open(file_path) as pdf_file:
        for page_num in range(len(pdf_file)):
            page = pdf_file.load_page(page_num)
            text += page.get_text()
    return text

def extract_metadata(file_path):
    # Using PyMuPDF for extracting metadata
    with fitz.open(file_path) as pdf_file:
        metadata = pdf_file.metadata
        number_of_pages = len(pdf_file)
        return metadata, number_of_pages

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    # Remove common words
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text


#file writing part missing
#part to save the the extracted/preprocessed text to an output.txt file which goes to the document store
file_path = "wallflower.pdf"
text = extract_text(file_path)
metadata, number_of_pages = extract_metadata(file_path)

text = preprocess_text(text)

#write the text to output.txt
with open("output.txt", "w", encoding='utf-8') as text_file:
  text_file.write(text)

print("Text extracted, preprocessed and saved to output.txt!")

#splits the file to 9999 char lengts files
def split_file(filename, max_chars=9999):
  """Splits a long text file into parts with a maximum character limit and stores file paths.

  Args:
    filename: The name of the long text file to split.
    max_chars: The maximum number of characters allowed per part (default 9999).

  Returns:
    A list containing the paths to all the created output files.
  """

  # Open the file in read mode
  with open(filename, 'r') as f:
    # Read the entire content of the file
    data = f.read()

  # Initialize variables
  part_number = 1
  current_part = ""
  file_paths = []

  # Loop through the data character by character
  for char in data:
    # Add character to current part
    current_part += char

    # Check if current part exceeds limit
    if len(current_part) > max_chars:
      # Write current part to a file
      with open(f"output{part_number}.txt", "w") as f:
        f.write(current_part)

      # Add file path to list and reset variables
      file_paths.append(f"output{part_number}.txt")
      current_part = ""
      part_number += 1

  # Write the final part (if any) and add path
  if current_part:
    with open(f"output{part_number}.txt", "w") as f:
      f.write(current_part)
    file_paths.append(f"output{part_number}.txt")

  print(f"The long text file has been split into parts with a maximum of {max_chars} characters each. File paths stored in file_paths list.")
  return file_paths

# Replace 'your_long_file.txt' with the actual name of your long text file
file_paths = split_file('output.txt')

# Now you can access the list of file paths:
print(file_paths)
#paths to splitted text files 
paths = [path for path in file_paths]


indexing_pipeline = TextIndexingPipeline(document_store)
indexing_pipeline.run_batch(file_paths=paths)
retriever = BM25Retriever(document_store=document_store)
#My model, hosted on Hugging Face (just use the username / and the model name)
#piece of code to check if an gpu is available
reader = FARMReader(model_name_or_path="dusarpi/roberta-squad", use_gpu=True) #gpu?
pipe = ExtractiveQAPipeline(reader, retriever)

#asking a question
#function for running the queries
def run_query(questions):
  answers_all = []
  for question in questions:
    if pipe:  # Use Haystack pipeline if available
      prediction = pipe.run(query=question, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}})
      vmi = prediction['answers'][0].answer
      answers_all.append(vmi)  # Extract answers from prediction
    else:  # If no Haystack pipeline, provide guidance for alternative processing
      print(f"Please provide a Haystack pipeline or implement your own question processing logic for question: {question}")
  return answers_all

#calling the function with question(s) argument
questions = [] #fill with question(s)
run_query(questions)