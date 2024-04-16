<p align="center">
<img src="./assets/teachmylogo.png" width="250px"></p>
</p>

<br>

<h5 align="center">
  <a href="#data-science">Data Science</a>  |
  <a href="#deep-learning">Deep Learning</a> 
</h5>


# Data Science

# Deep Learning
First, we implemented a question-answering agent, which answers question(s) from the user, based on an uploaded pdf document. The uploaded pdf(s) serves as a knowledge base.
For this part of the project, RoBERTa language model, an extractive language model finetuned on the SQuaD dataset, was used. The SQuaD dataset is a common dataset containing question-answer pairs, where the answers are segments of text from Wikipedia articles. The extractive language model is different than the generative one. The major difference is in its capability to return information from a context, rather than generating new text. It seems less capable than its bigger brothers, in some scenarios when textual faithfulness is needed, they are the way to go.  As a rule of thumb, the more sensitive the application, the higher the need for textual faithfulness, and the lower tolerance for paraphrases.

The Quiz Generator initiative began with the selection of the T5 model, chosen for its suitability for sequence-to-sequence tasks, including question generation. The model was fine-tuned using the SQuAD and NewsQA datasets. Between the 'small' and 'base' versions of T5, the 'base' model demonstrated better results and was subsequently prioritized for further training.

The primary challenge encountered was that, although the model generated grammatically correct questions, these questions often lacked relevance to the given context. To overcome this, a two-model approach was adopted. A first T5 model was tasked with identifying plausible answers within the text, and a second T5 model then formulated questions based on these identified answers. This methodology proved to be the most effective in producing contextually relevant and answerable questions.
