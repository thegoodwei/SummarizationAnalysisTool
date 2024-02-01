import torch
from scipy.spatial.distance import cosine
from transformers import BertModel, BertTokenizer, BartForConditionalGeneration, BartTokenizer, BertForSequenceClassification, BertForNextSentencePrediction
import pysrt
import collections
import numpy as np
import textwrap
from scipy.stats import chi2_contingency
import spacy
import gc

nlp = spacy.load("en_core_web_lg")
BERT_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
BERT_embedding_model = BertModel.from_pretrained('bert-base-uncased')
BERT_embedding_model.config.pad_token_id
BART_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
BART_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
BERT_nsp_model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
BERT_nsp_model.config.pad_token_id

def evaluate_summarization_bias(content_title, research_question, codebook, summarization_ratio=1/10):
    # Content can be either Youtube video id or .srt file path
    if ".srt" in content_title:
        transcript=load_and_preprocess_srt(content_title)
    else:
        srt_dict = YouTubeTranscriptApi.get_transcript(content_title)
        transcript = ' '.join(sub['text'] for sub in srt_dict)
    full_codebook_scores = {}
    abstract_summary = {}
    abstract_summary_scores = {}
    extractive_summary = {}  
    extractive_summary_scores = {}  
    results = ""
    results += "Evaluating transcript from youtube video id:" + content_title 
    total_tokens = len(BART_tokenizer.tokenize(transcript))
    max_tokens = total_tokens*summarization_ratio
    results += f"\ntotal tokens: {total_tokens}\n Summarized to: <{max_tokens}"
    results += f"\n\n To classify, each BERT sentence embedding compares to a research question and category to predict most likely.\n Research Question:\n {research_question}:"
    results += str(codebook) + "\n\n"
    print(results)
    abstract_summary[content_title] = summarize_text(transcript, max_tokens=max_tokens)
    print("Abstract summary:")
    print(abstract_summary[content_title])
#   Align the summary with primary-source quotes, potentially reducing measurable bias
    extractive_summary[content_title] = create_grounded_summary(transcript, abstract_summary[content_title])

    print("Grounded summary:")
    print(extractive_summary[content_title])
    print("Classifying:")
#   Score every sentence in each summary and in the current text against the codebook
    full_codebook_scores[content_title] = classify_sentences(transcript, codebook, research_question)
    abstract_summary_scores[content_title] = classify_sentences(abstract_summary[content_title], codebook, research_question=research_question)
    extractive_summary_scores[content_title] = classify_sentences(extractive_summary[content_title], codebook, research_question=research_question)

#   Compare the distribution of sentence categories applied in the summary to the original source distribution
    results += compare_distributions(full_codebook_scores[content_title], abstract_summary_scores[content_title])
    results += f'\n' #'change in distribution comparing for original text to abstract summary versus original text to representative sample of quotes:\n\n'
    results += compare_distributions(full_codebook_scores[content_title], extractive_summary_scores[content_title], summary_type="Quoted Sample")

    results += "\n\n\n" + generate_chi_square_heatmap(full_codebook_scores[content_title], abstract_summary_scores[content_title], name=content_title)
    categories = list(set(full_codebook_scores[content_title].keys()).union(abstract_summary_scores[content_title].keys(), extractive_summary_scores[content_title].keys()))
    results +="\n\n\n" + generate_line_graph(full_codebook_scores[content_title], abstract_summary_scores[content_title], extractive_summary_scores[content_title], categories, name=content_title)
    print("\n\n\nRESULTS:\n")
    print(results)
    return results

def generate_bart_summary(chunk, max_length=250):
    """
    Summarize the input text using the BART model.

    Args:
    text (str): The input text to summarize.
    model_name (str): The BART model name.
    max_tokens (int): The maximum length of the summary.

    Returns:
    str: The summarized text.
    """
#    inputs = BART_tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    inputs = BART_tokenizer(chunk, return_tensors='pt', truncation=True, padding="max_length")

    summary_ids = BART_model.generate(inputs.input_ids, num_beams=4, max_length=max_length, min_length=0, early_stopping=True)
    summary = BART_tokenizer.decode(summary_ids[0], skip_special_tokens=True) 
    return summary

def summarize_text(input_text, max_tokens=1000):
    """
    Summarize the input text using the BART model. Ensure that no content is truncated and 
    the final summary does not exceed max tokens.

    Args:
    input_text (str): The input text to summarize.
    max_tokens (int): The maximum length of the summary.

    Returns:
    str: The summarized text.
    """
    # Initialize conditions
    num_output_summary_tokens = max_tokens + 1  # Initialize to enter the while loop
    tokens_per_chunk = 512  # 512 Determine the initial optimal chunk size

    current_text = input_text  # Initialize current_text to be the input_text

    while num_output_summary_tokens > max_tokens:
        print(num_output_summary_tokens)
        # We chunk the current text (initially the input text, later could be the summary of the input text)
        chunks = textwrap.wrap(current_text, width=tokens_per_chunk)

        # Summarize each chunk
        chunk_summaries = [generate_bart_summary(chunk) for chunk in chunks if len(chunk)>5]
        gc.collect()
        # Combine the chunk summaries
        current_text = ' '.join(chunk_summaries).strip()

        # Update the number of tokens in the current text (which is the total summary at this point)
        num_output_summary_tokens = len(BART_tokenizer.tokenize(current_text))
        # If the total summary length is within the limit, we break the loop, else we continue
        # The already summarized text (current_text) will be re-summarized in the next iteration if necessary
    return current_text

from sklearn.neighbors import NearestNeighbors

def create_grounded_summary(original_text, summarized_text):
    """
    Ground each sentence of the summarized text to the closest original subtitle.
    
    Args:
    original_text (str): The original text.
    summarized_text (str): The summarized text.
    
    Returns:
    List[str]: A list of ground-truth normalized sentences.
    """
    # Sentence Tokenization
    original_sentences =  [sent.text for sent in nlp(original_text).sents]
    summarized_sentences =  [sent.text for sent in nlp(summarized_text).sents]
    # Embedding Generation
    original_embeddings = [get_embedding(sent) for sent in original_sentences]
    summarized_embeddings = [ get_embedding(sent) for sent in summarized_sentences]
    
    gc.collect()
    # Nearest Neighbor Search
    nbrs = NearestNeighbors(n_neighbors=1, metric='cosine').fit(original_embeddings)
    
    grounded_sentences = []
    available_indices = set(range(len(original_sentences)))  # Set of indices of available sentences

    for summarized_embedding in summarized_embeddings:
        # Fit the model only with available sentences
        available_embeddings = [original_embeddings[i] for i in available_indices]
        nbrs.fit(available_embeddings)
        
        # Find the nearest neighbor among available sentences
        distances, indices = nbrs.kneighbors([summarized_embedding])
        closest_index = indices[0][0]
        
        # Find the actual index in the original list of sentences
        actual_index = list(available_indices)[closest_index]
        
        # Add the sentence to the grounded_sentences
        grounded_sentences.append(original_sentences[actual_index])
        
        # Remove the index from available indices
        available_indices.remove(actual_index)

    return ". ".join(grounded_sentences).strip() + "."

def get_embedding(text):
    inputs = BERT_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = BERT_embedding_model(**inputs)
    # Use the pooled output for sentence-level representation
    return outputs.pooler_output.squeeze().detach().numpy()

    
def classify_sentences(text, codebook=["categories","activities","concepts"], research_question="The statement favors"):
    sentences =  [sent.text for sent in nlp(text).sents]
    sentences = [sent for sent in sentences if sent!="" and sent!=" "]
    categorized_sentences = {}#{phrase for phrase in codebook}  # Initialize scores 
    sentence_embeddings = [get_embedding(sentence) for sentence in sentences]

    for sentence, sentence_embedding in zip(sentences, sentence_embeddings):
        sentence_nsp_scores = {}
        for category in codebook:
             #Prepare text pair for NSP
            hypothesis = f"| {research_question} {category}"
            inputs = BERT_tokenizer.encode_plus(sentence, hypothesis, return_tensors='pt', max_length=512, truncation=True)
            
            # Get NSP score
            nsp_outputs = BERT_nsp_model(**inputs)
            
            
            nsp_score = nsp_outputs.logits[:,0].detach().numpy()[0]  # Index 0 for 'isNext' label
            sentence_nsp_scores[category] = nsp_score

        most_favored_category = max(sentence_nsp_scores, key=sentence_nsp_scores.get)
        categorized_sentences[sentence]= most_favored_category

    category_counts = dict(collections.Counter(category for sentence, category in categorized_sentences.items()))
    sorted_category_counts = dict(sorted(category_counts.items(), key=lambda item: item[1], reverse=True))

    return calculate_percentages(sorted_category_counts)


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency

def calculate_percentages(counts):
    total = sum(counts.values())
    percentages = {category: (count / total) * 100 for category, count in counts.items()}
    return percentages

def compare_distributions(original, summarized, summary_type="Summarized"):
    category_diffs=""
    for category in original.keys():
        orig_pct = original.get(category, 0)
        summ_pct = summarized.get(category, 0)
        category_diffs+= str(f"Category '{category}': Original = {orig_pct:.2f}%, {summary_type} = {summ_pct:.2f}%")
        category_diffs+=f"\nPercentage difference {(summ_pct - orig_pct):.2f}%\n"
    results =   str(category_diffs)
    results += "\n" + perform_chi_square_test(original, summarized)
    return results

def perform_chi_square_test(original_counts, summarized_counts):
    # Ensure both dictionaries have the same keys
    all_categories = set(original_counts.keys()).union(set(summarized_counts.keys()))

    # Fill in missing values with 0
    original_data = [original_counts.get(category, 0) for category in all_categories]
    summarized_data = [summarized_counts.get(category, 0) for category in all_categories]

    # Create a contingency table
    data = [original_data, summarized_data]

    # Perform the chi-square test
    chi2, p, dof, expected = chi2_contingency(data)
    results = ""
    results += f"Chi-square Statistic: {chi2}\n"
    results += f"Degrees of Freedom: {dof}\n"
    results += f"P-value: {p}\n"
    results += f"Expected Frequencies:\n"
    results += str(expected) +"\n"
    
    # Interpret p-value
    alpha = 0.05  # significance level
    if p < alpha:

        results += str("Significant differences exist between the distributions (reject H0)\n")
    else:
        results += str("No significant differences exist between the distributions (fail to reject H0)\n")

    return results #chi2, p, dof, expected

def generate_chi_square_heatmap(original_counts, summarized_counts, name=""):
    """
    Generate a heatmap based on the chi-square test results for each category.
    
    Args:
    original_counts (dict): Category counts for the original text.
    summarized_counts (dict): Category counts for the summarized text.
    
    Returns:
    None
    """
    # Ensure both dictionaries have the same keys
    all_categories = set(original_counts.keys()).union(set(summarized_counts.keys()))

    # Fill in missing values with 0
    original_data = [original_counts.get(category, 0) for category in all_categories]
    summarized_data = [summarized_counts.get(category, 0) for category in all_categories]

    # Create a contingency table
    data = [original_data, summarized_data]

    # Perform the chi-square test
    chi2, p, dof, expected = chi2_contingency(data)

    # Create a matrix for the heatmap
    matrix = np.array([original_data, summarized_data, expected[0], expected[1], [chi2]*len(all_categories), [p]*len(all_categories)])
    
    # Create a heatmap
    sns.set(style="white")
    plt.figure(figsize=(10, 6))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap='YlGnBu', xticklabels=list(all_categories), yticklabels=["Original", "Summarized", "Expected Original", "Expected Summarized", "Chi-Square", "P-Value"])
    plt.title('Chi-Square Test Heatmap')
    plt.yticks(rotation=0)  # Rotate labels
    plt.xticks(rotation=45, ha='right')  # Rotate labels
    plt.tight_layout()
    # Save or display the heatmap
    name=name+'_chi_square_heatmap.png' 
    plt.savefig(name, dpi=300)
    plt.show()
    return name

import matplotlib.pyplot as plt

def generate_line_graph(full_codebook_scores, abstract_summary_scores, extractive_summary_scores, categories, name=""):
    """
    Generate a line graph showing the average change in distribution for each category.

    Args:
    full_codebook_scores (dict): Category percentages for the original text.
    abstract_summary_scores (dict): Category percentages for the abstract summary.
    extractive_summary_scores (dict): Category percentages for the extractive summary.
    categories (list): List of all categories.

    Returns:
    None
    """
    # Initialize lists to store the average percentages
    data = []
    abstract_differences = []
    extractive_differences = []


    # Collect the data for each category
    for category in categories:
        original_val = full_codebook_scores.get(category, 0)
        abstract_val = abstract_summary_scores.get(category, 0)
        extractive_val = extractive_summary_scores.get(category, 0)
        # Calculate the absolute percentage differences
        abstract_differences.append(abs(abstract_val - original_val))
        extractive_differences.append(abs(extractive_val - original_val))
        abstract_change = abstract_val - original_val
        extractive_change = extractive_val - original_val
        data.append([0, abstract_change, extractive_change])  # Original text starts at 0

    # Calculate the average absolute percentage change
    avg_abstract_change = np.mean(abstract_differences)
    avg_extractive_change = np.mean(extractive_differences)

    # Create the line graph
    plt.figure(figsize=(10, 6))
    text_types = ['Original Text', 'BART Abstract Summary', 'K-NN Extracted Quotes']
    
    for i, category_data in enumerate(data):
        plt.plot(text_types, category_data, label=categories[i], marker='o')
    # Add the average change lines
    # Plot the average points
    # Plot the average points
    #avg_data = [0, avg_abstract_change, avg_extractive_change]
    #plt.scatter(text_types, avg_data, color='black', zorder=5)
    plt.scatter(text_types[1:], [avg_abstract_change, avg_extractive_change], color='black', zorder=5)

    plt.text(text_types[1], avg_abstract_change, f'  Avg: {avg_abstract_change:.2f}%', verticalalignment='bottom', color='black')
    plt.text(text_types[2], avg_extractive_change, f'  Avg: {avg_extractive_change:.2f}%', verticalalignment='bottom', color='black')
    
    # Customize the graph
    plt.title('Change in Category Distributions Across Summary Types')
  #  plt.xlabel('Text Type')
    plt.ylabel('Percentage Difference')
    plt.xticks(rotation=45, ha='right')  # Rotate text type names for better visibility
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add grid for better readability
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place the legend outside the plot
    
    # Save the graph as an image file
    name = name + "_line_graph.png"
    plt.savefig(name, format='png', dpi=300, bbox_inches='tight')
    
    # Display the graph
    plt.show()

    return name

from youtube_transcript_api import YouTubeTranscriptApi
def load_and_preprocess_srt(file_path):
    """
    Load and preprocess subtitles from an .srt file.
    
    Args:
    file_path (str): The path to the .srt file.
    
    Returns:
    str: A preprocessed string containing the text from the .srt file.
    """
    try:
        subs = pysrt.open(file_path, encoding='iso-8859-1')
        texts = [' '.join(sub.text_without_tags.split()) for sub in subs]
        preprocessed_text = ' '.join(texts)
        return preprocessed_text
    except Exception as e:
        print(f"Error loading or processing file {file_path}: {e}")
        return ""

if __name__ == '__main__':
    research_question = "The prior statement politically favors "
    video_id="-dmXJ99oC4k" #GOP Primary
    categories=["Republicans", "Democrats", "Independants", "War", "Peace"]
    out = evaluate_summarization_bias(content_title=video_id, research_question=research_question,codebook=categories)
    video_id= "2a7CDKqWcZ0" # Ukraine interview
    categories=["War", "Peace", "Ukraine", "Russia"]
    out = evaluate_summarization_bias(content_title=video_id, research_question=research_question,codebook=categories)
    with open("model_bias_results.txt", 'w') as f:
        f.write(out)
