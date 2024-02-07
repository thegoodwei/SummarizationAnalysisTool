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

def evaluate_summarization_bias(content_title, research_question, codebook=None,summarization_ratio=1/10):
    # Content can be either Youtube video id or .srt file path
    if ".srt" in content_title:
        transcript=load_and_preprocess_srt(content_title)
    else:
        srt_dict = YouTubeTranscriptApi.get_transcript(content_title)
        transcript = ' '.join(sub['text'] for sub in srt_dict)
    if codebook is None:
        codebook=generate_codebook(transcript)
    summary_scores={}

    results = ""
    results += "Evaluating transcript from youtube video id:" + content_title
    total_tokens=len(BART_tokenizer.tokenize(transcript))
    max_tokens = total_tokens*summarization_ratio
    results += f"\ntotal tokens: {total_tokens}\n Summarized to: <{max_tokens}"
    results += str(codebook) + "\n\n"
    print(results)
    summaries = {}
    # ADD ANY MODEL TO COMPARE, USE IT TO ADD TO SUMMARIES DICT :
    # ANTHROPIC_KEY = 0#'YOUR_ANTHROPIC_KEY'
    # summaries['claude'] = claude_summarize_text(transcript, max_tokens=max_tokens, ANTHROPIC_KEY=ANTHROPIC_KEY)
    # summary_scores['claude'] = classify_sentences(summaries['claude'], codebook, research_question)
  #  OPENAI_APIKEY = ""
 #   summaries['gpt4'] = gpt_summarize_text(transcript, max_tokens=max_tokens, OPENAI_API_KEY=OPENAI_APIKEY)
 #   results += "\n\nGPT4 SUMMARY: \n" + summaries['gpt4'] 
    fulltext_scored = classify_sentences(transcript, codebook, research_question)



    summaries['bart'] = bart_summarize_text(transcript, max_tokens=max_tokens)
    results += "\n\nBART SUMMARY: \n" + summaries['bart']
    summaries['kmeans'] = kmeans_centroid_summary(transcript)
    results += "\n\nK-MEANS CENTROID SENTENCES: \n" + summaries['kmeans']
    # For each summary algorithm, summarize transcript and create a scoreboard as a list of sentence classification frequencies
    results += "\n\n"
    knn_summaries={}
    # Create grounded summaries for non-knn summaries
    for summary_key in [key for key in summaries.keys() if not key.endswith('kmeans')]:
        knn_key = f'{summary_key}_knn'
        knn_summaries[knn_key] = create_grounded_summary(transcript, summaries[summary_key])
    summaries.update(knn_summaries)

    summary_scores = {}
    # Classify sentences for all summaries 
    for summary_key, summary_text in summaries.items():
        summary_scores[summary_key] = classify_sentences(summary_text, codebook, research_question)

    results += f"\n\n\n To classify, each BERT sentence embedding compares to a research question and category to predict most likely.\n Research Question:\n {research_question}:"
    for summary_type, summary_scoreboards in summary_scores.items():
    #   Compare the distribution of sentence categories applied in the summary to the original source distribution
        distribution_compared = compare_distributions(fulltext_scored, summary_scoreboards)
        results += distribution_compared
        if "(reject H0)" not in distribution_compared: 
            results += f"\n\nGENERATED A HEATMAP OF THE STATISTICAL SIGNIFICANCE FOR {summary_type}\n"  + generate_chi_square_heatmap(fulltext_scored, summary_scoreboards, name = summary_type + content_title)
    results += f'\n\n~\n' #'change in distribution comparing for original text to abstract summary versus original text to representative sample of quotes:\n\n'

    unique_keys = set()
    # Add keys from fulltext_scored to the set
    # unique_keys.update(fulltext_scored.keys())  #This will take all categories from main; not just the coded categories
    # Iterate through each value in summary_scores and add its keys to the set
    for summary_scoreboard in summary_scores.values():
        unique_keys.update(summary_scoreboard.keys())
    # Convert the set of unique keys to a list
    categories = list(unique_keys)


    #results +="\n\n\n" + generate_line_graph(fulltext_scored, comparative_scores, categories, name=content_title)
    results +="\n\n\n" + line_graph(fulltext_scored, summary_scores, categories, file_name=str(content_title+"percentchangeline"))
#    results+="\n"+generate_comparative_line_graph(full_codebook_scores, comparative_scores, categories, name="percentage_change_graph.png")
    print("\n\n\nRESULTS:\n")
    print(results)
    return results
import torch
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from transformers import BertTokenizer, BertModel
def generate_codebook(transcript,NUM_THEMES=5):
    # Define the number of themes to extract
    NUM_THEMES =  5

    # Load pre-trained BERT model and tokenizer
    tokenizer = BERT_tokenizer #BertTokenizer.from_pretrained('bert-base-uncased')
    model = BERT_embedding_model #BertModel.from_pretrained('bert-base-uncased')

    # Preprocess the text data
    text_data = [sent.text for sent in nlp(transcript).sents]
   # Tokenize and encode the text data
    inputs = tokenizer(text_data, padding=True, truncation=True, max_length=512, return_tensors="pt")

    # Generate embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:,  0, :].numpy()

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=NUM_THEMES, random_state=0).fit(embeddings)

    # Identify the most dominant words in each cluster
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform([' '.join(doc.split()) for doc in text_data])
    word_counts = X.sum(axis=0).flatten()
    most_common_words = [(word, word_counts[0, idx]) for word, idx in vectorizer.vocabulary_.items()]

    # Sort the words by frequency
    sorted_words = sorted(most_common_words, key=lambda x: x[1], reverse=True)

    # Extract the themes
    themes = [[] for _ in range(NUM_THEMES)]
    for i in range(NUM_THEMES):
        for j, (word, _) in enumerate(sorted_words):
                # Ensure the index is within the bounds of kmeans.labels_
            if j < len(kmeans.labels_) and kmeans.labels_[j] == i:
                themes[i].append(word)

        # Print the identified themes
    for i, theme in enumerate(themes):
        print(f"Theme {i+1}: {' '.join(theme)}")
        
    return [str(" ".join(theme)) for theme in themes] # Best line graph
def line_graph(full_codebook_scores, scoreboards, categories, file_name="line_graph.png"):
    """ 
    Generate a scalable line graph showing the percentage difference for each category across various text types and save it as an image file. 
    Display the average absolute value percentage change for each summary type compared to the full_codebook_scores.

    Args:
    full_codebook_scores (dict): Category percentages for the original text.
    scoreboards (dict of dict): Dictionary of dictionaries, each representing category percentages for a specific summary type.
    categories (list): List of all categories.
    file_name (str): Name of the file to save the graph.

    Returns:
    None
    """
    # Initialize lists to store the percentage differences for each category in each text type
    percentage_diffs = {text_type: {category: 0 for category in categories} for text_type in scoreboards.keys()}
    avg_percentage_diffs = {text_type: 0 for text_type in scoreboards.keys()}
    
    # Calculate the percentage differences for each category in each text type
    for category in categories:
        original_val = full_codebook_scores.get(category, 0)
        for text_type, scores in scoreboards.items():
            val = scores.get(category, 0)
            percent_diff = val - original_val  # Calculate percentage difference
            percentage_diffs[text_type][category] = percent_diff  # Store the percentage difference
    
    # Calculate the average absolute percentage change for each text type
    for text_type, cat_diffs in percentage_diffs.items():
        avg_percentage_diffs[text_type] = np.mean([abs(diff) for diff in cat_diffs.values()])  # Calculate and store the average of the absolute values
    # Sort text types based on their average absolute percentage change 
    #text_types = ["Original"] + sorted(scoreboards.keys(), key=lambda x: avg_percentage_diffs[x])
    
    text_types = ["Original"] 
    text_types.extend(sorted(scoreboards.keys())) #alphabetical order, to put _knn normalizations aside the abstract sums in the graph
    # No difference compared to baseline

    avg_percentage_diffs["Original"] = 0
    percentage_diffs["Original"] = {category:0 for category in categories}
    #for category in categories:
   #     percentage_diffs["Original"][category]=0
    # Create the line graph
    plt.figure(figsize=(10, 6))


        # Plot lines for each category, following the sorted order of text_types
    for category in categories:
        category_data = [percentage_diffs[text_type][category] for text_type in text_types]
        plt.plot(text_types, category_data, label=category, marker='o')

    # Plot the average points for each summary type, following the sorted order
    for i, text_type in enumerate(text_types):
        plt.scatter(text_type, avg_percentage_diffs[text_type], color='black', zorder=5)
        plt.text(text_type, avg_percentage_diffs[text_type], f'  Avg: {avg_percentage_diffs[text_type]:.2f}%' if avg_percentage_diffs[text_type]>0 else '', verticalalignment='bottom', color='black')

    """
    # Plot lines for each category
    for category in categories:
        category_data = [percentage_diffs[text_type][category] for text_type in scoreboards.keys()]
        plt.plot(text_types, category_data, label=category, marker='o')

    # Plot the average points for each summary type
    for i, text_type in enumerate(text_types): #enumerate(scoreboards.keys()):
        plt.scatter(text_types[i], avg_percentage_diffs[text_type], color='black', zorder=5)
        plt.text(text_types[i], avg_percentage_diffs[text_type], f'  Avg: {avg_percentage_diffs[text_type]:.2f}%', verticalalignment='bottom', color='black')
"""

    # Customize the graph
    plt.title('Thematic Biasses Compared Across Summarization Methods Tested')
    plt.xlabel('Text Type')
    plt.ylabel('Percentage Difference')
    plt.xticks(rotation=45, ha='right')  # Rotate text type names for better visibility
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add grid for better readability
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place the legend outside the plot
    
    file_name = file_name + ".png" if ".png" not in file_name else file_name
    # Save the graph as an image file
    plt.savefig(file_name, format='png', dpi=300, bbox_inches='tight')
    # Display the graph
    plt.show()
    return file_name
    
def generate_percentage_change_graph_revised_line(full_codebook_scores, scoreboards, categories, file_name="percentage_change_graph"):
    """
    Generate a line graph showing the percentage change for each category from the original text, and save it as an image file. 
    Display the average percentage change for each summary type from the original text.

    Args:
    full_codebook_scores (dict): Category percentages for the original text.
    abstract_summary_scores (dict): Category percentages for the abstract summary.
    extractive_summary_scores (dict): Category percentages for the extractive summary.
    categories (list): List of all categories.
    file_name (str): Name of the file to save the graph.

    Returns:
    None
    """
    # Initialize lists to store the percentage changes
    avg_changes = [0]


    
    category_data=[[0 for _ in categories]]
    plt.figure(figsize=(10, 6))
    text_types = ['Original Text']
    text_types.extend([sumtype for sumtype,_ in scoreboards.items()])
    # Collect the data for each category and calculate the percentage changes

    for category in categories:
        changes=[]
        original_val = full_codebook_scores.get(category, 0)
        for (sumtype,scoreboard) in scoreboards.items():
            category_score = scoreboard.get(category, 0)
            change = abs(category_score - original_val)
            changes.append(change)
        category_data.append(changes)
        avg_change = np.mean(changes)
        avg_changes.append(avg_change)
        plt.scatter(category, changes, color='blue', zorder=5)

        # category_changes=[]
        #abstract_vals.append(abstract_val)
    #extractive_val = extractive_summary_scores.get(category, 0)
    #extractive_change = extractive_val - original_val

        # Store the changes for average calculation
        #abstract_changes.append(abstract_change)
        #extractive_changes.append(extractive_change)

        # Calculate the average percentage change
    #avg_extractive_change = np.mean(extractive_changes)
    
#    plt.scatter(text_types, avg_changes, color='blue', zorder=5)
        
    for i,change in enumerate(avg_changes):
        #plt.text(text_types[i], change, f'  Avg: {change:.2f}%', verticalalignment='bottom', color='red')
        plt.text(text_types[i], change, f'  Avg: {change:.2f}%', verticalalignment='bottom', color='red')
    
    for i,sumtype in enumerate(text_types):
        for k,category in enumerate(categories):
        #for j,changes in enumerate(category_data):
            plt.plot(sumtype, category_data[i][k], label=category, marker='o')


#    plt.plot(text_types, category_data, label=categories[i], marker='o')
    # Create the line graph
    #plt.plot(text_types[0], 0, marker='o')
    #for i, category_data in enumerate(data):
        
    
    # Plot the average points only for 'Abstract Summary' and 'Extractive Summary'
    #plt.scatter(text_types[1:], [avg_abstract_change, avg_extractive_change], color='red', zorder=5)
    #plt.text(text_types[1], avg_abstract_change, f'  Avg: {avg_abstract_change:.2f}%', verticalalignment='bottom', color='red')
    #plt.text(text_types[2], avg_extractive_change, f'  Avg: {avg_extractive_change:.2f}%', verticalalignment='bottom', color='red')
    
    # Customize the graph
    plt.title('Percentage Change per Category From Original Text')
    plt.xlabel('Text Type')
    plt.ylabel('Percentage Change')
    plt.xticks(rotation=45, ha='right')  # Rotate text type names for better visibility
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add grid for better readability
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place the legend outside the plot
    
    # Save the graph as an image file
    file_name+=".png" if ".png" not in file_name else ''
    plt.savefig(file_name, format='png', dpi=300, bbox_inches='tight')
    
    # Display the graph
    plt.show()
    return file_name
# Example usage within your existing code:
# categories = list(set(fulltext_scored.keys()).union(abstract_summary_scores[content_title].keys(), extractive_summary_scores[content_title].keys()))
# generate_percentage_change_graph(fulltext_scored, abstract_summary_scores[content_title], extractive_summary_scores[content_title], categories, "percentage_change_graph.png")

def generate_line_graph(full_codebook_scores:dict, scoreboards:dict, categories:dict, name:str=""):
    """
    Generate a line graph showing the average change in distribution for each category.

    Args:
    full_codebook_scores (dict): Category percentages for the original text.
    scoreboards (list[(str,dict)]): Category percentages for the abstract summary.
    categories (list): List of all categories.

    Returns:
    None
    """
    # Initialize lists to store the average percentages
    data = []
    abstract_differences = []

    plt.figure(figsize=(10, 6))
    #text_types = [summary_type for summary_type, _ in scoreboards.items()]
    for i,(text_type,abstract_summary) in enumerate(scoreboards.items()):

    # Collect the data for each category
        for category in categories:
            original_val = full_codebook_scores.get(category, 0)
            abstract_val = abstract_summary.get(category, 0)
            difference=abs(abstract_val - original_val)
            abstract_differences.append(difference)
            plt.plot(text_type, difference, label=category, marker='o')
           # abstract_change = abstract_val - original_val
            #data.append(abstract_change) 
            #avg_change = np.mean(abstract_differences)
            #plt.scatter(text_types[1:], [abs(abstract_val - original_val)], color='red', zorder=2)
        avg_change = np.mean(abstract_differences)
        # abstract_val = abstract_summary_scores.get(category, 0)
        # extractive_val = extractive_summary_scores.get(category, 0)
        # Calculate the absolute percentage differences
        # extractive_differences.append(abs(extractive_val - original_val))
        # extractive_change = extractive_val - original_val
        # data.append([0, abstract_change, extractive_change])  # Original text starts at 0

    # Calculate the average absolute percentage change
    #avg_extractive_change = np.mean(extractive_differences)
        plt.scatter(text_type, [avg_change], color='black', zorder=1)
        plt.text(text_type, avg_change, f'  AvgDif: {avg_change:.2f}%', verticalalignment='bottom', color='black')
    # Create the line graph
        
        #for i, category_data in enumerate(abstract_differences):
         #   plt.plot(text_type, category_data, label=categories[i], marker='o')
# Add the average change lines
# Plot the average points
#avg_data = [0, avg_change, avg_extractive_change]
#plt.scatter(text_types, avg_data, color='black', zorder=5)
#plt.scatter(text_types[1:], [avg_change, avg_extractive_change], color='black', zorder=5)
#        plt.text(text_types[2], avg_extractive_change, f'  Avg: {avg_extractive_change:.2f}%', verticalalignment='bottom', color='black')
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

def generate_comparative_line_graph(full_codebook_scores, comparative_scores, categories, name="comparative_line_graph"):
    """
    Generate a dynamic line graph showing the percentage change for each category from the original text 
    compared to multiple sets of scores, and save it as an image file. Display the average percentage 
    change for each set of comparative scores from the original text.

    Args:
    full_codebook_scores (dict): Category percentages for the original text.
    comparative_scores (dict): Dictionary of {label: scores} for different sets of comparative scores.
    categories (list): List of all categories.
    name (str): Base name of the file to save the graph.

    Returns:
    str: The filename where the graph is saved.
    """
    # Initialize a list to store the data for the graph
    data = {label: [] for label in comparative_scores}
    avg_changes = {label: [] for label in comparative_scores}
    
    # Iterate through categories and calculate percentage changes
    for category in categories:
        original_val = full_codebook_scores.get(category, 0)
        for label, scores in comparative_scores.items():
            comparative_val = scores.get(category, 0)
            change = comparative_val - original_val
            data[label].append(change)
            avg_changes[label].append(abs(change))  # Note: Using absolute change for average

    # Calculate the average absolute percentage change for each set of comparative scores
    avg_changes = {label: np.mean(changes) for label, changes in avg_changes.items()}

    # Create the line graph
    plt.figure(figsize=(10, 6))
    text_types = ['Original Text'] + list(comparative_scores.keys())

    for label, changes in data.items():
        plt.plot(text_types, [0] + changes, label=label, marker='o')  # Original text starts at 0
    
    # Plot the average points for each set of comparative scores
    for i, (label, avg_change) in enumerate(avg_changes.items(), start=1):
        plt.scatter(text_types[i], avg_change, color='black', zorder=5)
        plt.text(text_types[i], avg_change, f'  Avg: {avg_change:.2f}%', verticalalignment='bottom', color='black')
    
    # Customize the graph
    plt.title('Percentage Change per Category From Original Text Across Different Summary Types')
    plt.xlabel('Text Type / Summary Type')
    plt.ylabel('Percentage Change')
    plt.xticks(rotation=45, ha='right')  # Rotate text type names for better visibility
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add grid for better readability
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place the legend outside the plot
    
    # Save the graph as an image file
    file_name = f"{name}_summarybias_compared.png"
    plt.savefig(file_name, format='png', dpi=300, bbox_inches='tight')
    
    # Display the graph
    plt.show()

    return file_name
def generate_percentage_change_graph(full_codebook_scores, comparative_scores, categories, name="percentage_change_graph.png"):
    """
    Generate a dynamic line graph showing the percentage change for each category from the original text compared to multiple sets of scores, and save it as an image file. 
    Display the average percentage change for each set of comparative scores from the original text.

    Args:
    full_codebook_scores (dict): Category percentages for the original text.
    comparative_scores (dict): Dictionary of {label: scores} for different sets of comparative scores.
    categories (list): List of all categories.
    file_name (str): Name of the file to save the graph.

    Returns:
    None
    """
    # Initialize a list to store the data for the graph
    data = {label: [] for label in comparative_scores}
    avg_changes = {label: [] for label in comparative_scores}
    
    # Iterate through categories and calculate percentage changes
    for category in categories:
        original_val = full_codebook_scores.get(category, 0)
        for label, scores in comparative_scores.items():
            comparative_val = scores.get(category, 0)
            change = comparative_val - original_val
            data[label].append(change)
            avg_changes[label].append(change)

    # Calculate the average percentage change for each set of comparative scores
    avg_changes = {label: np.mean(changes) for label, changes in avg_changes.items()}

    # Create the line graph
    plt.figure(figsize=(10, 6))
    text_types = ['Original Text'] + list(comparative_scores.keys())

    for label, changes in data.items():
        plt.plot(text_types, [0] + changes, label=label, marker='o')  # Original text starts at 0
    
    # Plot the average points for each set of comparative scores
    for i, (label, avg_change) in enumerate(avg_changes.items(), start=1):
        plt.scatter(text_types[i], avg_change, color='black', zorder=5)
        plt.text(text_types[i], avg_change, f'  Avg: {avg_change:.2f}%', verticalalignment='bottom', color='black')
    
    # Customize the graph
    plt.title('Percentage Change per Category From Original Text Across Different Summary Types')
    plt.xlabel('Text Type / Summary Type')
    plt.ylabel('Percentage Change')
    plt.xticks(rotation=45, ha='right')  # Rotate text type names for better visibility
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add grid for better readability
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place the legend outside the plot
    
    # Save the graph as an image file
    plt.savefig(name, format='png', dpi=300, bbox_inches='tight')
    
    # Display the graph
    plt.show()

    return name
# Example usage:
# categories = list(set(fulltext_scored.keys()).union(*[scores.keys() for scores in comparative_scores.values()]))
# generate_percentage_change_graph(full_codebook_scores, comparative_scores, categories, "percentage_change_graph.png")

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



from textwrap import wrap

import anthropic
def claude_summarize_text(input_text, max_tokens=32000, ANTHROPIC_KEY=""):
    """
    Summarize the input text using the Claude model with a  32,000 token context.

    Args:
    input_text (str): The input text to summarize.
    max_tokens (int): The maximum length of the summary.

    Returns:
    str: The summarized text.
    """
    # Define the chunk size to be slightly less than the model's context window
    chunk_size =  100000 -  100  # Subtracting some buffer space
    client = anthropic.Client(ANTHROPIC_KEY)

    # Split the input text into chunks
    chunks = wrap(input_text, width=chunk_size)

    # Initialize an empty string for the summary
    summary = ""

    # Process each chunk
    for chunk in chunks:
        # Construct the prompt for the API call
        prompt = f"{anthropic.HUMAN_PROMPT}: Summarize the following text, should be readable:\n\n{chunk}\n\n{anthropic.AI_PROMPT}:\n\nSummary"

        # Call the Claude API to generate the summary for the current chunk
        response = client.completion(
            prompt=prompt,
            model="claude-v1.3-100k",
            max_tokens_to_sample=max_tokens
        )

        # Append the summary of the current chunk to the overall summary
        summary += response["completion"]

    # Return the final summary text
    return summary
import openai
def gpt_summarize_text(input_text, max_tokens=32000, OPENAI_API_KEY="YOUR_OPENAI_KEY"):
    """
    Summarize the input text using the GPT-4 API with  32,000 token context.

    Args:
    input_text (str): The input text to summarize.
    max_tokens (int): The maximum length of the summary.

    Returns:
    str: The summarized text.
    """
    # Initialize conditions
    openai.api_key = OPENAI_API_KEY
    num_output_summary_tokens = max_tokens +  1  # Initialize to enter the while loop
    tokens_per_chunk =  512  # Initial optimal chunk size

    current_text = input_text  # Initialize current_text to be the input_text
    max_tokens = tokens_per_chunk -  1 if tokens_per_chunk > max_tokens else max_tokens  # Make the loop condition smaller than attention window

    while num_output_summary_tokens > max_tokens:
        # We chunk the current text (initially the input text, later could be the summary of the input text)
        chunks = wrap(current_text, width=tokens_per_chunk)
        print(f"{len(chunks)} chunks for GPT4 to analyze in these transcripts")
        # Summarize each chunk
        chunk_summaries = []
        for chunk in chunks:
            if len(chunk) >  5:
                message = {"role": "system", "content": f"Summarize the following text: {chunk}"}
                response = openai.ChatCompletion.create(
                    model="gpt-4-32k",
                    max_tokens=max_tokens,
                    temperature=0.7,
                    messages=[message]
                )
                gpt_response=response['choices'][0]['message']['content']
                print("GPT_RESPONSE: \n"+gpt_response)
                chunk_summaries.append(gpt_response)

        # Combine the chunk summaries
        current_text = ' '.join(chunk_summaries).strip()

        # Update the number of tokens in the current text (which is the total summary at this point)
        num_output_summary_tokens = len(current_text.split())

    # Return the final summary text, which is now guaranteed to be within the max_tokens limit
    return current_text

def bart_summarize_text(input_text, max_tokens=1000):
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
    max_tokens = tokens_per_chunk - 1 if tokens_per_chunk > max_tokens else max_tokens# make the loop condition smaller than attention window
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
        # This way, we are always summarizing symmetrically, maintaining the context as much as possible

    # Return the final summary text, which is now guaranteed to be within the max_tokens limit

    return current_text

# Normalize the abstractive summary to the original text quotes
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np

def kmeans_centroid_summary(original_text, summarization_ratio=.10):
    # Step 1: Preprocess the text to split it into sentences
    # Assuming original_text is a string, you would typically use a library like NLTK to split it into sentences
    # For simplicity, let's assume sentences is a list of sentences
    print("USING KMEANS CENTROID")
    if isinstance(original_text,list):
        sentences = original_text
    else:
        sentences = [sent.text for sent in nlp(original_text).sents]


    # Step 2: Generate Sentence Embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(sentences)

    # Step 3: Apply K-Means Clustering
    n_clusters = int(len(sentences) * summarization_ratio)
    print("kmeans")
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
    print(".predict")
    # Step 4: Assign sentences to clusters
    sentence_clusters = kmeans.predict(embeddings)
    print("SUCCESSFUL KMEANS, clusters:", len(n_clusters))

    # Step 5: Find the representative sentence for each cluster
    # Initialize a dictionary to store the representative sentence for each cluster
    representative_sentences = {}
    for cluster_id in range(n_clusters):
        # Filter sentences belonging to the current cluster
        cluster_sentences = [sent for sent, cluster in zip(sentences, sentence_clusters) if cluster == cluster_id]
        
        # Compute distances to the cluster center for the current cluster's sentences
        cluster_center = kmeans.cluster_centers_[cluster_id]
        cluster_distances = np.linalg.norm(embeddings[[sentences.index(sent) for sent in cluster_sentences]] - cluster_center, axis=1)
        
        # Select the sentence with the smallest distance to the cluster center
        closest_sentence_index = np.argmin(cluster_distances)
        representative_sentences[cluster_id] = cluster_sentences[closest_sentence_index]

    # Step 6: Form the summary by sorting the representative sentences by their original positions
    # Sort the representative sentences by their indices in the original text
    sorted_sentences = sorted(representative_sentences.values(), key=lambda x: sentences.index(x))
    summary = " ".join(sorted_sentences)

    return summary


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
    #
    
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
        results += str("No significant differences exist between the distributions (H0)\n")

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

# pip install youtube_transcript_api
if __name__ == '__main__':
    research_question = "The implication is "
    categories = ["War", "Peace"]
    video_id = "XpC7SVDXimg" # Israel 1
    out = evaluate_summarization_bias(content_title=video_id, research_question=research_question,codebook=categories)
    video_id = "34wA_bdG6QQ" # Palestine 2
    out += "\n\n" + evaluate_summarization_bias(content_title=video_id, research_question=research_question,codebook=categories)
    video_id = "Mde2q7GFCrw" # Peace 3
    out += "\n\n" + evaluate_summarization_bias(content_title=video_id, research_question=research_question,codebook=categories)

    research_question = "The statement politically favors "
    video_id="tYrdMjVXyNg" # Lex destiny shapiro political debate 
    categories=["Republicans", "Democrats", "Independants"]#, "Leaders", "Community", "Family", "Wealth", "Small Businesses", "Markets", "Big Corporations", "Global Trade", "Local Culture", "Libertarians", "War", "Peace"]
    out += '\n\n' + evaluate_summarization_bias(content_title=video_id, research_question=research_question,codebook=categories)
    with open("model_bias_results.txt", 'w') as f:
        f.write(out)
