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

def evaluate_summarization_bias(name, content_id, research_question, codebook=None,summarization_ratio=1/10):
    # Content can be either Youtube video id or .srt file path
    results = ""
    if ".srt" in content_id:
        transcript=load_and_preprocess_srt(content_id)
        results += "\nEvaluating transcript:" + content_id
    else:
        transcript=''
        for content in content_id if isinstance(content_id,list) else [content_id]:
            srt_dict = YouTubeTranscriptApi.get_transcript(content)
            transcript += ' '.join(sub['text'] for sub in srt_dict)
            results += "\nEvaluating transcript from youtube video id:" + content

    if codebook is None or isinstance(codebook,int):
        codebook=generate_codebook(transcript, NUM_THEMES=codebook if isinstance(codebook,int) else 3)
    summary_scores={}

    total_tokens=len(BART_tokenizer.tokenize(transcript))
    max_tokens = total_tokens*summarization_ratio
    results += f"\ntotal tokens: {total_tokens}\n Summarized to: <{max_tokens}\n\n"
    results += f"Themes: \n {str(codebook)}\n\n"
    print(results)
    summaries = {}
    # ADD ANY MODEL TO COMPARE, JUST ADD TO SUMMARIES DICT TO BE EVALUATED:
    
    # ANTHROPIC_KEY = 0#'YOUR_ANTHROPIC_KEY'
    # summaries['claude'] = claude_summarize_text(transcript, max_tokens=max_tokens, ANTHROPIC_KEY=ANTHROPIC_KEY)
    # summary_scores['claude'] = classify_sentences(summaries['claude'], codebook, research_question)
    # OPENAI_APIKEY = ""
    # summaries['gpt4'] = gpt_summarize_text(transcript, max_tokens=max_tokens, OPENAI_API_KEY=OPENAI_APIKEY)
    # results += "\n\nGPT4 SUMMARY: \n" + summaries['gpt4'] 

    print("getting sentence embeddings.")
    sentences =  [sent.text for sent in nlp(transcript).sents]
    sentences = [sent for sent in sentences if sent!="" and sent!=" "]
    sentence_embeddings = [get_embedding(sentence) for sentence in sentences]

    print("classifying baseline for original text.")
    fulltext_scored = classify_sentences(sentences, codebook, research_question, sentence_embeddings=sentence_embeddings)

    
    
    summaries["roberta_model"] = roberta_summarize(" ".join(sentences))
    results += "\n\n ROBERTA_LARGE SUMMARY: \n" + summaries['roberta_model']
    summaries["t5_base"] = t5summarize(transcript)
    results += "\n\nT5_BASE SUMMARY: \n" + summaries['t5_base']
    summaries["agglomerative_clustering"] = agglomerative_summary(sentences)
    results += "\n\nAGGLOMERATIVE CLUSTERING SUMMARY: \n" + summaries['agglomerative_clustering']    
    summaries['bart_model'] = bart_summarize_text(transcript, max_tokens=max_tokens)
    results += "\n\nBART SUMMARY: \n" + summaries['bart_model']
    summaries['kmeans_clustering'] = kmeans_centroid_summary(sentences)
    results += "\n\nK-MEANS CENTROID SENTENCES: \n" + summaries['kmeans_clustering']
    # For each summary algorithm, summarize transcript and create a scoreboard as a list of sentence classification frequencies
    results += "\n\n"
    knn_summaries={}
    # Create grounded summaries for non-knn summaries
    for summary_key in [key for key in summaries.keys() if not key.endswith('clustering')]:
        knn_key = f'{summary_key}_knn'
        knn_summaries[knn_key] = create_grounded_summary(transcript, summaries[summary_key], sentence_embeddings)
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
            results += f"\n\nGENERATED A HEATMAP OF THE STATISTICAL SIGNIFICANCE FOR {summary_type}\n"  + generate_chi_square_heatmap(fulltext_scored, summary_scoreboards, name = summary_type + content_id)
    results += f'\n\n~\n' #'change in distribution comparing for original text to abstract summary versus original text to representative sample of quotes:\n\n'

    unique_keys = set()


    # Iterate through each value in summary_scores and add its keys to the set
    for summary_scoreboard in summary_scores.values():
        unique_keys.update(summary_scoreboard.keys())
    # Convert the set of unique keys to a list
    categories = list(unique_keys)

    results +="\n\n\n" + line_graph(fulltext_scored, summary_scores, categories, file_name=str(content_id+"percentchangeline"), title=name)

    print("\n\n\nRESULTS:\n")
    print(results)
    return results


from transformers import T5Tokenizer, T5ForConditionalGeneration
def t5summarize(input_text: str, summarization_ratio: float = 0.2, max_chunk_len: int = 508) -> str:
        tokenizer=T5Tokenizer.from_pretrained('t5-base')
        model=T5ForConditionalGeneration.from_pretrained('t5-base')
        # Validate input text
        if not input_text or not isinstance(input_text, str):
            raise ValueError("Input text must be a non-empty string")

        # Validate summarization_ratio
        if summarization_ratio <= 0 or summarization_ratio >= 1:
            raise ValueError("Summarization ratio must be between 0 and 1")

        # Initialize variables
        summary_text = ""
        if  len(tokenizer.tokenize(input_text))>max_chunk_len:
            chunks = wrap(input_text, width=max_chunk_len)
        else:
            chunks = [input_text]
        print("T5 to summarize, wordlen", len(input_text.split(" ")))
        for chunk in chunks:
                # Tokenize the input chunk without padding and truncation to find the actual length of the chunk
            input_text = "summarize: " + chunk

            # Tokenize the input text
            input_tokens = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

            # Calculate the desired number of new tokens for the summary based on the summarization ratio and chunk length
            max_chunk_len = input_tokens.shape[1]
            actual_chunk_len = input_tokens.shape[1] - 2  # Exclude the special tokens
            max_new_tokens = int(min(actual_chunk_len, max_chunk_len) * summarization_ratio)

            # Generate the summary
            summary_ids = model.generate(
                input_tokens,
                max_length=max_new_tokens,
                min_length=0,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )

            # Decode the summary back to text
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            summary_text += summary + " "
            #print(summary_text)
        print("T5 to summarylen", len(summary_text.split(" ")))
        return summary_text.strip()

#import math
#from transformers import RobertaTokenizer, RobertaForCausalLM
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

def roberta_summarize(text, summarization_ratio=0.1):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base')
    # Split the text into sentences or coherent chunks

    chunks = wrap(text, width=510)  




    important_chunks = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        # Assuming the 'important' class is labeled as 1
        if predictions[0][1] > 0.1:  # This threshold can be adjusted
            important_chunks.append(chunk)

    # Concatenate selected chunks to form a summarized content
    summary = ". ".join(important_chunks[:int(len(important_chunks) * summarization_ratio)])
    print("ROBERTA SUMMARIZED:")
    print(summary)
    return summary
  
import torch
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from transformers import BertTokenizer, BertModel
def generate_codebook(transcript,NUM_THEMES=5):
    # Define the number of themes to extract
   
    # Load pre-trained BERT model and tokenizer
    tokenizer = BERT_tokenizer #BertTokenizer.from_pretrained('bert-base-uncased')
    model = BERT_embedding_model #BertModel.from_pretrained('bert-base-uncased')
    if isinstance( transcript, str):
        # Preprocess the text data
        sentences = [sent.text for sent in nlp(transcript).sents]
    else:
        sentences=transcript
   # Tokenize and encode the text data
    inputs = tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors="pt")

    # Generate embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:,  0, :].numpy()

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=NUM_THEMES, random_state=0).fit(embeddings)

    # Identify the most dominant words in each cluster
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform([' '.join(doc.split()) for doc in sentences])
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
def line_graph(full_codebook_scores, scoreboards, categories, file_name="line_graph.png",title=""):
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
    plt.title(f'Thematic Distribution Across Summarization Methods of {title}')
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
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import calinski_harabasz_score
from collections import OrderedDict
from nltk.corpus import stopwords
from nltk import word_tokenize

def agglomerative_summary(sentences: list[str], summarization_ratio=0.2,model=None,subs=None):
        if subs is not None:
            sub_time_text_list = [(i, subtitle.start.total_seconds(), subtitle.end.total_seconds(), subtitle.content) for i, subtitle in enumerate(subs)]
            i=""
            formatted_strings_from_subtitles = [f"{i}: [{start:.2f} - {end:.2f}] {text}" for i, start, end, text in sub_time_text_list]
            formatted_strings_from_subtitles = [f"{i}:  {text}  - [{end:.2f}]" for i, start, end, text in sub_time_text_list]
            sentences = formatted_strings_from_subtitles


        sentences = [s for s in list(OrderedDict.fromkeys(sentences)) if s]
        clean_sentences = []
        stop_words = set(stopwords.words('english'))
        for s in sentences:
            # Tokenize and lower case
            words = word_tokenize(s.lower())
            
            # Remove stop words and whitespace
            clean_words = [w for w in words if w not in stop_words and not w.isspace()]
            # Join the words back into a sentence and add to the list
            clean_sentences.append(' '.join(clean_words))

  #      sentences = [sentence for sentence in sents if sentence is not None and sentence != ""]
  #      clean_sentences = [' '.join(w for w in word_tokenize(s.lower()) if w not in stop_words and not w.isspace()) for s in sentences]
        #print("clean sentences",clean_sentences)
        # Generate sentence embeddings
        if model is None:
            model = SentenceTransformer('all-MiniLM-L6-v2') 
        
        sentence_embeddings = [model.encode(sentence) for sentence in clean_sentences]

        if not sentence_embeddings:
            return
        try:
            Z = linkage(sentence_embeddings, method='ward')
            clusters = [[s] for s in sentences]
            best_ch_score = -np.inf
            best_num_clusters = 1
            # Iterate over each step in the hierarchical clustering
            for num_clusters in range(2, len(Z) + 1):
                labels = fcluster(Z, criterion='maxclust', t=num_clusters)
                if len(labels)>1:
                    ch_score = calinski_harabasz_score(sentence_embeddings, labels)

                    if ch_score > best_ch_score:
                        best_ch_score = ch_score
                        best_num_clusters = num_clusters
                        if ch_score>10:
                            print("ch score", ch_score)
            # Use the optimal number of clusters
            labels = fcluster(Z, criterion='maxclust', t=best_num_clusters)
            # Map each sentence to its cluster label
            sentence_to_cluster = {sentence: label for sentence, label in zip(sentences, labels)}

            # Map each sentence to its original index
            sentence_to_index = {sentence: index for index, sentence in enumerate(sentences)}

            # Sort sentences based on their cluster label and original order in the text
            sentences_sorted = sorted(sentences, key=lambda s: (sentence_to_cluster[s], sentence_to_index[s]))

            # Select the most representative sentences based on the summarization ratio
            num_sentences_summary = max(int(len(sentences_sorted) * summarization_ratio), 1)
            summary_sentences = sentences_sorted[:num_sentences_summary]
            chronological_summary = [sentence for sentence in sentences if sentence in summary_sentences]
            # Combine the selected sentences into a summary
            summary = ' '.join(chronological_summary)

            return summary
        except:
        #    logging.warning(f"AVERTED: ValueError: The number of observations cannot be determined on an empty distance matrix.")
            return "".join(clean_sentences)
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
    print("SUCCESSFUL KMEANS, clusters:", n_clusters)

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


def create_grounded_summary(original_text, summarized_text, original_text_sentence_embeddings=None):
    """
    Ground each sentence of the summarized text to the closest original subtitle.
    
    Args:
    original_text (str): The original text.
    summarized_text (str): The summarized text.
    
    Returns:
    List[str]: A list of ground-truth normalized sentences.
    """
    if original_text_sentence_embeddings:
        original_embeddings=original_text_sentence_embeddings
    else:
        original_embeddings = [get_embedding(sent) for sent in original_sentences]
    if isinstance(original_text,str):
        original_sentences =  [sent.text for sent in nlp(original_text).sents]
    else: 
        original_sentences=original_text

    summarized_sentences =  [sent.text for sent in nlp(summarized_text).sents]
    summarized_embeddings = [ get_embedding(sent) for sent in summarized_sentences]

    gc.collect()
    # Nearest Neighbor Search
    nbrs = NearestNeighbors(n_neighbors=1, metric='cosine').fit(original_embeddings)

    grounded_sentences = []
    available_indices = set(range(len(original_sentences)))  # Set of indices of available sentences

    for summarized_embedding in summarized_embeddings:
        # Initialize a variable to track if a match has been found
        match_found = False
        
        while not match_found:
            # Find the nearest neighbor among all sentences, ignoring availability for now
            distances, indices = nbrs.kneighbors([summarized_embedding])
            
            for index in indices[0]:
                if index in available_indices:
                    # Found an available match
                    grounded_sentences.append(original_sentences[index])
                    available_indices.remove(index)
                    match_found = True
                    break  # Exit the for loop once a match is found
            
            # If no available match was found in this iteration, break the while loop to avoid infinite loop
            if not match_found:
                break

    return ". ".join(grounded_sentences).strip() + "."

def get_embedding(text):
    inputs = BERT_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = BERT_embedding_model(**inputs)
    # Use the pooled output for sentence-level representation
    return outputs.pooler_output.squeeze().detach().numpy()

    
def classify_sentences(text, codebook=["categories","themes","values"], research_question="The statement favors", sentence_embeddings=None):
    print("len codebook:",len(codebook))
    if isinstance(codebook[0], tuple):
        print("\ncodebook is tuples")
        pos_codebook = [code for (code,_) in codebook]
        print("len pos_codebook",len(pos_codebook))
        neg_codebook = [code for (_,code) in codebook]
        print("len neg_codebook",len(neg_codebook))
        codebook = pos_codebook
        codebook.extend(neg_codebook)  
        print(f"\nnow codebook is type {type(codebook)}\n of length {len(codebook)}" )
    else:
        neg_codebook=None
    if isinstance(text,list):
        sentences=text
    elif isinstance(text,str):
        sentences =  [sent.text for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if sent!="" and sent!=" "]
        sentence_embeddings = [get_embedding(sentence) for sentence in sentences] if sentence_embeddings is None else sentence_embeddings
    
    categorized_sentences = {} #{phrase for phrase in codebook}  # Initialize scores 
    for sentence, sentence_embedding in zip(sentences, sentence_embeddings):
        sentence_nsp_scores = {}
        for category in codebook:
            
             #Prepare text pair for NSP
            hypothesis = f"{category} {research_question}"

            inputs = BERT_tokenizer.encode_plus(hypothesis, sentence, return_tensors='pt', max_length=512, truncation='only_second')

            # Get NSP score
            nsp_outputs = BERT_nsp_model(**inputs)

            nsp_score = nsp_outputs.logits[:,0].detach().numpy()[0]  # Index 0 for 'isNext' label

            sentence_nsp_scores[category] = nsp_score

        most_favored_category = max(sentence_nsp_scores, key=sentence_nsp_scores.get)
        categorized_sentences[sentence]= most_favored_category

    category_counts = dict(collections.Counter(category for sentence, category in categorized_sentences.items()))

    if neg_codebook:
        # we need to score each category with a positive theme and a negative theme
        pos_counts = {}
        neg_counts = {}
        total_counts = {}
        counts = []
        for (category,count) in category_counts.items():
            if category in neg_codebook:

                neg_counts[category] = count
            elif category in pos_codebook:

                pos_counts[category] = count
        for pos_code,neg_code in zip(pos_codebook,neg_codebook):
            if pos_code not in pos_counts:
                pos_counts[pos_code] = 0
            if neg_code not in neg_counts:
                neg_counts[neg_code] = 0
            count = pos_counts[pos_code] - neg_counts[neg_code]
            total_counts[pos_code+'//' + neg_code] = count
            counts.append(count)
        # Normalize to positive for chi-square test
        for code , count in total_counts.items():
            total_counts[code] = count + min(counts)
            
    else:
        total_counts = category_counts 
    sorted_total_counts = dict(sorted(total_counts.items(), key=lambda item: item[1], reverse=True))

    return calculate_percentages(sorted_total_counts)

    
 
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

    research_question = "This theme is represented in the passage: "

    gospels_codebook = [
    ("Compassion: Demonstrating care and understanding towards others, acting on the principle of empathy.",
     "Apathy: Showing a lack of interest or concern for the hardships or suffering of others."),
    ("Mercy: Exercising forgiveness and leniency towards those who have erred or caused harm.",
     "Retribution: Advocating for punitive measures as a response to wrongdoing, without consideration for forgiveness."),
    ("Modesty: Exhibiting humility and a lack of arrogance in one's achievements and interactions with others.",
     "Arrogance: Displaying an exaggerated sense of one's own importance or abilities, often at the expense of others."),
    ("Trustworthiness: Being reliable and faithful in one's commitments and responsibilities.",
     "Skepticism: Exhibiting doubt and a critical questioning of motives, possibly leading to mistrust."),
    ("Altruism: Prioritizing the welfare of others and engaging in selfless acts of kindness.",
     "Egoism: Focusing on one's own interests and well-being without regard for the needs of others.")
]

    video_id = "Ed41paFWSKM" # 60min Parables
    name="Moral Summaries: \n The Parables of Jesus Christ\n"
    out=name
    out += evaluate_summarization_bias(name=name, content_id=video_id, research_question=research_question,codebook=gospels_codebook)
    with open(video_id + "model_bias_results.txt", 'w') as f:
        f.write(out)
    name="\n\n\n\n\nThe gospels:\n"
    out+=name
    video_id = "3UxowslJeTI" # 8 hours of the gospels
    out += evaluate_summarization_bias(name=name, content_id=video_id, research_question=research_question,codebook=gospels_codebook)

    with open(video_id + "model_bias_results.txt", 'w') as f:
        f.write(out)
    name="\n\n\n\n\nAesop's Fables:\n"
    out+=name


    aesops_fables_codebook = [
        ("Prudence: Exhibiting caution and wisdom in decision-making, foreseeing and avoiding unnecessary risks.",
        "Recklessness: Acting hastily without forethought, leading to avoidable trouble or danger."),
        ("Ingenuity: Demonstrating cleverness and originality to solve problems or overcome obstacles.",
        "Simplicity: Lacking complexity or depth in thought, sometimes leading to naive or ineffective actions."),
        ("Integrity: Adhering to moral and ethical principles, especially honesty and fairness in actions.",
        "Deceit: Employing dishonesty or trickery for personal gain or to deceive others."),
        ("Perseverance: Showing steady persistence in a course of action, especially in spite of difficulties or obstacles.",
        "Indolence: Avoiding activity or exertion; laziness, leading to missed opportunities or failure."),
        ("Social Harmony: Valuing and working towards the well-being of the community and relationships.",
        "Selfishness: Placing one's own needs and desires above the welfare of others, disrupting social harmony.")
    ]

    video_id="aaMLVsH6ikE" #aesops fables 3hr
    out += evaluate_summarization_bias(name=name, content_id=video_id, research_question=research_question,codebook=aesops_fables_codebook)
    with open(video_id + "model_bias_results.txt", 'w') as f:
        f.write(out)

    zen_koans_codebook = [
        ("Insight: Achieving a deep understanding or realization that transcends conventional thought and logic.",
        "Confusion: Remaining entangled in intellectual or logical reasoning, missing the essence of Zen."),
        ("Mindfulness: Maintaining a moment-by-moment awareness of thoughts, feelings, bodily sensations, and the surrounding environment.",
        "Distraction: Losing focus on the present moment, being caught up in thoughts about the past or future."),
        ("Simplicity: Embracing the value of living simply and recognizing the essence of reality without unnecessary complexity.",
        "Complexity: Adding unnecessary layers of thought or material desires that cloud the true nature of reality."),
        ("Detachment: Letting go of attachments to outcomes, opinions, and the ego, to see the true nature of things.",
        "Attachment: Clinging to personal desires, opinions, and the ego, which obstructs clear understanding and realization."),
        ("Non-duality: Recognizing the interconnectedness and oneness of all things, transcending the distinctions between self and other.",
        "Dualism: Perceiving the world in terms of binary opposites, such as self vs. other, right vs. wrong, which limits understanding of the true nature of reality.")
    ]
    name="\n\n\n\n\nZen Koans:\n"
    out+=name

    video_id= "Y0p663Ot8mo"#zen koans 1hr
    out += evaluate_summarization_bias(name=name, content_id=video_id, research_question=research_question,codebook=zen_koans_codebook)
    
    with open(video_id + "model_bias_results.txt", 'w') as f:
        f.write(out)

    with open("model_bias_results.txt", 'w') as f:
        f.write(out)
    print("model_bias_results.txt")
