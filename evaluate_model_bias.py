import torch
import scipy.spatial.distance
import scipy.stats
from scipy.spatial.distance import cosine
import transformers
from transformers import BertModel, BertTokenizer, BartForConditionalGeneration, BartTokenizer, BertForSequenceClassification, BertForNextSentencePrediction
import pysrt
import collections
import numpy as np
import textwrap
from scipy.stats import chi2_contingency
import spacy
import gc
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# These are used for classification functions called for each model evaluation; load them once at start:
nlp = spacy.load("en_core_web_lg")
BERT_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
BERT_tokenizer.add_special_tokens({'pad_token': '[PAD]'})# OR = BERT_tokenizer.eos_token
BERT_embedding_model = BertModel.from_pretrained('bert-base-uncased')
BERT_embedding_model.config.pad_token_id = BERT_tokenizer.pad_token_id
BERT_nsp_model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
BERT_nsp_model.config.pad_token_id = BERT_tokenizer.pad_token_id
GPT2_model = GPT2LMHeadModel.from_pretrained("gpt2") 
GPT2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def evaluate_summarization_bias(name, content_id, research_question, codebook=None,summarization_ratio=1/10):
    # Content can be either Youtube video id or .srt file path
    results = ""

    transcript = ""
    content_ids = ""
    for content in content_id if isinstance(content_id,list) else [content_id]:
        content_ids += content_id
        if ".srt" in content:
            transcript += " " + load_and_preprocess_srt(content)
        else:    
            srt_dict = YouTubeTranscriptApi.get_transcript(content)
            transcript += ' '.join(sub['text'].replace("[Music]","") for sub in srt_dict)
          
        results += f"Summarize the Moral of the Story: {name}\nPrimary Source: https://youtu.be/{content}\n"
    content_id = content_ids
    if codebook and isinstance(codebook[0], tuple):
        pos_codebook,neg_codebook=[],[]
        flat_codebook = []
        for (pos,neg) in codebook:
            pos_codebook.append(pos)
            neg_codebook.append(neg)
            flat_codebook.extend([pos,neg]) # Keep them in order
        codebook = flat_codebook
    else:
        pos_codebook = False
    summary_scores = {}
    total_tokens = len(BERT_tokenizer.tokenize(transcript))
    max_tokens = total_tokens*summarization_ratio
    results += f"\nTokenln Primary Source: {total_tokens} * Summarization Ratio: {summarization_ratio} = \n Tokenln Per Summary: <{max_tokens}\n\n"
    results += f"Themes for Classification: \n {str(codebook)}\n\n"
    results += "Models Evaluated: LLAMA-7b-chat-hf, Mistral-7b, phi-2, T5-base, bart-large-cnn, gpt2, roberta_extractive_modeling, Bert_Agglomerative_Clustering, k-Means_Clustering_(all-MiniLM-L6-v2) \n"
    results += "Post-summary grounding algorithm evaluated: K-Means Nearest-Neighbor for each Abstract Summary Sentence to the nearest unique Primary Source Quote sentence"
    print(results)
    summaries = {}
    sentences =  [sent.text for sent in nlp(transcript).sents if sent.text!=" "]
    transcript = ". ".join(sentences).strip()
    print("\n\n\nTranscript:\n"+transcript)

    # ADD MODELS TO SUMMARIES DICT TO BE EVALUATED:
    # ANTHROPIC_KEY = 0#'YOUR_ANTHROPIC_KEY'
    # summaries['claude'] = claude_summarize_text(transcript, max_tokens=max_tokens, ANTHROPIC_KEY=ANTHROPIC_KEY)
    # summary_scores['claude'] = classify_sentences(summaries['claude'], codebook, research_question)
    # OPENAI_APIKEY = ""
    # summaries['gpt4'] = gpt4_summarize_text(transcript, max_tokens=max_tokens, OPENAI_API_KEY=OPENAI_APIKEY)
    # results += "\n\nGPT4 SUMMARY: \n" + summaries['gpt4'] 

    summaries["roberta_extraction"] = extractive_summary_roberta(transcript, summarization_ratio=summarization_ratio)
    results += f"\n\n ROBERTA SUMMARY: \n {summaries['roberta_extraction']}"
    print("Roberta Summary: " + summaries["roberta_extraction"])
    assert (1<len(summaries['roberta_extraction'].split(" ")))

    summaries["phi-2_model"] = phi2_summarize(transcript, summarization_ratio=summarization_ratio)
    results += f"\n\n Phi-2 MODEL SUMMARY: \n {summaries['phi-2_model']}"
    print("Phi2 Summary: " + summaries["phi-2_model"])
    assert (1<len(summaries['phi-2_model'].split(" ")))
    
    summaries['bart_model'] = bart_summarize_text(transcript, summarization_ratio=summarization_ratio)
    results += f"\n\nBART SUMMARY: \n {summaries['bart_model']}" 
    print("Bart Summary: " + summaries["bart_model"])
    assert (1<len(summaries['bart_model'].split(" ")))
    
    summaries["mistral_model"] = mistral_summarize(transcript, summarization_ratio=summarization_ratio)
    results += f"\n\n MISTRAL MODEL SUMMARY: \n {summaries['mistral_model']}" 
    print("Mistral Summary: " + summaries["mistral_model"])
    print("LEN OF MISTRAL:", len(BERT_tokenizer.tokenize(summaries["mistral_model"])))
    assert (1<len(summaries['mistral_model'].split(" ")))

    summaries["llama_model"] = llama_summarize(transcript, summarization_ratio=summarization_ratio)
    results += "\n\n LLAMA SUMMARY: \n" + summaries['llama_model']
    print("LLAMA SUMMARY", summaries["llama_model"])
    assert (1<len(summaries['llama_model'].split(" ")))
    
    summaries["gpt2_model"] = gpt2_summarize(transcript, summarization_ratio=summarization_ratio)
    results += f"\n\n GPT2 MODEL SUMMARY: \n {summaries['gpt2_model']}"
    print("GPT2 Summary: " + summaries["gpt2_model"])
    assert (1<len(summaries['gpt2_model'].split(" ")))

    summaries["t5_model"] = t5_summarize(transcript, summarization_ratio=summarization_ratio)
    results += f"\n\nT5_BASE SUMMARY: \n{summaries['t5_model']}"
    print("T5 Summary", summaries["t5_model"])
    assert (1<len(summaries['t5_model'].split(" ")))

    summaries["agglomerative_clustering"] = agglomerative_sampling(transcript, summarization_ratio=summarization_ratio)
    results += f"\n\nAGGLOMERATIVE CLUSTERING SUMMARY: \n{summaries['agglomerative_clustering']}"     
    print("Agglomerative Clustering with Bert Embeddings", summaries["agglomerative_clustering"])
    assert (1<len(summaries['agglomerative_clustering'].split(" ")))
    
    summaries['kmeans_clustering'] = kmeans_centroid_sampling(transcript, summarization_ratio=summarization_ratio)
    results += f"\n\nK-MEANS CENTROID SENTENCES: \n {summaries['kmeans_clustering']}" 
    print("K-Means Clustering with all-MiniLM-L6-v2", summaries["agglomerative_clustering"])
    assert (1<len(summaries['kmeans_clustering'].split(" ")))
    results += "\n\n"
    
    gc.collect()
    sentence_embeddings = [get_embedding(sentence) for sentence in sentences]

    print("GENERATING CODEBOOK FROM SUMMARIES")
    inductive_codebook = generate_codebook(transcript + ". ".join([summary for summary in summaries.values()]), num_groups=(len(codebook)))
    print("LEN INDUCTIVE CODEBOOK:", len(inductive_codebook))
    print("LEN CODEBOOK:", len(codebook))
    codebook_comp = [" \n'".join(sents)+"'" for sents in ground_to_knn_pairs(codebook, inductive_codebook)]
    print(len(codebook) == len(inductive_codebook) == len(codebook_comp))
    assert(len(codebook) == len(inductive_codebook) == len(codebook_comp))
    results += f"\n\nInductive Codebook Extracted Keywords from All Summaries:\n"
    for i,code in enumerate(codebook_comp):
        results += f"Group {i} | {code}\n" # These groups could each be added to the nearest neighbor category to better classify, at cost of context window. ##TODO future research
        print(f"\nINDUCTIVE CODEBOOK GROUP {i} | {code}\n\n")# These groups could each be added to the nearest neighbor category to better classify, at cost of context window. ##TODO future research
        
        
    # K-NearestNeighbor post processing to adapt the abstractive summaries into extractive summaries
    for summary_key in [key for key in summaries.keys() if not ('clustering' or 'extract') in key]: #key.endswith('clustering')]:
        # Create grounded summaries for non-knn summaries
        knn = ". ".join([extracted_sent for extracted_sent,abstract_sent in ground_to_knn_pairs(transcript, summaries[summary_key], sentence_embeddings)]) + "."
        if knn != summaries[summary_key]:
            knn_key = f'{summary_key}_knn'
            summaries[knn_key] = knn

    print("classifying baseline for original text.")
    fulltext_scored, average_mean_difference = classify_sentences(transcript, codebook, research_question)
    results+=f"\nMeasuring difference confidence for BERT NSP and GPT2 classification:\nPrimary_Source_Text Average Mean Difference = {average_mean_difference}\n"
    summary_scores = {}
    
    # Classify sentences for all summaries 
    # For each summary algorithm, summarize transcript and create a scoreboard as a list of sentence classification frequencies
    for summary_key, summary_text in summaries.items():
        summary_scores[summary_key], average_mean_difference = classify_sentences(summary_text, codebook, research_question)
        results+=f"\nMeasuring difference confidence for BERT NSP and GPT2 classification:\n{summary_key} Average Mean Difference = {average_mean_difference}\n"

    scored_differences = {}
    results += f"\n\n\n To classify, each BERT sentence embedding compares to a research question and category to predict most likely.\n Research Question:\n {research_question}:"
    for summary_key, summary_scoreboard in summary_scores.items():
    #   Compare the distribution of sentence categories applied in the summary to the original source distribution
        distribution_compared = compare_distributions(fulltext_scored, summary_scoreboard, summary_type = summary_key + "_",name=name)

        results += distribution_compared
        scored_differences[summary_key] = {category:sumscore-origscore for category,sumscore in summary_scoreboard.items() for theme,origscore in fulltext_scored.items() if (theme==category)}
    results += '\n\n' + generate_and_export_table(scored_differences, export_csv_path = str(content_id + "_net_theme_table"))
    results += f'\n\n\n' #'change in distribution comparing for original text to abstract summary versus original text to representative sample of quotes:\n\n'
    # Iterate through each value in summary_scores and add its keys to the set
    unique_keys = set()
    gc.collect()
    for summary_scoreboard in summary_scores.values():
        unique_keys.update(summary_scoreboard.keys())
    categories = list(unique_keys)
    results +="\n\n\n" + model_span_linegraph(fulltext_scored, summary_scores, categories, file_name=str(content_id+"model_span_line"), title=name)
    results +="\n\n\n" + model_dif_scatterplot(fulltext_scored, summary_scores, categories, file_name=str(content_id+"theme_span_scatterplot"), title=name)
    if pos_codebook: #
    # results +="\n\n\n" + theme_span_linegraph(fulltext_scored, summary_scores, categories, file_name=str(content_id+"model_theme_evaluation"), title=name)
        scoreboard_ranges = {key:calculate_theme_ranges(summary_score,pos_codebook,neg_codebook) for key,summary_score in summary_scores.items()}
        fulltext_score_ranges = calculate_theme_ranges(fulltext_scored, pos_codebook,neg_codebook)
        categories = [key for key in fulltext_score_ranges.keys()]

        results +="\n\n\n" + bar_graph(fulltext_score_ranges, scoreboard_ranges, categories, file_name=str(content_id+"net_theme_bars"), title=name)

        net_scores_base = {category:pos-neg for category,(pos,neg) in fulltext_score_ranges.items()}
        net_scoreboards = {scoreboardkey:{category:pos-neg for category,(pos,neg) in scoreboard.items()} for scoreboardkey,scoreboard in scoreboard_ranges.items()}
        results +="\n\n\n" + theme_span_linegraph(net_scores_base, net_scoreboards, categories, file_name=str(content_id+"Net_theme_evaluation"), title=name)
    print("\n\n\nRESULTS:\n")
    print(results)
    return results
from sklearn.metrics.pairwise import euclidean_distances

#from sklearn.feature_extraction.text import TfidfVectorizer 
def generate_codebook(transcript,num_groups=5,n_sentences_per_cluster=3):
    tokenizer = BERT_tokenizer #BertTokenizer.from_pretrained('bert-base-uncased')
    model = BERT_embedding_model #BertModel.from_pretrained('bert-base-uncased')
    
    if isinstance( transcript, str):
        # Preprocess the text data
        sentences = [sent.text for sent in nlp(transcript).sents]
    else:
        sentences=transcript
        
    # Tokenize and encode the sentences using BERT.
    inputs = tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors="pt")
    
    # Generate sentence embeddings.
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()

    # Apply K-means clustering on the embeddings.
    kmeans = KMeans(n_clusters=num_groups, random_state=0).fit(embeddings)

    # Initialize a list to hold the key sentences for each cluster.
    key_sentences = [None] * num_groups
    
    # For each cluster, identify the sentences that belong to it and find the closest one to the centroid.
    for cluster_num in range(num_groups):
        # Find indices of sentences belonging to the current cluster.
        cluster_indices = [idx for idx, label in enumerate(kmeans.labels_) if label == cluster_num]
        
        # Extract the embeddings of sentences in the current cluster.
        cluster_embeddings = embeddings[cluster_indices]
        
        # Calculate the centroid of the current cluster.
        centroid = kmeans.cluster_centers_[cluster_num]
        
        # Calculate distances of all sentences in this cluster from the centroid.
        distances = euclidean_distances([centroid], cluster_embeddings)
        
                    # Find the index (in cluster_embeddings) of the sentence that is closest to the centroid.
        closest_sentence_idx = distances.argmin()

        # Map this index back to the original sentence index.
        original_sentence_idx = cluster_indices[closest_sentence_idx]
        
        # Set the closest sentence as the key sentence for this cluster.
        key_sentences[cluster_num] = sentences[original_sentence_idx]

    # Return the key sentences representing each cluster.
    return key_sentences





import math
## CLASSIFY CONTENT BY CODEBOOK (generated or provided)
def chunk_text(text: str, max_chunk_len: int) -> list[str]:
    all_tokens = BERT_tokenizer.tokenize(text)
    # Check if the text length is less than or equal to the chunk size

    # Initialize variables
    if len(all_tokens) <= max_chunk_len:
        return [text]



    sents =  [sent.text for sent in nlp(text).sents] if isinstance(text,str) else text
    
    # Calculate target chunk length 
    target_num_chunks = math.ceil(len(all_tokens) / max_chunk_len)
    target_chunk_len = math.ceil(len(all_tokens) / target_num_chunks)
    curr_chunk_tokens = 0
    curr_chunk_sents = []
    chunks = []

    for i, sent in enumerate(sents):
        
        sent_tokens = BERT_tokenizer.tokenize(sent)
        
        # Check if adding current sentence would exceed target length
        if curr_chunk_tokens + len(sent_tokens) > target_chunk_len:
            if curr_chunk_tokens + len(sent_tokens) < max_chunk_len:
                # Length with new sent is between target and max chunk 
                curr_chunk_sents.append(sent)
                # Add the current chunk with the new sentence, 
                chunks.append(str(" ".join(curr_chunk_sents)))
                curr_chunk_sents = []
                curr_chunk_tokens = 0
            else:
                # Length without new sent is between target and max chunk 
                # Finalize current chunk and add sent to next chunk list
                chunks.append(" ".join(curr_chunk_sents))
                # Add unused sentence to next chunk
                curr_chunk_sents = [sent] 
                # Reset current chunk token measure
                curr_chunk_tokens = len(sent_tokens)

    # Add any remaining sentences in final chunk
    if curr_chunk_sents:
        chunks.append(" ".join(curr_chunk_sents))

    return chunks

def get_embedding(text):
    inputs = BERT_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = BERT_embedding_model(**inputs)
    # Use the pooled output for sentence-level representation
    return outputs.pooler_output.squeeze().detach().numpy()

def normalize(values):
    """
    Normalizes a list of numerical values to a range between 0 and 1.
    
    Parameters:
        values (list of float): The list of numerical values to normalize.
        
    Returns:
        list of float: The list of normalized values.
    """
    if not values:  # Check if the list is empty
        return []

    min_value = min(values)
    max_value = max(values)

    # Avoid division by zero if all values in the list are the same
    if min_value == max_value:
        return [0.5 for _ in values]  # or return [1.0 for _ in values] depending on your choice

    # Apply min-max normalization
    return [(value - min_value) / (max_value - min_value) for value in values]
def classify_sentences(text, codebook=["categories","themes","values"], research_question="The statement favors", agreement_threshold=.5):
    # Options net_moral_representation and absolute_theme_presence require codebook to be a list of tuples, each containing a positive and negative theme string
    if isinstance(codebook[0], tuple):
            pos_codebook = [code for (code,_) in codebook]
            neg_codebook = [code for (_,code) in codebook]
            codebook = pos_codebook
            codebook.extend(neg_codebook)
    sentences =  [sent.text for sent in nlp(text).sents] if isinstance(text,str) else text
    sentences = [sent for sent in sentences if sent!="" and sent!=" "]
    num_sentences = max(len(sentences),1)
    print("NUM SENTENCES", num_sentences)
    avg_mean_difference = []
    sentence_score_differences = {}
    for sentence in sentences:
        sentence_nsp_scores = {}
        sentence_nll_scores = {}
        for category in codebook:
            # Concatenate sentence and hypothesis
            hypothesis = f"{research_question} {category}"  
            inputs = BERT_tokenizer.encode_plus(sentence, hypothesis, 
                                            return_tensors='pt', 
                                            padding='max_length', 
                                            truncation=True,
                                            max_length=512)
                                            
            # Get NSP scores from model  
            outputs = BERT_nsp_model(**inputs)
            logits = outputs.logits
            
            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=1)  
            
            # Extract probability of 'isNext' class
            nsp_score = probs[:,0].item()
            
            sentence_nsp_scores[category] = nsp_score
            

            ## Prepare text pair for BERT NSP
            #hypothesis = f"{research_question} {category}"
            #inputs = BERT_tokenizer.encode_plus(sentence, hypothesis, return_tensors='pt', padding=True, max_length=512, truncation=True)
            ## Get BERT NSP score
            #nsp_outputs = BERT_nsp_model(**inputs)
            #nsp_score = nsp_outputs.logits[:,0].detach().numpy()[0]  # Index 0 for 'isNext' label
            #sentence_nsp_scores[category] = nsp_score # The higher the better
            # GPT-2 NLL scoring
            combined_text = f"{sentence}\n{research_question} {category}. "
            gpt2_inputs = GPT2_tokenizer.encode(combined_text, return_tensors='pt')
            with torch.no_grad():
                outputs = GPT2_model(gpt2_inputs, labels=gpt2_inputs)
                # Apply softmax to get probabilities
            loss = outputs.loss.item()  # The lower, the better
#            probs = torch.softmax(loss, dim=1)  
            sentence_nll_scores[category] = - loss # negative loss , The higher the better
            # Normalize category preferences
            normalized_nsp_scores = normalize(list(sentence_nsp_scores.values()))
            normalized_nll_scores = normalize(list(sentence_nll_scores.values()))
            # Calculate the differences for each category
            score_differences = {category: abs(normalized_nsp - normalized_nll) 
                                for category, normalized_nsp, normalized_nll 
                                in zip(sentence_nsp_scores.keys(), normalized_nsp_scores, normalized_nll_scores)}
        mean_diff = np.mean([abs(score) for score in score_differences.values()])
        sentence_score_differences[sentence] = (mean_diff,score_differences)
        avg_mean_difference.append(mean_diff)

    avg_mean_difference = np.mean(avg_mean_difference)# find the average mean difference as a measure of avg classification agreement in all sentence:category data
    print("Avg mean diff", avg_mean_difference)
    # If the models agree on category scores for this sentence more than the average sentence, or above the provided agreement_threshold classify it.
    categorized_sentences = {}  
    for sentence in sentences:
        mean_diff,score_differences = sentence_score_differences[sentence]
        if mean_diff <= avg_mean_difference: # agreement threshold will find the average agreement to verify classification of half of the sentences 
            most_favored_category = max(score_differences, key=score_differences.get)
            categorized_sentences[sentence] = most_favored_category
            print("MeanDiff=", abs(mean_diff)," \n Sentence: ", sentence, " \n Classified as ", most_favored_category.split(" ")[0])
        else:
            print("Classification Models Disagree. Classification Skipped for:\n", sentence)
    category_counts = dict(collections.Counter(category for sentence, category in categorized_sentences.items()))
    category_distribution = calculate_percentages(category_counts)
    return category_distribution, avg_mean_difference

def calculate_theme_ranges(category_distribution, pos_codebook, neg_codebook):
        # we need to score each category with a positive theme and a negative theme
        pos_counts = {}
        neg_counts = {}
        scores= []
        for (category,count) in category_distribution.items():
            scores.append(count)
            if any(neg_codebook.keys() in category):
                neg_counts[category] = count
            elif any(pos_codebook.keys() in category):
                pos_counts[category] = count

        net_themes = {}
        for pos_code,neg_code in zip(pos_counts.keys(),neg_counts.keys()):
            # if absolute_theme_presence: #min(counts)<1
            #     total_counts[pos_code + '\n -- \n' + neg_code + '\n\n'] = pos_counts[pos_code] - neg_counts[neg_code] 
            # else:
            # if net_moral_representation: # net moral representation #combine positive themes minus negative themes to get polarity / theme preference
            # count = pos_counts[pos_code] + neg_counts[neg_code]
            net_themes[f"{pos_code}\n -  {neg_code}\n"] = (pos_counts[pos_code]), (neg_counts[neg_code])

        return net_themes
    
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency

def calculate_percentages(counts):
    total = sum(counts.values())
    percentages = {category: (count / total) * 100 for category, count in counts.items()}
    return percentages

def compare_distributions(orig_dist, summ_dist, summary_type="Summarized", name=""): 
        category_diffs=""
        for category in orig_dist.keys():
            orig_pct = orig_dist.get(category, 0)
            summ_pct = summ_dist.get(category, 0)
            category_diffs+= str(f"Category '{category.split()[0]}': Original = {orig_pct:.2f}%, {summary_type} {name} = {summ_pct:.2f}%")
            category_diffs+=f"\nPercentage difference {(summ_pct - orig_pct):.2f}%\n"
        results =   str(category_diffs)        
        minimum = 1 + abs(min(min(orig_dist.values()), min(summ_dist.values()), 0) ) # normalize min negative value to 0
        #Chi square does not allow negative values; normalize to 0
        normalized_orig = {key:num + minimum for key, num in orig_dist.items()}
        normalized_sum =  {key:num + minimum for key, num in summ_dist.items()}

        results += "\n" + perform_chi_square_test(normalized_orig, normalized_sum)

        if "Significant differences exist between the distributions (reject H0)" in results: 
            results += f"\n\nGENERATED A HEATMAP OF THE STATISTICAL SIGNIFICANCE FOR {summary_type} IN {name}\n"  + generate_chi_square_heatmap(normalized_orig, normalized_sum, name=name,summary_type = summary_type) + "\n\n"
        return str(results)

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
        results += str("No significant differences exist between the distributions (H0) null hypothesis\n")

    return results #chi2, p, dof, expected

def generate_chi_square_heatmap(original_counts, summarized_counts, name="", summary_type=""):
    """
    Generate a heatmap based on the chi-square test results for each category.
    
    Args:
    original_counts (dict): Category counts for the original text.
    summarized_counts (dict): Category counts for the summarized text.
    
    Returns:
    None
    """
    original_counts = {key.split()[0]:count for key,count in original_counts.items()}
    summarized_counts = {key.split()[0]:count for key,count in summarized_counts.items()}
    # Ensure both dictionaries have the same keys
    all_categories = [content for content in set(original_counts.keys()).union(set(summarized_counts.keys()))]

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
    plt.title(f'Chi-Square Test Heatmap for {summary_type} summarization of {name}')
    plt.yticks(rotation=0)  # Rotate labels
    plt.xticks(rotation=45, ha='right')  # Rotate labels
    plt.tight_layout()
    # Save or display the heatmap
    name=name+'_chi_square_heatmap.png' 
    plt.savefig(name, dpi=300)
    plt.show()
    return name

## SUMMARIZE TEXT TO COMPARE MODEL THEME SALIENCE 

from transformers import LlamaForCausalLM, LlamaTokenizer
def llama_summarize(input_text, summarization_ratio=.1,max_chunk_len=2500, command="Summary"):
    # ! Pip install huggingface-hub
    # $ huggingface-cli login
    # <token> 
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    # Create a bits and bytes config to quantize the 7b parameter models LLAMA and Mistral
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = LlamaForCausalLM.from_pretrained( 
            model_name,
          #  quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="cuda",

        )
    pipe = pipeline(
        "text-generation",  
        model=model,tokenizer=tokenizer,
        device_map="cuda",
        #trust_remote_code=True,
        torch_dtype=torch.bfloat16,  # Quantized bfloat for Mistral to work on consumer hardware
        # early_stopping=True, num_beams=2
    )
    pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
    pipe.model.config.pad_token_id = pipe.tokenizer.pad_token_id
 
    max_summary_len = max((len(tokenizer.tokenize(input_text)) * summarization_ratio), max_chunk_len/2)
    summary = input_text
    print("LLAMA to summarize:" )
    print(f"INPUT LENGTH TO LLAMA: {len(tokenizer.tokenize(input_text))}")
    while max_summary_len < len(tokenizer.tokenize(summary)):
        chunks = chunk_text(input_text, max_chunk_len)
        chunksum=""
        for chunk in chunks:
            # Tokenize the input chunk without padding and truncation to find the actual length of the chunk
            prompt = f"Summary: {chunk}\n{command}:"
            prompt_length = len(tokenizer.tokenize(prompt))
            outputs = pipe(
                prompt,
                max_new_tokens = min(max_summary_len,int(prompt_length/2)),
               # num_beams=4,
            )
            result = str((outputs[0]["generated_text"]) )
            result = " ".join( result.split(" ")[len(prompt.split(" ")):] ) if prompt in result else result
            chunksum += result
        summary = chunksum
        

        input_text = summary
    print(f"OUTPUT LENGTH OF LLAMA: {len(tokenizer.tokenize(summary))}")

    return summary
# !pip install autoawq
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
def mistral_summarize(text, summarization_ratio=.1,max_chunk_len=2650, command="Summary"):
#    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    #tokenizer = AutoTokenizer.from_pretrained(modelname)
    #model = AutoModelForCausalLM.from_pretrained(modelname).to(device)
    # ! apt-get install lzma
    # ! pip install bitsandbytes
    # Load the AWQ quantized model
    #model = AutoModelForCausalLM.from_pretrained(model_name)

    # Use the model with the text generation pipeline
   # text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    gc.collect()
    device = "cuda"
    #model_name = "TheBloke/Mistral-7B-Instruct-v0.1-AWQ"
    
        # Create a bits and bytes config to quantize the 7b parameter models LLAMA and Mistral
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
            model_name,
        #    quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
        )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    pipe = pipeline(
        "text-generation",  
        model=model,
        tokenizer=tokenizer,
        device_map="cuda",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,  # Quantized bfloat for Mistral to work on consumer hardware
        #early_stopping=True, num_beams=2
    )
    # pipe.tokenizer.to(device)
    max_summary_len = min((len(BERT_tokenizer.tokenize(text)) * summarization_ratio), 100)
    print("Mixtral to summarize:" )
   # tokenizer.pad_token_id = tokenizer.eos_token_id
    sumlen=len(BERT_tokenizer.tokenize(text))
    while max_summary_len < sumlen :
        chunks = chunk_text(text, max_chunk_len)
        print("MIXTRAL HAS THIS MANY CHUNKS:", len(chunks))
        chunksum = ""
        for chunk in chunks:
            prompt = f"Text: {chunk}\n{command}: "

            prompt_length=len(BERT_tokenizer.tokenize(prompt))
            outputs = pipe(
                prompt,
                max_new_tokens = min(max_summary_len,int(prompt_length/2)),
            #   num_beams=4,
                top_k=50,
            #   min_length=10
            #   top_p=0.95,
            )
            result = str((outputs[0]["generated_text"]) )
            result = " ".join( result.split(" ")[len(prompt.split(" ")):] ) if prompt in result else result
           # inputs['attention_mask'] = inputs['input_ids'].ne(tokenizer.pad_token_id).long()
            #generate_ids = model.generate(**inputs, max_new_tokens = int(prompt_len*0.5), min_length=0, early_stopping=True, num_beams=2)
            #result = tokenizer.decode(generate_ids[0], skip_special_tokens=True)[prompt_len:]

            print(result)

            print("LEN PROMPT", prompt_length)
            result_length = len(BERT_tokenizer.tokenize(result))
            print("LEN result", result_length)
            chunksum += result + " "
            assert(prompt_length > result_length)
        text = chunksum
        sumlen = len(BERT_tokenizer.tokenize(text))
        print("SUMLEN:", sumlen)
        assert (sumlen < max_summary_len/summarization_ratio)

    return text
#!pip install -q -U transformers
#!pip install -q -U accelerate
from transformers import pipeline

def phi2_summarize(input_text, summarization_ratio=.1, max_chunk_len=1350, command="task: **Rewrite** the above paragraph into a elementary school level textbook section while keeping as much of the lesson as possible"):
    model_name = "microsoft/phi-2"
    gc.collect()

    pipe = pipeline(
        "text-generation",  
        model=model_name,
        device_map="cuda",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,  #bfloat16
       # early_stopping=True, num_beams=4,
    )

    pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
    pipe.model.config.pad_token_id = pipe.tokenizer.pad_token_id
    max_summary_len = max((len(BERT_tokenizer.tokenize(input_text)) * summarization_ratio),max_chunk_len/2)
    summary = input_text
    sumlen = len(BERT_tokenizer.tokenize(summary))
    print("Phi-2 to summarize from tokenln:", sumlen)
    while max_summary_len < sumlen:
        starting_len = len(BERT_tokenizer.tokenize(summary))
        chunks = chunk_text(summary, max_chunk_len)
        result = ""
        for chunk in chunks:# if chunks else [summary]:
            # Tokenize the input chunk without padding and truncation to find the actual length of the chunk
            prompt = f"{chunk}\n{command}\n "
            prompt_length = len(pipe.tokenizer.tokenize(prompt))
            print("promptlen:", prompt_length)

            outputs = pipe(
                prompt,
                max_new_tokens = min(max_summary_len,int(prompt_length/2)),
                top_k=50,
                #num_beams=2,
                #do_sample=False, # temperature=0.7,
                #top_p=0.95,
            )
            chunksum = str(" ".join(outputs[0]["generated_text"].split(" ")[len(prompt.split(" ")):]) )
            print(f"chunksumlen: {len(pipe.tokenizer.tokenize(chunksum))}")
            print(chunksum)
            result += chunksum
            result_length = len(BERT_tokenizer.tokenize(chunksum)) 
            assert(0 < prompt_length - result_length)
        summary = result
        sumlen = len(BERT_tokenizer.tokenize(summary))
        print("\n Summary Length:", sumlen)
        print("\n Starting Length:", starting_len)
        print("\n Chunked Length:", len(BERT_tokenizer.tokenize("".join(chunks))))
        assert sumlen < starting_len

    return summary

from transformers import T5Tokenizer, T5ForConditionalGeneration
def t5_summarize(input_text: str, summarization_ratio: float = 0.2, max_chunk_len: int = 508,command="Summarize") -> str:
        tokenizer=T5Tokenizer.from_pretrained('t5-base')
        model=T5ForConditionalGeneration.from_pretrained('t5-base')
        # Validate input text
        if not input_text or not isinstance(input_text, str):
            raise ValueError("Input text must be a non-empty string")
        gc.collect()
        # Initialize variables
        summary = input_text
        orig_length = len(tokenizer.tokenize(input_text))

        max_summary_len = summarization_ratio * orig_length
        print(f"T5 to summarize \n full length:{orig_length}, \n  * Summarization Ratio: {summarization_ratio}\n           =     target_length: {max_summary_len}\n")
        while max_summary_len < len(tokenizer.tokenize(summary)):
            summary_text = ""
            chunks = chunk_text(summary, max_chunk_len)

            for chunk in chunks:
                    # Tokenize the input chunk without padding and truncation to find the actual length of the chunk
                input_text = f"Text: {chunk}\n{command}:"
                # Tokenize the input text
                input_tokens = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
                print("T5 chunk len prompt:", len((tokenizer.tokenize(input_text))))
                # Calculate the desired number of new tokens for the summary based on the summarization ratio and chunk length
                summary_length = len(input_tokens[0]) + int(summarization_ratio * len(input_tokens[0]))

                # Generate the summary
                summary_ids = model.generate(
                    input_tokens,
                    max_new_tokens=int(summary_length/2),
                    #max_length=min(max_summary_len,int(summary_length/2)),
                    min_length=0,
                    length_penalty=2.0,
                    num_beams=4,
                    early_stopping=True
                )
                # Decode the summary back to text
                result = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                summary_text += " " + result

            summary = summary_text
        print("T5 summarylen", len(summary.split()))
        return summary.strip()

import torch
def gpt2_summarize(input_text, summarization_ratio=.1, max_chunk_len=800, command="Summary"):
    """
    Summarizes the given text using the GPT-2 model.

    Parameters:
    - text (str): The text to summarize.
    - model_name (str): The GPT-2 model variant to use. Default is 'gpt2'.
    - max_length (int): The maximum length of the summary.
    Returns:
    - summaries (list of str): The generated summaries.
    """
    # Load pre-trained model and tokenizer
    model = GPT2_model #GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2_tokenizer # GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # Encode text input and add the EOS token
    max_summary_len = max((len(tokenizer.tokenize(input_text)) * summarization_ratio), max_chunk_len)
    summary = input_text
    sumlen=len(tokenizer.tokenize(summary))

    while max_summary_len < sumlen:
        chunks = chunk_text(input_text, max_chunk_len)
        chunksum=""

        for chunk in chunks:
          prompt = f"Text:{chunk}.\n{command}:"

          if len(tokenizer.tokenize(chunk)) > 10:
            input_ids = tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=True, padding=True)
            prompt_length=len(tokenizer.tokenize(prompt))
            #print(f"prompt length: {prompt_length}")
            # Generate summary output
            summary_ids = model.generate(input_ids,
                                        max_length = min(1000,int(prompt_length*(1.5))), 
                                        #do_sample=True,
                                        #temperature=.7,
                                        repetition_penalty=1.2,
                                        num_beams=4,
                                        length_penalty=2.0, 
                                        num_return_sequences=1, 
                                        early_stopping=True,
                                        eos_token_id=tokenizer.eos_token_id,
                                        pad_token_id=tokenizer.pad_token_id)
                                    #    top_k=50,
                                    #   top_p=0.95,
                                    #    no_repeat_ngram_size=2,
                                    #    truncation=True,
                                    #    attention_mask=input_ids['attention_mask'],
            output = tokenizer.decode(summary_ids[0][prompt_length:], skip_special_tokens=True) #str([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])
           # print(f"OUTPUT:  {output}")
            #print(f"summary length:  {len(tokenizer.tokenize(output))}")
            chunksum += output
        summary = chunksum
        sumlen=len(tokenizer.tokenize(summary))
            # Decode and return the summaries
        print("LEN SUM", sumlen, "is greater than max target len", max_summary_len)
    return summary


            
        
#import math
#from transformers import RobertaTokenizer, RobertaForCausalLM
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

def extractive_summary_roberta(text, summarization_ratio=0.1):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base')

#    tokenizer = RobertaTokenizer.from_pretrained('roberta-large') 
#    model = RobertaForSequenceClassification.from_pretrained('roberta-large')

    # Split the text into sentences or coherent chunks
    summary = ""


    print("ROBERTA:")
    chunks = chunk_text(text, 310)
    
    important_chunks = []
    importance_scores = []
    for chunk in chunks if chunks else [text]:
        print("roberta ",chunk)
        inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)
        
        # Assuming the second label corresponds to 'important'
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        importance_score = probabilities[:, 1]  # Assuming index 1 corresponds to 'important'
        importance_scores.append(importance_score.item())

    # Calculate dynamic threshold based on summarization ratio
    sorted_scores = sorted(importance_scores, reverse=True)
    num_top_chunks = int(len(chunks) * summarization_ratio)
    adaptive_threshold = sorted_scores[num_top_chunks - 1] if num_top_chunks > 0 else min(importance_scores) 

    # Select chunks based on the dynamic threshold
    selected_chunks = [(score, chunk) for score, chunk in zip(importance_scores, chunks) if score >= adaptive_threshold]
    bagged_sample = " ".join([(chunk) for score, chunk in zip(importance_scores, chunks) if score >= adaptive_threshold])
    summary = ""
    for chunk in chunks: # Put it back in chronological order
        if chunk in bagged_sample:
            summary+="\n " + chunk
    print("ROBERTA SUMMARIZED:")
    print(summary)

    return summary
  

    
    chunks = chunk_text(text, 210)  
    important_chunks = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        # Assuming the 'important' class is labeled as 1
        if predictions[0][1] > .005:  # This threshold can be adjusted
            important_chunks.append(chunk)

    # Concatenate selected chunks to form a summarized content
    summary = ". ".join(important_chunks[:int(len(important_chunks) * summarization_ratio)])
    return summary
def bart_summarize_text(input_text, summarization_ratio=.1):
    """
    Summarize the input text using the BART model. Ensure that no content is truncated and 
    the final summary does not exceed max tokens.

    Args:
    input_text (str): The input text to summarize.
    max_tokens (int): The maximum length of the summary.

    Returns:
    str: The summarized text.
    """
    num_output_summary_tokens = len(BERT_tokenizer.tokenize(input_text)) + 1  # Initialize to enter the while loop
    max_summary_len = int(num_output_summary_tokens * summarization_ratio)
    # Initialize conditions
    tokens_per_chunk = 512# 345  # 512 Determine the initial optimal chunk size
    BART_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    BART_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

    def bart(chunk, max_length=250):
        """
        Summarize the input text using the BART model.

        Args:
        text (str): The input text to summarize.
        model_name (str): The BART model name.
        max_tokens (int): The maximum length of the summary.

        Returns:
        str: The summarized text.
        """
        inputs = BART_tokenizer(chunk, return_tensors='pt', truncation=True, padding="max_length")
        summary_ids = BART_model.generate(inputs.input_ids, num_beams=4, max_new_tokens=max_length/2, min_length=0, early_stopping=True)
        summary = BART_tokenizer.decode(summary_ids[0], skip_special_tokens=True) 
        return summary

    current_text = input_text  # Initialize current_text to be the input_text
    max_tokens = tokens_per_chunk - 1 if tokens_per_chunk > max_summary_len else max_summary_len# make the loop condition smaller than attention window
    while num_output_summary_tokens > max_tokens:
        print(num_output_summary_tokens)
        # We chunk the current text (initially the input text, later could be the summary of the input text)
        chunks = chunk_text(current_text, tokens_per_chunk)

        # Summarize each chunk
        chunk_summaries = [bart(chunk) for chunk in chunks if len(chunk)>5]
        gc.collect()
        # Combine the chunk summaries
        current_text = ' '.join(chunk_summaries).strip()

        # Update the number of tokens in the current text (which is the total summary at this point)
        num_output_summary_tokens = len(BART_tokenizer.tokenize(current_text))
        
        # If the total summary length is within the limit, we break the loop, else we continue
        # The already summarized text (current_text) will be re-summarized in the next iteration if necessary
        # This way, we are always summarizing symmetrically, maintaining the context as much as possible
    gc.collect()
    return current_text

    # Return the final summary text, which is now guaranteed to be within the max_tokens limit

def agglomerative_sampling(original_text, summarization_ratio=0.2,model=None,subs=None):
        sentences = chunk_text(original_text, 100)

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


# Normalize the abstractive summary to the original text quotes
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np

def kmeans_centroid_sampling(original_text, summarization_ratio=.10):
    # Step 1: Preprocess the text to split it into sentences

    sentences = chunk_text(original_text, 100)
    # Step 2: Generate Sentence Embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(sentences)

    # Step 3: Apply K-Means Clustering
    n_clusters = max(1, int(len(sentences) * summarization_ratio))
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
    chunks = chunk_text(input_text, chunk_size)

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
def gpt4_summarize_text(input_text, max_tokens=32000, OPENAI_API_KEY="YOUR_OPENAI_KEY"):
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
        chunks = chunk_text(current_text, tokens_per_chunk)
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


def ground_to_knn_pairs(original, comparison, original_text_sentence_embeddings=None):
    """
    Ground each sentence of the summarized text to the closest original subtitle.
    
    Args:
    original_text (str): The original text.
    summarized_text (str): The summarized text.
    
    Returns:
    List[str]: A list of ground-truth normalized sentences.
    """
    original =  [sent.text for sent in nlp(original).sents] if isinstance(original,str) else original
    comparison =  [sent.text for sent in nlp(comparison).sents] if isinstance(comparison,str) else comparison

    assert(len(original) >= len(comparison))
    if original_text_sentence_embeddings:
        original_embeddings=original_text_sentence_embeddings
    else:
        original_embeddings = [get_embedding(sent) for sent in original]
    comparable_embeddings = [ get_embedding(sent) for sent in comparison]
# Initialize NearestNeighbors with n_neighbors set to the length of the original list.
    # This ensures we have enough neighbors to find unique matches.
    nbrs = NearestNeighbors(n_neighbors=len(original), metric='cosine').fit(original_embeddings)

    # Prepare the list to store the pairs and a set to track used original sentences.
    grounded_sentences = []
    available_indices = set(range(len(original)))  # All original indices are initially available.

    # Find nearest neighbors for each sentence in the comparison list.
    distances, indices = nbrs.kneighbors(comparable_embeddings)

    # Iterate through each set of neighbors found for each sentence in comparison.
    for comp_idx, neighbor_indices in enumerate(indices):
        # Go through the list of neighbors for the current comparable sentence.
        for neighbor_index in neighbor_indices:
            if neighbor_index in available_indices:
                # If this neighbor hasn't been used yet, create a pair.
                grounded_sentences.append((original[neighbor_index], comparison[comp_idx]))
                # Remove the index from available indices and break out of the loop.
                available_indices.remove(neighbor_index)
                break  # Move on to the next comparable sentence once a pair is made.

    # Return the list of unique nearest neighbor pairs.
    return grounded_sentences


## VISUALIZE THE DATA

import matplotlib.pyplot as plt

def generate_and_export_table(data_dict, sort_by=None, ascending=True, export_csv_path=None, export_img_path=None):
    # Convert the dictionary of dictionaries into a pandas DataFrame
    df = pd.DataFrame(data_dict)
    
    # Sort the DataFrame if a sort_by column is specified
    if sort_by and sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=ascending)
    
    # Plotting the DataFrame as a heatmap
    plt.figure(figsize=(10,len(df)+1))
    heatmap = sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".2f", cbar=False)
    plt.yticks(rotation=0)
    
    # Export the table image if an export path is provided
    if export_img_path:
        export_img_path +=".png" if ".png" not in export_img_path else export_img_path
        plt.savefig(export_img_path)
    plt.show()  # Show the plot after saving to avoid a blank file
    
    # Export the DataFrame to CSV if a file path is provided
    if export_csv_path:
        export_csv_path +=".csv" if ".csv" not in export_csv_path else export_csv_path
        df.to_csv(export_csv_path)
    return f"\nRESULTS TABLE:\n{export_img_path if export_img_path else ''} \n{export_csv_path if export_csv_path else ''}\n"

def bar_graph(full_codebook_scores, scoreboards, categories, file_name="bar_graph.png", title=""):
    """
    Generate a bar graph showing the range of thematic representation change for each category across various models.
    """

    def calculate_differences(original_range, model_range):
        """
        Calculate the percentage change for the min and max values of the model's range
        from the original's range.
        """
        orig_pos, orig_neg = original_range
        pos, neg = model_range
        return pos-orig_pos,neg-orig_neg

    # Setup figure and axes
    fig, ax = plt.subplots(figsize=(14, 8))

    # Determine the number of models and categories for plotting
    n_categories = len(categories)
    n_models = len(scoreboards)
    model_names = list(scoreboards.keys())

    # Calculate bar width and positions
    bar_width = 0.8 / n_categories
    indices = np.arange(n_models)

    # Plot data
    for i, category in enumerate(categories):
        category_values = []
        for model in model_names:
            original_range = full_codebook_scores[category]
            model_range = scoreboards[model][category]
            pos_percent_difference, neg_percent_difference = calculate_differences(original_range, model_range)
            category_values.append((pos_percent_difference, neg_percent_difference))

        # Calculate bottom and height for bars
        #bottoms = [val[0] for val in category_values]
        #heights = [val[1] - val[0] for val in category_values]
        heights = [val[0] for val in category_values]
        bottoms = [val[1] for val in category_values]

        # Plot bars for this category
        ax.bar(indices + i * bar_width, heights, bar_width, bottom=bottoms, label=category.split()[0])

    # Customize the plot
    ax.set_xlabel('Model')
    ax.set_ylabel('Change in Net Theme Values after Summarization ∆(pos - neg)')
    ax.set_title(title)
    ax.set_xticks(indices + bar_width * (n_categories / 2))
    ax.set_xticklabels(model_names)
    ax.legend(title="Category", loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, which='major', axis='y', linestyle='--')
    plt.xticks(rotation=45, ha='right')  # Rotate text type names for better visibility

    plt.tight_layout()
#    plt.savefig(file_name)
    plt.show()
    # Save and show the graph
    file_name = file_name if file_name.endswith(".png") else file_name + ".png"
    plt.savefig(file_name, format='png', dpi=300, bbox_inches='tight')

    
    return file_name

def theme_span_linegraph(full_codebook_scores, scoreboards, categories, file_name="line_graph.png", title=""):
    """
    Generate a scalable line graph showing the percentage difference for each category across various text types and save it as an image file. 
    Display the average absolute value percentage change for each summary type compared to the full_codebook_scores.

    Args:
    full_codebook_scores (dict): Category percentages for the original text.
    scoreboards (dict of dict): Dictionary of dictionaries, each representing category percentages for a specific summary type.
    categories (list): List of all categories.
    file_name (str): Name of the file to save the graph.
    title (str): The title of the content used for moral summary evaluation
    """
    # Initialize a dictionary to store percentage differences for each category across text types
    
    percentage_diffs = {category: {text_type: 0 for text_type in ['Original'] + list(scoreboards.keys())} for category in categories}
    
    # Calculate the percentage differences for each category across text types
    for category in categories:
        original_val = full_codebook_scores.get(category, 0)
       # percentage_diffs[category]['Original'] = 0  # No difference for the original
        for text_type, scores in scoreboards.items():
            val = scores.get(category, 0)
            percent_diff = val - original_val  # Calculate percentage difference
            percentage_diffs[category][text_type] = percent_diff  # Store the percentage difference
    
    # Create the line graph
    plt.figure(figsize=(12, 8))
    
    # Define text types including 'Original' for baseline comparison
    #text_types = ['Original'] + sorted(scoreboards.keys())
    text_types =   sorted(scoreboards.keys())
    
    # Plot lines for each text type across categories
    for text_type in text_types:
        # Extract data for the current text type across all categories
        text_type_data = [percentage_diffs[category][text_type] for category in categories]
        plt.plot([category.split()[0] for category in categories], text_type_data, label=text_type, marker='o')
    
    # Customize the graph
    plt.title(f'Comparison of Models\' Summary of: \n{title}')
    plt.xlabel('Theme')
    plt.ylabel('Change in Percent of Sentences Classified by Theme')
    plt.xticks(rotation=45, ha='right')  # Rotate category names for better visibility
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add grid for better readability
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place the legend outside the plot
    
    # Save and show the graph
    file_name = file_name if file_name.endswith(".png") else file_name + ".png"
    plt.savefig(file_name, format='png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return file_name

def model_span_linegraph(full_codebook_scores, scoreboards, categories, file_name="line_graph.png", title=""):
    """
    Generate a line graph showing the average absolute value percentage change for each model compared to the full_codebook_scores,
    with category-specific changes shown as scatter points.
    """
    # Initialize dictionaries to store the percentage differences and averages
    percentage_diffs = {text_type: {category: 0 for category in categories} for text_type in scoreboards.keys()}
    avg_percentage_diffs = {text_type: 0 for text_type in scoreboards.keys()}
    
    # Calculate the percentage differences for each category in each text type
    for category in categories:
        original_val = full_codebook_scores.get(category, 0)
        for text_type, scores in scoreboards.items():
            val = scores.get(category, 0)
            percent_diff = (val - original_val) / original_val * 100 if original_val != 0 else 0  # Adjust for percentage calculation
            percentage_diffs[text_type][category] = percent_diff
    
    # Calculate the average absolute percentage change for each model
    for text_type, cat_diffs in percentage_diffs.items():
        avg_percentage_diffs[text_type] = np.mean([abs(diff) for diff in cat_diffs.values()])

    # Sort models for plotting
    text_types = sorted(scoreboards.keys(), key=lambda x: avg_percentage_diffs[x])  # Sort based on average change

    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot scatter points for each category's change
    for category in categories:
        category_data = [percentage_diffs[text_type][category] for text_type in text_types]
        plt.scatter(text_types, category_data, label=category, marker='o')

    # Plot a line graph for the average change
    avg_data = [avg_percentage_diffs[text_type] for text_type in text_types]
    plt.plot(text_types, avg_data, label='Average Change', color='black', marker='D', linestyle='-', linewidth=2)

    # Customize the graph
    plt.title(title)
    plt.xlabel('Model Name')
    plt.ylabel('Percentage Change from Original')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Save and show the graph
    file_name = file_name if file_name.endswith(".png") else file_name + ".png"
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.show()

    return file_name

# Best line graph
def model_dif_scatterplot(full_codebook_scores, scoreboards, categories, file_name="line_graph.png",title=""):
    """ 
    Generate a scalable line graph showing the percentage difference for each category across various text types and save it as an image file. 
    Display the average absolute value percentage change for each summary type compared to the full_codebook_scores.

    Args:
    full_codebook_scores (dict): Category percentages for the original text.
    scoreboards (dict of dict): Dictionary of dictionaries, each representing category percentages for a specific summary type.
    categories (list): List of all categories.
    file_name (str): Name of the file to save the graph.
    title (str): The title of the content used for moral summary evaluation
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
    text_types = ["Original"] 
    text_types.extend(sorted(scoreboards.keys())) # Alphabetical order, to put _knn normalizations aside the abstract sums in the graph

    # No difference compared to baseline
    avg_percentage_diffs["Original"] = 0
    percentage_diffs["Original"] = {category:0 for category in categories}
    # Create the line graph
    plt.figure(figsize=(10, 6))
        # Plot lines for each category, following the sorted order of text_types
    for category in categories:
        category_data = [percentage_diffs[text_type][category] for text_type in text_types]
        plt.scatter(text_types, category_data, label=category, marker='o')
    # Plot the average points for each summary type, following the sorted order
    for i, text_type in enumerate(text_types):
    #    plt.scatter(text_type, avg_percentage_diffs[text_type], color='black', zorder=5)
        plt.text(text_type, avg_percentage_diffs[text_type], f'{avg_percentage_diffs[text_type]:.2f}%' if avg_percentage_diffs[text_type]>0 else '',  color='black')#,verticalalignment='bottom',)
    avg_data = [avg_percentage_diffs[text_type] for text_type in text_types]
    plt.plot(text_types, avg_data, label='Average Change', color='black', marker='D', linestyle='-', linewidth=2)

    # Customize the graph
    plt.title(f'Change in Thematic Distribution of Story Morals across Models\' Summary of: \n {title}')
    plt.xlabel('Text Type')
    plt.ylabel('Percentage Difference')
    plt.xticks(rotation=45, ha='right')  # Rotate text type names for better visibility
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add grid for better readability
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place the legend outside the plot
    file_name = file_name + ".png" if ".png" not in file_name else file_name
    plt.savefig(file_name, format='png', dpi=300, bbox_inches='tight')
    plt.show()
    return file_name
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import calinski_harabasz_score
from collections import OrderedDict
from nltk.corpus import stopwords
from nltk import word_tokenize

# pip install youtube_transcript_api
from youtube_transcript_api import YouTubeTranscriptApi
def load_and_preprocess_srt(file_path):
        subs = pysrt.open(file_path, encoding='iso-8859-1')
        texts = [' '.join(sub.text_without_tags.split()) for sub in subs]
        preprocessed_text = ' '.join(texts)
        return preprocessed_text

if __name__ == '__main__':

    research_question = "This passage represents a theme of  "
    videos=[]
    codebooks = []
    gospels_codebook = [
    ("Compassion: Demonstrating care and understanding towards others, acting on the principle of empathy.",
     "Apathy: Showing a lack of interest or concern for the hardships or suffering of others."),
    ("Mercy: Exercising forgiveness and leniency towards those who have erred or caused harm.",
     "Retribution: Advocating for punitive measures as a response to wrongdoing, justice."),
    ("Modesty: Exhibiting humility and a lack of arrogance in one's achievements and interactions with others.",
     "Arrogance: Displaying an exaggerated sense of one's own importance or abilities, often at the expense of others."),
    ("Trustworthiness: Being reliable and faithful in one's commitments and responsibilities.",
     "Skepticism: Exhibiting doubt and a critical questioning of motives, possibly leading to mistrust."),
    ("Altruism: Prioritizing the welfare of others and engaging in selfless acts of kindness.",
     "Egoism: Focusing on one's own interests and well-being without regard for the needs of others.")
]
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
    platos_republic_codebook = [
    ("Justice: Advocating for fairness, virtue, and the appropriate distribution of goods and responsibilities within the state and the individual.",
     "Injustice: The disruption of harmony and order, leading to social and personal discord and the prioritization of personal gain over the common good."),
    ("Wisdom: The pursuit and love of knowledge, particularly by the ruling class, guiding the state wisely and justly.",
     "Ignorance: The lack of knowledge or enlightenment, leading to poor judgment and the mismanagement of the state."),
    ("Virtue: The cultivation of moral and intellectual excellence, contributing to the well-being of the individual and the state.",
     "Vice: The presence of moral flaws or bad habits that corrupt the individual and, by extension, the society."),
    ("Enlightenment: The process of moving from ignorance to knowledge, symbolized by the ascent from the Cave into the light of the sun.",
     "Delusion: Remaining in the shadows, mistaking the illusions of the Cave for reality, and resisting the journey towards truth and enlightenment."),
    ("Philosopher-King: The ideal ruler who, possessing both wisdom and virtue, governs for the benefit of all, not personal gain.",
     "Tyranny: The rule by those who seek power for their own sake, leading to oppression and the degradation of society.")
]
    unified_moral_codebook = [
        ("Faith: Complete trust in moral and spiritual principles that guide ethical decisions and resilience.",
        "Skepticism: Critical questioning that, while valuable for insight, may hinder action without resolution."),
        ("Equity: Ensuring fair treatment and opportunity for all, emphasizing moral righteousness and social harmony.",
        "Bias: Prejudices that lead to unfair treatment, disrupting social equity and moral justice."),
        ("Enlightenment: Attaining profound insight and understanding, guiding ethical actions and moral judgments.",
        "Naivety: A lack of experience or understanding that results in poor judgment or vulnerability."),
        ("Altruism: Prioritizing the well-being of others, promoting communal harmony and mutual support.",
        "Egoism: Self-centeredness that overlooks the needs of others, often harming communal integrity."),
        ("Modesty: Acknowledging one's abilities and achievements without arrogance, fostering community respect.",
        "Vanity: Excessive pride in oneself that clouds judgment and alienates others."),
        ("Benevolence: Generous sharing of resources and kindness, enriching the community.",
        "Avarice: Greed for wealth or material gain, often compromising ethical standards and communal welfare."),
        ("Integrity: Consistent honesty and moral uprightness in all actions.",
        "Fraudulence: Deliberate deceit or trickery, undermining trust and ethical relationships."),
        ("Empathy: Understanding and sharing the feelings of others, driving compassionate actions.",
        "Apathy: Indifference to the experiences and suffering of others, neglecting moral duty."),
        ("Temperance: Self-restraint and moderation in all pursuits, supporting moral and spiritual well-being.",
        "Excess: Overindulgence that leads to moral lapse or harm to oneself and others."),
        ("Harmony: Working together for common goals, promoting peace and cooperative strength.",
        "Conflict: Disagreements or discord that fracture community unity and hinder collective progress.")
    ]    


    name = "The Allegory of the Cave" #TEST 6 minute video
    video_id = "OFylXQRbolM" #  "https://www.youtube.com/watch?v=OFylXQRbolM \n"
    out = '\n\n\n\n\n'+name + "\n"
    videos.append(video_id)
    out += evaluate_summarization_bias(name=name, content_id=video_id, research_question=research_question,codebook=gospels_codebook)

    name = "The Parables of Jesus Christ"
    video_id = "Ed41paFWSKM" # 60min Parables "https://www.youtube.com/watch?v=Ed41paFWSKM \n"
    out = '\n\n\n\n\n'+name + "\n"
    videos.append(video_id)
    out += evaluate_summarization_bias(name=name, content_id=video_id, research_question=research_question,codebook=gospels_codebook)
    with open(name.replace(" ", "").strip() + "_model_moralsum_results.txt", 'w') as f:
        f.write(out)
    print(video_id + "_model_moralsum_results.txt")

    gc.collect()

    name = "The Gospels of Matthew, Mark, Luke, & John"
    out += '\n\n\n\n\n'+name+"\n"
    video_id = "3UxowslJeTI" # 8 hours of the gospels  "https://youtu.be/3UxowslJeTI \n"
    out += evaluate_summarization_bias(name=name, content_id=video_id, research_question=research_question,codebook=gospels_codebook, summarization_ratio=1/50)
    videos.append(video_id)    
    with open(name.replace(" ", "").replace(",","").replace("&","").strip() + "_model_moralsum_results.txt", 'w') as f:
        f.write(out)
    print(video_id + "model_bias_results.txt")

    gc.collect()

    name = "Plato's Republic \n"
    out += '\n\n\n\n\n'+name+"\n"
    video_id = "CqGsg01ycpk" #4hr audiobook of the Republic #"https://youtu.be/CqGsg01ycpk \n"
    videos.append(video_id)
    out += evaluate_summarization_bias(name=name, content_id=video_id, research_question=research_question,codebook=platos_republic_codebook, summarization_ratio=1/20)
    with open(video_id + "model_bias_results.txt", 'w') as f:
        f.write(out)
    print(video_id + "_model_moralsum_results.txt")

    gc.collect()

    name = "Aesop's Fables"
    out += '\n\n\n\n\n'+name+"\n"
    video_id = "aaMLVsH6ikE" #aesops fables 3hr   "https://youtu.be/aaMLVsH6ikE \n"
    out += evaluate_summarization_bias(name=name, content_id=video_id, research_question=research_question,codebook=aesops_fables_codebook, summarization_ratio=1/15)
    videos.append(video_id)
    with open(name.replace(" ", "").replace(",","").strip() + "_model_bias_results.txt", 'w') as f:
        f.write(out)
    print(video_id + "_model_moralsum_results.txt")

    gc.collect()

    name = "101 Zen Stories and Koans"
    out += '\n\n\n\n\n'+name+"\n"
    video_id= "Y0p663Ot8mo" # zen koans 1hr  "https://youtu.be/Y0p663Ot8mo \n"
    videos.append(video_id)
    out += evaluate_summarization_bias(name=name, content_id=video_id, research_question=research_question,codebook=zen_koans_codebook)
    with open(video_id + "model_bias_results.txt", 'w') as f:
        f.write(out)
    print(video_id + "_model_moralsum_results.txt")

    gc.collect()

    name = "The Fables of the Panchatantra"
    out += '\n\n\n\n\n'+name+"\n"
    video_id = "lWnJem4hKxY" # 8 hours of the Panchatantra  "https://youtu.be/lWnJem4hKxY \n"
    out += evaluate_summarization_bias(name=name, content_id=video_id, research_question=research_question,codebook=unified_moral_codebook, summarization_ratio=1/50)
    videos.append(video_id)    
    with open(name.replace(" ", "").strip() + "_model_moralsum_results.txt", 'w') as f:
        f.write(out)
    print(video_id + "_model_moralsum_results.txt")

    gc.collect()
    
    name = "Full Catalog of Parables, Gospels, Koans, Allegories, and Fables"
    out = evaluate_summarization_bias(name=name, content_id=videos, research_question=research_question,codebook=unified_moral_codebook, summarization_ratio=1/100)
        
    with open("cumulative_model_moralsum_results.txt", 'w') as f:
        f.write(out)
    print("cumulative_model_moralsum_results.txt")
