import torch
import transformers
from transformers import BertModel, BertTokenizer, BartForConditionalGeneration, BartTokenizer, BertForNextSentencePrediction, GPT2Tokenizer, GPT2LMHeadModel
from sklearn.metrics.pairwise import euclidean_distances
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.stats import chi2_contingency
from sklearn.metrics import calinski_harabasz_score
from nltk.corpus import stopwords
from nltk import word_tokenize
import spacy
import numpy as np
import pandas as pd
import collections
from collections import OrderedDict
import pysrt
import gc
import math
import re
# for pulling text data from the internet
import requests
from bs4 import BeautifulSoup
# These are used for classification functions called for each model evaluation; load them once at start:
nlp = spacy.load("en_core_web_lg")

BERT_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
BERT_tokenizer.add_special_tokens({'pad_token': '[PAD]'}) 
BERT_embedding_model = BertModel.from_pretrained('bert-base-uncased')
#BERT_embedding_model.config.pad_token_id = BERT_tokenizer.pad_token_id
BERT_nsp_model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
BERT_nsp_model.config.pad_token_id = BERT_tokenizer.pad_token_id
GPT2_model = GPT2LMHeadModel.from_pretrained("gpt2") 
GPT2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def analyze_summarization_effects(name, content_id, research_question, codebook=None,summarization_ratio=.05, use_ensemble=False):
    resultfile = name.replace(" ", "").replace(",","").replace("'","").replace("&","").strip() + "_ModelMoralSum_results.md"
    with open(resultfile, 'w') as f:
        f.write(f"# A Comparison of Story-Moral Salience Across AI Models \nMeasuring Theme Representations in Model Summarization of: {name}\n\n")
    # Content can be either Youtube video id or .srt file path

    transcript = "" 
    content_title=""
    for content in content_id if isinstance(content_id,list) else [content_id]:
        if "https://" in content and "youtu.be/" not in content:
            try:
                print("Downloading data to summarize")
                content_title += f"[youtu.be/{content}](https://youtu.be/{content}) \n"
                response = requests.get(content)
                response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code
                soup = BeautifulSoup(response.content, 'html.parser')
                transcript += soup.get_text()
            except requests.RequestException as e:
                print(f"An error occurred: {e}")
                transcript += ""
        elif ".srt"  in content:
            content_title += content + "\n"
            print("SRT FILE?", content)
            transcript += " " + load_and_preprocess_srt(content)
        elif ".txt" in content:
            content_title += content + "\n"
            with open(content, 'r') as f:
                transcript += f.read()
        elif len(content.split())<100:    #Try as if its a youtube video id
            content = content if "youtu.be/" not in content else content.split("youtu.be/",1)[1].replace("/","")
            srt_dict = YouTubeTranscriptApi.get_transcript(content)
            transcript += ' '.join(sub['text'].replace("[Music]","") for sub in srt_dict)
            content_title += f"[youtu.be/{content}](https://youtu.be/{content}) \n"
        else:
            transcript += content

    with open(resultfile, "a+") as f:
        f.write(f"Summarize the Lessons of: {name}\nPrimary Source:{content_title}")

    if codebook and isinstance(codebook[0], tuple):
        pos_codebook, neg_codebook  = [], []
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
    with open(resultfile, "a+") as f:
        f.write("<br>Models Evaluated: LLAMA-7b-chat-hf, Mistral-7B-Instruct-v0.2, Gemma-2b-it, Phi-2, T5-base, Bart-large-cnn, gpt2, Roberta-base_NLL, BERT-large_K-Means_Clustering, and all-MiniLM-L6-v2_Agglomerative_Clustering <br>\n")
        f.write(f"<br>Tokenln Primary Source: {total_tokens} * Summarization Ratio: {summarization_ratio} = \n<br> Tokenln Per Model Summary:<{max_tokens}<br><br>")
        f.write(f"## Themes for Classification:<br> \n {str(codebook)}<br><br>\n\n")

    summaries = {}
    sentences =  [sent.text for sent in nlp(transcript).sents if sent.text != " "]
    transcript = ". ".join(sentences).strip().replace(". .",". ").replace("..",". ").strip()
    print("\n\n\nTranscript:\n"+transcript)
    # TO EVALUATE MORE MODELS: ADD THE MODEL SUMMARIES TO THE DICT LIKE THIS:
    # ANTHROPIC_KEY = 'YOUR_ANTHROPIC_KEY'
    # summaries['claude'] = claude_summarize_text(transcript, max_tokens=max_tokens, ANTHROPIC_KEY=ANTHROPIC_KEY)

    # OPENAI_APIKEY = "YOUR_OPENAI_KEY"
    # summaries['gpt4'] = gpt4_summarize_text(transcript, max_tokens=max_tokens, OPENAI_API_KEY=OPENAI_APIKEY)



    
    summaries['Bert-base_Kmeans'] = kmeans_centroid_sampling(transcript, summarization_ratio=summarization_ratio)
    gc.collect()
    torch.cuda.empty_cache()

    summaries["all-MiniLM-L6-v2_Agglomerative"] = agglomerative_sampling(transcript, summarization_ratio=summarization_ratio)
    gc.collect()
    torch.cuda.empty_cache()

    summaries["Roberta-base_nll"] = extractive_summary_nll(transcript, summarization_ratio=summarization_ratio)
    gc.collect()
    torch.cuda.empty_cache()

    summaries["GPT2_summary"] = gpt2_summarize(transcript, summarization_ratio=summarization_ratio)
    gc.collect()
    torch.cuda.empty_cache()

    summaries["t5-base_summary"] = t5_summarize(transcript, summarization_ratio=summarization_ratio)
    gc.collect()
    torch.cuda.empty_cache()

    summaries['bart-large-cnn_summary'] = bart_summarize_text(transcript, summarization_ratio=summarization_ratio)
    gc.collect()
    torch.cuda.empty_cache()

    summaries["Phi-2_summary"] = phi2_summarize(transcript, summarization_ratio=summarization_ratio)
    gc.collect()
    torch.cuda.empty_cache()

    # Gemma 7b gets killed on rtx3060 let's use gemma 2b with 2000 token window until some kind stranger buys me enough coffee to figure that out
    summaries["Gemma-2b-it_summary"] = gemma_summarize(transcript, summarization_ratio=summarization_ratio)
    gc.collect()
    torch.cuda.empty_cache()

    summaries["Llama-2-7b-chat-hf_summary"] = llama_summarize(transcript, summarization_ratio=summarization_ratio)
    gc.collect()
    torch.cuda.empty_cache()

    summaries["Mistral-7B-Instruct-v0.2_summary"] = mistral_summarize(transcript, summarization_ratio=summarization_ratio)
    gc.collect()
    torch.cuda.empty_cache()
 
    meandiffs={}
    primary_text_themes, average_mean_difference = classify_sentences(transcript, codebook, research_question)
    primary_theme_scores = get_category_distribution(primary_text_themes, codebook)
    meandiffs[name]=average_mean_difference
    with open(resultfile, "a+") as f:
        f.write(f"\n<br>### Source Theme Classification\n<br> Average Mean Difference for primary source text = {average_mean_difference}\nThis measures the confidence of BERT NSP and GPT2 for classifying with the codebook. Lower is better.")
        f.write("\n\n<br><br> Distribution of themes across sentences:\n")
        f.write("\n<br>".join([f"{score:.1f}  | {category} " for category,score in primary_theme_scores.items()]) + "<br><br>\n\n")

    if use_ensemble:
        summaries['Ensemble (abstract summaries)'] = "" #abstract_ensemble
        summaries['Ensemble (extract summaries)'] = "" #extract_ensemble 
    sentence_embeddings = [get_embedding(sentence) for sentence in sentences]
    gc.collect()
    torch.cuda.empty_cache()
    # Ground the ensemble to the primary source with k-nearest-neighbors
    knn_summaries = {}
    for summary_key in [key for key in summaries.keys()]: 

        if 'summary' in summary_key: # Skip the grounding of clustered embeddings, they're already extracts of the primary source quotes
            # Create extract summaries for non-knn summaries

            knn_extracted_sents,abstract_sents = ground_to_knn_pairs(transcript, summaries[summary_key], sentence_embeddings)
            knn_summaries[summary_key] = ". ".join(knn_extracted_sents).replace(". .",". ").replace("..",". ").strip()
            if use_ensemble:
                summaries['Ensemble (abstract summaries)'] += summaries[summary_key] + ". "
                summaries['Ensemble (extract summaries)'] += ". ".join(knn_extracted_sents) + ". " 



        else:
            if use_ensemble:
                summaries['Ensemble (extract summaries)'] += summaries[summary_key] + ". "
        # This works to generate sample quotes from each model, slightly redundant calculations if classification was first priority over model bias.
        # when dealing with classification for the study data we can store the scoreboard dicts in a database with mean-diff on ensemble codebook classifications, enabling you to remove this logic and pull the sentences from vector db relating to each theme.
    for summary_key in [key for key in summaries.keys()]: 
        print("Top Coded Sentences from", summary_key)
        representative_sentences = bert_kcentroid_quotes(summaries[summary_key].replace(". .",". "), num_quotes=(2*len(codebook))) #half will be filtered out by classification reranking
        categories, quotes = ground_to_knn_pairs(codebook, representative_sentences)
        categorized_quotes,mean_diff = classify_sentences(quotes,categories, research_question) # second value returned is the codebook:quote dict
        meandiffs[summary_key] = mean_diff
        with open(resultfile, "a+") as f:
            f.write(f"<br><br>\n\nAverage Mean Difference for {summary_key}: {mean_diff}\n<br>")
            for i,(quote,category) in enumerate(categorized_quotes.items()):
                f.write(f"    {i+1} | {category.split(':')[0]} <br>\n{quote}\n<br>" )# These groups could each be added to the nearest neighbor category to better classify, at cost of context window. ##TODO future research
        gc.collect()
        torch.cuda.empty_cache()
 
    summary_scores = {}
    # For each summary algorithm, summarize transcript and create a scoreboard as a list of sentence classification frequencies
    with open(resultfile, "a+") as f:
            f.write(f"\n\n\n ### Summary Theme Quantification\n Each BERT and GPT sentence embedding compares to a research question and category to classify.\n Research Question:\n {research_question}:\n")
    for (summary_key, summary_text) in summaries.items(): # Graphing all the knn grounded summaries gets messy and out of scope of the primary research question
            categorized_sentences, average_mean_difference = classify_sentences(summary_text, codebook, research_question)
            summary_scores[summary_key] = get_category_distribution(categorized_sentences, codebook)
            meandiffs[summary_key]=average_mean_difference
            with open(resultfile, "a+") as f:
                f.write(f"#### {summary_key}:<br>\n Average Mean Difference = {average_mean_difference}\nTheme Distribution Scores:\n<br>" + str(summary_scores[summary_key])+"<br><br>\n\n")
    gc.collect()
    torch.cuda.empty_cache()
    scored_differences={}
#   Compare the distribution of sentence categories applied in the summary to the original source distribution
    for summary_key, summary_scoreboard in summary_scores.items():
        with open(resultfile, "a+") as f:
            f.write(compare_distributions(primary_theme_scores, summary_scoreboard, name = name, summary_type = summary_key)+"<br><br>\n\n")
        scored_differences[summary_key] = {category:sumscore-origscore for category,sumscore in summary_scoreboard.items() for theme,origscore in primary_theme_scores.items()}
    gc.collect()
    torch.cuda.empty_cache()

    # Print summaries onto results txt after we have already found the best representative sentences for the codebook and outlined the data
    with open(resultfile, "a+") as f:
        f.write( f'\n\n\n<br><br><br>## Graphing the change in theme distributions across models:<br><br>\n\n')
        mean = list(meandiffs.values())
        mean = sum(mean)/len(mean)
        heatmapfile,csvfile = generate_heatmap(scored_differences,name=name)
        
        f.write( f"\n\n\n<br><br><br> ### Resulting Average Theme:Content Mean Difference Across Models: {mean}\n\n<br><br> Table:<br>\n{csvfile}<br><br>\n\n![Heatmap of Theme Differences:]({heatmapfile})<br>\n")
        f.write("![Theme Preferences Linegraph:](" + theme_span_linegraph(primary_theme_scores, summary_scores, file_name = f"{name}_theme_evaluation", title=name) +")<br>\n")
        f.write("![Model Difference Scatterplot:](" + model_dif_scatterplot(primary_theme_scores, summary_scores, file_name = f"{name}_model_mean-diff_scatterplot", title=name) +")<br>\n")
    # Graph the net polarity of themes, score each positive theme minus each negative theme to determine polarity of summary theme representation
    if pos_codebook: 
        print(pos_codebook,neg_codebook)
        print(summary_score for key,summary_score in summary_scores.items())
        fulltext_score_ranges = calculate_theme_ranges(primary_theme_scores, pos_codebook, neg_codebook)
        scoreboard_ranges = {key:calculate_theme_ranges(summary_score, pos_codebook, neg_codebook) for key, summary_score in summary_scores.items()}

        categories = [key for key in fulltext_score_ranges.keys()]
        for key,scoreboard in scoreboard_ranges.items():
            categories.extend([key for key in scoreboard.keys()])

        categories = list(set(categories))
        net_scores_base = {category:pos-neg for category, (pos,neg) in fulltext_score_ranges.items()}
        net_scoreboards = {scoreboardkey:{category:pos-neg for category, (pos,neg) in scoreboard.items()} for scoreboardkey,scoreboard in scoreboard_ranges.items()}
        with open(resultfile, "a+") as f:
            f.write("![Net Theme Bargraph:](" + bar_graph(fulltext_score_ranges, scoreboard_ranges, categories, file_name = str(name + "_net_model_theme_bars"), title=name) +")<br>\n")
            f.write("![Net Theme Linegraph:](" + theme_span_linegraph(net_scores_base, net_scoreboards, file_name = str(name+"_net_theme_polarity_lines"), title=name) +")<br>\n")

            f.write("\n\n\n\n\n## MODEL SUMMARIES TO AUDIT:")
            for model,summary in summaries.items():
                f.write(f"\n\n### Summary of {name} from {model} with a mean differnce of {meandiffs[model]}<br>\n{summary}")
    print(f"\n\n\nRESULTFILE:\n{resultfile}")
    with open(resultfile, "r") as f:
        return f.read()


def bert_kcentroid_quotes(text, num_quotes=5):
    tokenizer = BERT_tokenizer #BertTokenizer.from_pretrained('bert-base-uncased')
    model = BERT_embedding_model #BertModel.from_pretrained('bert-base-uncased')
    device="cpu"
    BERT_embedding_model.to(device)  # Or 'cuda' if GPU is available and preferable

    gc.collect()
    if isinstance(text, str):
        # Preprocess the text data
        sentences = chunk_sents_from_text(text, 100)#[sent.text for sent in nlp(transcript).sents]
    else:
        sentences = text
    print("KcentroidQuoting")
    num_quotes = max(1,min(num_quotes,len(sentences)))
    # Tokenize and encode the sentences using BERT.
    inputs = tokenizer(sentences, padding=True, truncation=True, max_length=510, return_tensors="pt")
    inputs = {key: val.to(device) for key, val in inputs.items()}  # Or 'cuda'

    # Generate sentence embeddings.
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()

    # Apply K-means clustering on the embeddings.
    kmeans = KMeans(n_clusters = max(min(len(sentences), num_quotes), 1), random_state=0).fit(embeddings)

    # Initialize a list to hold the key sentences for each cluster.
    key_sentences = [None] * num_quotes
    
    # For each cluster, identify the sentences that belong to it and find the closest one to the centroid.
    for cluster_num in range(num_quotes):
        gc.collect()
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


def classify_sentences(text, codebook=["categories","themes","values"], research_question="The statement favors"):
    if isinstance(codebook[0], tuple):
            pos_codebook = [code for (code,_) in codebook]
            neg_codebook = [code for (_,code) in codebook]
            codebook = pos_codebook
            codebook.extend(neg_codebook)
    # sentences =  [sent.text for sent in nlp(text).sents] if isinstance(text,str) else text
    # sentences = [sent for sent in sentences if sent!="" and sent!=" "]
    sentences = chunk_sents_from_text(text, 200) if isinstance(text,str) else text
    sentences = [sentence for sentence in sentences if sentence]
    transformers.logging.set_verbosity_error() # Silence warning if BART NSP overflows to truncate
    num_sentences = max(len(sentences),1)
    print("NUM CHUNKS TO CLASSIFY", num_sentences)
    avg_mean_difference = []
    sentence_score_differences = {}
    hypotheses_losses = {} # We don't recalculate the losses that we need to reuse
    for sentence in sentences:
        sentence_nsp_scores = {}
        sentence_nll_scores = {}
        
        gc.collect()
        torch.cuda.empty_cache()

        for category in codebook:
            # Concatenate sentence and hypothesis
            hypothesis = f"{research_question} {category}"
            inputs = BERT_tokenizer.encode_plus(sentence, hypothesis, 
                                            return_tensors='pt', 
                                            padding='max_length', 
                                            truncation=True,#'longest_first',
                                            max_length=512)

            # Get NSP scores from model  
            outputs = BERT_nsp_model(**inputs)
            logits = outputs.logits
            
            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=1)  
            
            # Extract probability of 'isNext' class
            nsp_score = probs[:,0].item()
            
            sentence_nsp_scores[category] = nsp_score

            # GPT-2 NLL scoring
            combined_text = f"{sentence}\n{category}. "


            gpt2_combined_inputs = GPT2_tokenizer.encode(combined_text, return_tensors='pt', max_length=512, truncation=True)
            gpt2_sentence_inputs = GPT2_tokenizer.encode(sentence, return_tensors='pt', max_length=512, truncation=True)
            gpt2_hypothesis_inputs = GPT2_tokenizer.encode(hypothesis, return_tensors='pt', max_length=512, truncation=True) if hypothesis not in hypotheses_losses.keys() else hypotheses_losses[hypothesis]

            # Calculate NLL for each context
            with torch.no_grad():
                combined_loss = GPT2_model(gpt2_combined_inputs, labels=gpt2_combined_inputs).loss.item()
                sentence_loss = GPT2_model(gpt2_sentence_inputs, labels=gpt2_sentence_inputs).loss.item()
                hypothesis_loss = GPT2_model(gpt2_hypothesis_inputs, labels=gpt2_hypothesis_inputs).loss.item()

            # Logic for comparing NLL scores
            # Here we could calculate how the addition of the category impacts the NLL relative to the base contexts
            # This can involve direct comparisons or more sophisticated statistical measures
            context_loss = (sentence_loss + hypothesis_loss) / 2  # Average loss for base contexts
            impact_of_category_loss = combined_loss - context_loss  # Measure the impact of adding the category, the lower the better match

            # Interpreting the impact score:
            # A lower or negative impact score suggests the category is well aligned with the sentence and research question.
            # A higher impact score suggests a discrepancy or lower relevance.
            sentence_nll_scores[category] = - impact_of_category_loss #impact_score # negative loss , The higher the better


        
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

    avg_mean_difference = np.mean(avg_mean_difference) # find the average mean difference as a measure of avg classification agreement in all sentence:category data
    print("Avg mean diff", avg_mean_difference)
    # If the models agree on category scores for this sentence more than the average sentence, or above the provided agreement_threshold classify it.
    categorized_sentences = {}  
    for sentence in sentences:
        mean_diff,score_differences = sentence_score_differences[sentence]
        if mean_diff <= avg_mean_difference: # agreement threshold will find the average agreement to verify classification of half of the sentences 
            most_favored_category = max(score_differences, key=score_differences.get)
            categorized_sentences[sentence] = most_favored_category
            print("MeanDiff=", abs(mean_diff),"  Agreed to Classify ", most_favored_category.split(" ")[0], "\nSentence: ", sentence, " \n\n")
        else:
            print("Models Disagree; Classification Skipped for sent:\n", sentence)
    return categorized_sentences, avg_mean_difference

def get_category_distribution(categorized_sentences, codebook):
    counts = dict(collections.Counter(category for sentence, category in categorized_sentences.items()))
    total = sum(counts.values())
    percentages = {category: (count / total) * 100 for category, count in counts.items()}
    for code in codebook:
        if code not in percentages:
            percentages[code] = 0 
    return percentages

def calculate_theme_ranges(category_distribution, pos_codebook, neg_codebook):
        # we need to score each category with a positive theme and a negative theme
        pos_counts = {}
        neg_counts = {}
        for (category,count) in category_distribution.items():
            if category in neg_codebook:
                neg_counts[category] = count
            elif category in pos_codebook:
                pos_counts[category] = count
        net_themes = {}
        for pos_code,neg_code in zip(pos_codebook,neg_codebook):
            if pos_code not in pos_counts:
                pos_counts[pos_code] = 0
            if neg_code not in neg_counts:
                neg_counts[neg_code] = 0

            net_themes[f"{pos_code}\n -  {neg_code}"] = ((pos_counts[pos_code], neg_counts[neg_code]))
        return net_themes



## SUMMARIZE TEXT TO COMPARE MODEL THEME SALIENCE 

from transformers import LlamaForCausalLM, LlamaTokenizer
def llama_summarize(input_text, summarization_ratio=.1,max_chunk_len=2600, command="Task: Briefly Summarize the lesson of the story as a concise summary paragraph"):
    # ! Pip install huggingface-hub
    # $ huggingface-cli login
    # <token> 
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    # Create a bits and bytes config to quantize the 7b parameter models LLAMA and Mistral
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False
    )
    
    model = LlamaForCausalLM.from_pretrained( 
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="cuda",

        )
    pipe = pipeline(
        "text-generation",  
        model=model,tokenizer=tokenizer,
        device_map="cuda",
        #trust_remote_code=True,
        torch_dtype=torch.bfloat16,  # Quantized bfloat for llama to work on consumer hardware rtx3060
        # early_stopping=True, num_beams=2
    )
 
    max_summary_len = max((len(tokenizer.tokenize(input_text)) * summarization_ratio), 50)
    summary = input_text
    sumlen = len(tokenizer.tokenize(summary))
    print(f"INPUT TO LLAMA: len{len(tokenizer.tokenize(input_text))}")
    while max_summary_len < sumlen:
        chunks = chunk_sents_from_text(summary, max_chunk_len)
        print(f"Llama has {len(chunks)} chunks")
        chunksum=""
        

        for chunk in chunks:
            gc.collect()
            torch.cuda.empty_cache()
            # Tokenize the input chunk without padding and truncation to find the actual length of the chunk
            prompt = f"{chunk} **{command}**:"
            prompt_length = len(tokenizer.tokenize(prompt))
            outputs = pipe(
                prompt,
                max_new_tokens = min(max_summary_len,int(prompt_length/2)),
            )

            result = str((outputs[0]["generated_text"]) ) 
            result = result.split(prompt)[1] if prompt in result else result
            chunksum += result.strip() + ". "
            print(result)
        summary = chunksum.strip().replace(". .",". ").replace("..",". ")
        sumlen = len(tokenizer.tokenize(summary))

    print(f"OUTPUT OF LLAMA: len{len(tokenizer.tokenize(summary))}")

    return summary.strip().replace("\n",".").replace("..",". ").replace(". .",". ").strip()

 
# !pip install autoawq
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
def mistral_summarize(text, summarization_ratio=.1,max_chunk_len=2650, command="Task: Summarize the lesson from the text into a concise tl;dr summary paragraph"):
    # ! apt-get install lzma
    # ! pip install bitsandbytes
    #!pip install -q -U accelerate
        # Create a bits and bytes config to quantize the 7b parameter models LLAMA and Mistral
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False
    )

    
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
        )



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
    max_summary_len = (len(BERT_tokenizer.tokenize(text)) * summarization_ratio) 
   # tokenizer.pad_token_id = tokenizer.eos_token_id
    summary=text
    sumlen=len(BERT_tokenizer.tokenize(summary))
    print("Mistral to summarize: len",sumlen, " tokens" )
    while max_summary_len < sumlen :
        chunks = chunk_sents_from_text(summary, max_chunk_len)
        print(f"Mistral has {len(chunks)} chunks of apx{max_chunk_len} tokens to summarize")
        chunksum = ""

        for chunk in chunks:
            gc.collect()
            torch.cuda.empty_cache()
            prompt = f"Text: {chunk} **{command}**: "

            prompt_length=len(BERT_tokenizer.tokenize(prompt))
            outputs = pipe(
                prompt,
                max_new_tokens = min(max_summary_len,int(prompt_length/2)),
            #   num_beams=4, # length_penalty = 5.0, early_stopping=True
                top_k=50, min_length=50,
            #   min_length=10
            #   top_p=0.95,
                

            )
            result = str((outputs[0]["generated_text"]) )
            result = result.split(prompt)[1] if prompt in result else result
            print(result)

            chunksum += result + ". "

        summary = chunksum.replace("..",".").replace(". .",". ")
        sumlen = len(BERT_tokenizer.tokenize(summary))
        print("Mixtral Summary: len", sumlen)

    return summary.strip().replace("\n",".").replace("..",". ").replace(". .",". ")

from transformers import pipeline
def phi2_summarize(input_text, summarization_ratio=.1, max_chunk_len=1348, command="Task: Briefly **summarize** the key lessson of the text into a concise one-paragraph conclusion for an elementary school textbook."):
    model_name = "microsoft/phi-2"
    pipe = pipeline(
        "text-generation",  
        model=model_name,
        device_map="cuda",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,  
       # early_stopping=True, num_beams=4,
    )
    pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
    pipe.model.config.pad_token_id = pipe.tokenizer.pad_token_id
    max_summary_len = (len(BERT_tokenizer.tokenize(input_text)) * summarization_ratio)
    summary = input_text
    sumlen = len(BERT_tokenizer.tokenize(summary))
    print("Phi-2 to summarize: len", sumlen)
    while max_summary_len < sumlen:
        starting_len = len(BERT_tokenizer.tokenize(summary))
        chunks = chunk_sents_from_text(summary, max_chunk_len)
        chunksum = ""
        print(f"Phi2 has {len(chunks)} chunks")

        for chunk in chunks:
            gc.collect()
            torch.cuda.empty_cache()
            # Tokenize the input chunk without padding and truncation to find the actual length of the chunk
            prompt = f"{chunk}. {command}: "
            prompt_length = len(pipe.tokenizer.tokenize(prompt))
            outputs = pipe(
                prompt,
                max_new_tokens = int(prompt_length/2),
                #max_length = int(prompt_length/2),
                top_k=50, # Lower value filters a focused interpretation of the text
                top_p=0.8, # lower values provide more focused and determistic generation, too low becomes a parrot, higher values are more diverse
                #num_beams = 4,  #early_stopping=True, # length_penalty = 8.0
                no_repeat_ngram_size=6,
                do_sample=True, min_length=25,
                temperature=0.1, # Prevent parroting loops, still somewhat deterministic
            )

            outputs = outputs[0]["generated_text"]
            result = outputs.split(prompt)[1] if prompt in outputs else outputs

            result = result.replace("Task: Briefly **summerize** the key lesson of the text","").replace("into a one-paragraph conclusion for a textbook.","").replace("Title:","").replace("Task:","").replace("task:","").replace("**Rewrite**","").replace("into a elementary school level textbook","").replace("the above paragraph","").replace("Answer:","").replace("Answers:","").replace("Question:","").replace("Questions:","").replace("Exercise:","").replace("Exercises:","").replace("##Your","").replace("section while keeping as many content as possible","").replace("Follow-Up","").replace("In the text,","").replace("Follow-up","").replace("Follow-ups","").replace("Introduction:","")
            pattern = r"(Answer|Answers|Question|Questions|Exercise|Exercises|Use Case|Follow-Up|Follow-Ups|Task|Chapter) \d+:"
            result = re.sub(pattern, "", result)


            print(result)
            chunksum +=  result.replace("\n",". ") 
        summary = chunksum.replace("..",".").replace(". .",". ")
        sumlen = len(BERT_tokenizer.tokenize(summary))
        print("\n Phi-2 Summary: len", sumlen)
    return summary.strip().replace("\n",".").replace(". .",". ").replace("..",". ")
        #print("\n Starting Length:", starting_len)
        #print("\n Chunked Length:", len(BERT_tokenizer.tokenize("".join(chunks))))

#from transformers import AutoTokenizer, AutoModelForCausalLM

from transformers import AutoTokenizer, GemmaForCausalLM
def gemma_summarize(input_text, summarization_ratio=.1, max_chunk_len=1500, command="Summarize"):
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
    model = GemmaForCausalLM.from_pretrained("google/gemma-2b-it")
    model_name = "google/gemma-2b-it"

    pipe = pipeline(
        "text-generation",  
        model=model,tokenizer=tokenizer,
        device_map="cuda",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,  
    )
    pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
    pipe.model.config.pad_token_id = pipe.tokenizer.pad_token_id
    max_summary_len = max(100,len(BERT_tokenizer.tokenize(input_text)) * summarization_ratio)
    summary = input_text
    sumlen = len(BERT_tokenizer.tokenize(summary))
    print("gemma-2 to summarize: len", sumlen)
    while max_summary_len < sumlen:
        chunks = chunk_sents_from_text(summary, max_chunk_len)
        chunksum = ""
        print(f"gemma has {len(chunks)} chunks")
        for chunk in chunks:
          
            gc.collect()
            torch.cuda.empty_cache()
            # Tokenize the input chunk without padding and truncation to find the actual length of the chunk
            prompt = f"{chunk}. **{command}**: "
            prompt_length = len(pipe.tokenizer.tokenize(prompt))
            outputs = pipe(
                prompt,
                max_new_tokens = min(8192 - (prompt_length), prompt_length/2),
                #max_length = int(prompt_length/2),
                top_k=50, # Lower value filters a focused interpretation of the text
                top_p=0.8, # lower values provide more focused and determistic generation, too low becomes a parrot, higher values are more diverse
                no_repeat_ngram_size=6,
                do_sample=True,
                temperature=0.2,
                min_length=50
            )
            outputs = outputs[0]["generated_text"]
            print("GEN")
            result = outputs.split(prompt)[1] if prompt in outputs else outputs
            print(result)
            chunksum +=  result + ". "
        summary = chunksum
        sumlen = len(BERT_tokenizer.tokenize(summary))
        print("\n gemma-2 Summary: len", sumlen)
        #print("\n Starting Length:", starting_len)
        #print("\n Chunked Length:", len(BERT_tokenizer.tokenize("".join(chunks))))

    return summary.strip().replace("\n",".").replace(". .",". ").replace("..",". ")

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
    # Initialize conditions
    tokens_per_chunk = 512# 345  # 512 Determine the initial optimal chunk size
    BART_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    BART_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    print("BART to Summarize")
    def bart(chunk, max_length=500):
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
        summary_ids = BART_model.generate(inputs.input_ids, num_beams=4, max_new_tokens=max_length/2, min_length=5, early_stopping=True)
        summary = BART_tokenizer.decode(summary_ids[0], skip_special_tokens=True) 
        print(summary)
        return summary
    sumlen = len(BERT_tokenizer.tokenize(input_text)) + 1  # Initialize to enter the while loop
    max_summary_len = int(sumlen * summarization_ratio)
    summary = input_text  # Initialize current_text to be the input_text
    max_tokens = tokens_per_chunk - 1 if tokens_per_chunk > max_summary_len else max_summary_len# make the loop condition smaller than attention window
    while max_tokens < sumlen:
        print(sumlen)
        # We chunk the current text (initially the input text, later could be the summary of the input text)
        chunks = chunk_sents_from_text(summary, tokens_per_chunk)
        print(f"BART has {len(chunks)} chunks")
        # Summarize each chunk
        chunk_summaries = [bart(chunk) for chunk in chunks if len(chunk)>5]
        gc.collect()
        # Combine the chunk summaries
        summary = ' '.join(chunk_summaries).strip()
        print(chunk_summaries)
        # Update the number of tokens in the current text (which is the total summary at this point)
        sumlen = len(BART_tokenizer.tokenize(summary))
        # If the total summary length is within the limit, we break the loop, else we continue
        # The already summarized text (summary) will be re-summarized in the next iteration if necessary
        # This way, we are always summarizing symmetrically, maintaining the context as much as possible
        gc.collect()
        torch.cuda.empty_cache()
    return summary.strip().replace("\n",". ").replace(". .",".").replace("..",". ")

from transformers import T5Tokenizer, T5ForConditionalGeneration

def t5_summarize(input_text: str, summarization_ratio: float = 0.2, max_chunk_len: int = 306,command="Summarize:") -> str:
        tokenizer=T5Tokenizer.from_pretrained('t5-base')
        model=T5ForConditionalGeneration.from_pretrained('t5-base')
        # Validate input text
        if not input_text or not isinstance(input_text, str):
            raise ValueError("Input text must be a non-empty string")
        gc.collect()
        # Initialize variables
        summary = input_text
        orig_length = len(tokenizer.tokenize(input_text))
        sumlen = len(tokenizer.tokenize(summary))
        max_summary_len = summarization_ratio * orig_length
        print(f"T5 to summarize: len{orig_length}")
        while max_summary_len < sumlen:
            chunksum = ""
            chunks = chunk_sents_from_text(summary, max_chunk_len)
            print(f"T5 has {len(chunks)} chunks")
            for chunk in chunks:
                    # Tokenize the input chunk without padding and truncation to find the actual length of the chunk
                input_text = f"{chunk}\n{command}:"
                # Tokenize the input text
                input_tokens = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
                print("T5 chunk len prompt:", len((tokenizer.tokenize(input_text))))
                # Calculate the desired number of new tokens for the summary based on the summarization ratio and chunk length
                #promptlen =  len(input_tokens[0]) # Summarize by no less than .5 each time for higher fidelity

                # Generate the summary
                summary_ids = model.generate(
                    input_tokens,
                    #max_length=500, #int(promptlen*1.5),
                    max_new_tokens=200,#min(max_summary_len,int(summary_length/2)),
                    min_length=25,
                    length_penalty=5.0,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=5,
                )
                # Decode the summary back to text
                result = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                print(result)
                chunksum += ". " + result
                gc.collect()
                torch.cuda.empty_cache()
            summary = chunksum.replace(". .",". ").replace("..",".")
            sumlen = len(tokenizer.tokenize(summary))

        print("T5 Summary: len", len(summary.split()))

        return summary.strip().replace("\n",". ").replace(". .",".").replace("..",". ")

def gpt2_summarize(input_text, summarization_ratio=.1, max_chunk_len=550, command=". In summary, the main lesson is"):
    """
    Summarizes the given text using the GPT-2 model.

    Parameters:
    - text (str): The text to summarize.
    - model_name (str): The GPT-2 model variant to use. Default is 'gpt2'.
    - max_length (int): The maximum length of the summary.
    Returns:
    - summaries (list of str): The generated summaries.
    """
    #NOTE: Max chunk length causes gpt2 to be killed
    # Loaded model and tokenizer at the top of file, this model nll also used for classification
    model = GPT2_model #GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2_tokenizer # GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # Encode text input and add the EOS token
    max_summary_len = max((len(tokenizer.tokenize(input_text)) * summarization_ratio), max_chunk_len)
    summary = input_text
    sumlen=len(tokenizer.tokenize(summary))

    while max_summary_len < sumlen:
        chunks = chunk_sents_from_text(summary, max_chunk_len)
        chunksum = ""
        print(f"GPT2 has {len(chunks)} chunks")

        for chunk in chunks:
          prompt = f"{chunk} {command}:"
          if len(tokenizer.tokenize(chunk)) > 10:
            input_ids = tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=True, padding=True, max_length = 1000, truncation=True)
            prompt_length=len(tokenizer.tokenize(prompt))
            # Generate summary output
            input_ids = input_ids[:, :max_chunk_len] #1024 max token length includes completion
            summary_ids = model.generate(input_ids,
                                        max_length = 1020,
                                        max_new_tokens = int(min((max_chunk_len/2),(1016 - len(input_ids)))), min_length=25,
                                        do_sample=True,
                                        #temperature=.7,
                                        repetition_penalty=2.0,
                                        num_beams=4,
                                        length_penalty=18.0, 
                                        num_return_sequences=1, 
                                        early_stopping=True,
                                        no_repeat_ngram_size=4,
                                        top_p=0.95,
                                        top_k=50,
                                        eos_token_id=tokenizer.eos_token_id,
                                        pad_token_id=tokenizer.pad_token_id)
                                    #   attention_mask=input_ids['attention_mask'],
            output = tokenizer.decode(summary_ids[0][prompt_length:], skip_special_tokens=True) #[ prompt_length:] to separate prompt from generation
            #str([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])
            print(output)

            chunksum += output +". "
            gc.collect()
            torch.cuda.empty_cache()
        summary = chunksum.replace("..",".").replace(". .",". ")
        sumlen=len(tokenizer.tokenize(summary))
        # Decode and return the summaries
        print("GPT2 Summary: len", sumlen)

    return summary.replace("\n",". ").replace("..",". ").replace(". .",". ")
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch


def extractive_summary_nll(text, summarization_ratio=0.1):
    # Initialize tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base')

    # Split the text into sentences or coherent chunks
    chunks = chunk_sents_from_text(text, 200)  # This function needs to be defined or replaced with actual logic to split text
    total_length = len(tokenizer.tokenize(text))
    target_summary_length = int(total_length * summarization_ratio)  # Target length based on summarization ratio

    # Process each chunk and compute importance scores
    importance_scores = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        importance_score = probabilities[:, 1].item()  # Assuming index 1 corresponds to 'important'
        importance_scores.append((importance_score, chunk))

    # Sort chunks by importance score
    importance_scores.sort(reverse=True, key=lambda x: x[0])

    # Select chunks until the target summary length is reached
    summary_length = 0
    selected_chunks = []
    for score, chunk in importance_scores:
        chunk_length = len(tokenizer.tokenize(chunk))
        if summary_length + chunk_length <= target_summary_length or not selected_chunks:
            selected_chunks.append(chunk)
            summary_length += chunk_length
        else:
            break  # Stop adding chunks once we reach the target length

    # Compile the final summary
    summary = ' '.join(selected_chunks)
    print("Original Length:", total_length, "Target Summary Length:", target_summary_length, "Actual Summary Length:", len(tokenizer.tokenize(summary)))

    # Clean up resources
    gc.collect()
    torch.cuda.empty_cache()

    return summary.replace("\n",". ").replace("..",". ").replace(". .",". ")




def agglomerative_sampling(original_text, summarization_ratio=0.2 ):
        sentences = chunk_sents_from_text(original_text, 100)

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


        model = SentenceTransformer('all-MiniLM-L6-v2')  
        
        sentence_embeddings = [model.encode(sentence) for sentence in clean_sentences]

        if not sentence_embeddings:
            return
        #try:
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
        summary = '. '.join(chronological_summary).strip().replace("..",". ").replace(". .",". ")
        gc.collect()
        torch.cuda.empty_cache()

        return summary

# Normalize the abstractive summary to the original text quotes
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np

def kmeans_centroid_sampling(original_text, summarization_ratio=.10):
    # Step 1: Preprocess the text to split it into sentences
    sentences = chunk_sents_from_text(original_text, 100)

    # Step 2: Generate Sentence Embeddings
    #model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = get_embedding(sentences)#model.encode(sentences)

    # Step 3: Apply K-Means Clustering
    n_clusters = max(1, int(len(sentences) * summarization_ratio))

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)

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
    summary = ". ".join(sorted_sentences).strip().replace("..",".").replace(". .",". ")

    return summary

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
    chunks = chunk_sents_from_text(input_text, chunk_size)
    print(f"Claude has {len(chunks)} chunks")

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
    tokens_per_chunk =  32000  # Initial optimal chunk size

    current_text = input_text  # Initialize current_text to be the input_text
    max_tokens = tokens_per_chunk -  1 if tokens_per_chunk > max_tokens else max_tokens  # Make the loop condition smaller than attention window

    while num_output_summary_tokens > max_tokens:
        # We chunk the current text (initially the input text, later could be the summary of the input text)
        chunks = chunk_sents_from_text(current_text, tokens_per_chunk)
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
        current_text = '. '.join(chunk_summaries).strip()

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


    if original_text_sentence_embeddings:
        original_embeddings=original_text_sentence_embeddings
    else:
        original_embeddings = [get_embedding(sent) for sent in original]
    comparable_embeddings = [ get_embedding(sent) for sent in comparison]
# Initialize NearestNeighbors with n_neighbors set to the length of the original list.
    # This ensures we have enough neighbors to find unique matches.
    num_neighbors = min(len(comparable_embeddings), len(original_embeddings) - 1)
    nbrs = NearestNeighbors(n_neighbors=num_neighbors, metric='cosine').fit(original_embeddings)
    # Prepare the list to store the pairs and a set to track used original sentences.
    grounded_sentences = []
    available_indices = set(range(len(original)))  # All original indices are initially available.
    # Find nearest neighbors for each sentence in the comparison list.
    distances, indices = nbrs.kneighbors(comparable_embeddings)

    abstract_sentences=[]
    # Iterate through each set of neighbors found for each sentence in comparison.
    for comp_idx, neighbor_indices in enumerate(indices):
        # Go through the list of neighbors for the current comparable sentence.
        for neighbor_index in neighbor_indices:
            if neighbor_index in available_indices:
                # If this neighbor hasn't been used yet, create a pair. (original[neighbor_index], comparison[comp_idx])
                grounded_sentences.append(original[neighbor_index])
                abstract_sentences.append(comparison[comp_idx])
                # Remove the index from available indices and break out of the loop.
                available_indices.remove(neighbor_index)
                break  # Move on to the next comparable sentence once a pair is made.
    # Return the list of unique nearest neighbor pairs.
    return grounded_sentences, abstract_sentences


## VISUALIZE THE DATA

import matplotlib
def generate_heatmap(scoreboards, name=""):
    """
    Generate a heatmap showing the theme scores for each model.
    
    Args:
    scoreboards (dict of dict): Dictionary of dictionaries, each representing theme scores for a specific model.
    name (str): Name for the plot, typically the text or dataset being analyzed.
    
    Returns:
    Tuple of file names for the heatmap image and the CSV file.
    """
    # Extract all models and themes from the scoreboards
    all_models = list(scoreboards.keys())
    # Assume all models have the same set of themes for simplicity
    all_themes = list(next(iter(scoreboards.values())).keys())

    # Initialize a matrix to hold the score data for the heatmap
    scores_matrix = np.zeros((len(all_models), len(all_themes)))

    # Fill in the scores matrix with values from the scoreboards
    for j, model in enumerate(all_models):
        for i, theme in enumerate(all_themes):
            model_score = scoreboards[model].get(theme, 0)  # Default to 0 if theme not present
            scores_matrix[j, i] = model_score
    theme_titles = [theme.split(":")[0] for theme in all_themes]
    # Creating the heatmap
    sns.set(style="whitegrid")
    plt.figure(figsize=(18, 10))  # Adjust size as needed
    sns.heatmap(scores_matrix, annot=True, fmt=".2f", cmap='coolwarm',
                yticklabels=all_models, xticklabels=theme_titles)
    plt.title(f"Theme Differences Across Models' Retellings of:\n{name}")
    plt.ylabel('Model')
    plt.xlabel('Theme')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    file_name = re.sub(r'[^\w\s]', '', name).replace(" ", "")
    # Save the heatmap
    heatmap_file_name = f'{file_name}_theme_scores_heatmap.png'
    plt.savefig(heatmap_file_name, dpi=300)
    plt.close()  # Close the plot to prevent it from displaying inline if undesired
    # Convert the scores matrix to a DataFrame for saving as CSV
    scores_df = pd.DataFrame(scores_matrix, index=all_models, columns=theme_titles)
    csv_file_name = f'{file_name}_theme_scores.csv'
    scores_df.to_csv(csv_file_name)
    
    return heatmap_file_name, csv_file_name

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
        return pos - orig_pos, neg - orig_neg

    # Setup figure and axes
    fig, ax = plt.subplots(figsize=(18, 12))

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
        ax.bar(indices + i * bar_width, heights, bar_width, bottom=bottoms, label=category)#.split()[0])
    title = f"Amplitude of Net Moral Value Bias in LLM Distillation of {title} \n"
    # Customize the plot
    ax.set_xlabel('Model')
    ax.set_ylabel('∆(pos - neg) Story Moral Representation  \n  Change in modeled occurrences of positive theme minus negative theme')
    ax.set_title(title)
    ax.set_xticks(indices + bar_width * (n_categories / 2))
    ax.set_xticklabels(model_names)
   # ax.legend(title="Category", loc='upper left', bbox_to_anchor=(1, 1))
    ax.legend(title=f"", loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=1)
    ax.grid(True, which='major', axis='y', linestyle='--')
    plt.xticks(rotation=45, ha='right')  # Rotate text type names for better visibility

    plt.tight_layout()
#    plt.savefig(file_name)
    plt.show()
    # Save and show the graph
    
    file_name = re.sub(r'[^\w\s]', '', file_name).replace(" ", "")
    file_name = file_name if file_name.endswith(".png") else file_name + ".png"
    plt.savefig(file_name, format='png', dpi=300, bbox_inches='tight')

    return file_name


def theme_span_linegraph(full_codebook_scores, scoreboards, file_name="line_graph.png", title=""):
    """
    Generate a scalable line graph showing the percentage difference for each category across various text types and save it as an image file. 
    Display the average absolute value percentage change for each summary type compared to the full_codebook_scores.

    Args:
    full_codebook_scores (dict): Category percentages for the original text.
    scoreboards (dict of dict): Dictionary of dictionaries, each representing category percentages for a specific summary type.
    file_name (str): Name of the file to save the graph.
    title (str): The title of the content used for moral summary evaluation
    """
    # Initialize a dictionary to store percentage differences for each category across text types
    categories = list(set(full_codebook_scores.keys()))

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
        plt.plot([category.split(":")[0] for category in categories], text_type_data, label=text_type, marker='o')
    
    # Customize the graph
    plt.title(f'Comparison of Models\' Summary of: \n{title}')
    plt.xlabel('Theme')
    plt.ylabel('Change in Percent of Sentences Classified by Theme')
    plt.xticks(rotation=45, ha='right')  # Rotate category names for better visibility
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add grid for better readability
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place the legend outside the plot
    # Save and show the graph
    
    file_name = re.sub(r'[^\w\s]', '', file_name).replace(" ", "")
    file_name = file_name if file_name.endswith(".png") else file_name + ".png"
    plt.savefig(file_name, format='png', dpi=300, bbox_inches='tight')
    # plt.show()
    return file_name

def model_dif_scatterplot(full_codebook_scores, scoreboards, file_name="line_graph.png",title=""):
    """ 
    Generate a scalable line graph showing the percentage difference for each category across various text types and save it as an image file. 
    Display the average absolute value percentage change for each summary type compared to the full_codebook_scores.

    Args:
    full_codebook_scores (dict): Category percentages for the original text.
    scoreboards (dict of dict): Dictionary of dictionaries, each representing category percentages for a specific summary type.
    file_name (str): Name of the file to save the graph.
    title (str): The title of the content used for moral summary evaluation
    """
    categories = list(set(full_codebook_scores.keys()))
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
    text_types.extend((scoreboards.keys()))

    # No difference compared to baseline split
    avg_percentage_diffs["Original"] = 0
    percentage_diffs["Original"] = {category:0 for category in categories}
    # Create the line graph
    plt.figure(figsize=(10, 6))
        # Plot lines for each category, following the sorted order of text_types
    for category in categories:
        category_data = [percentage_diffs[text_type][category] for text_type in text_types]
        plt.scatter(text_types, category_data, label=category.split(":")[0], marker='o')
    # Plot the average points for each summary type, following the sorted order
    for i, text_type in enumerate(text_types):
    #    plt.scatter(text_type, avg_percentage_diffs[text_type], color='black', zorder=5)
        plt.text(text_type, avg_percentage_diffs[text_type], f'  {avg_percentage_diffs[text_type]:.2f}%' if avg_percentage_diffs[text_type]>0 else '', rotation=45,  color='black')#,verticalalignment='bottom',)
    avg_data = [avg_percentage_diffs[text_type] for text_type in text_types]
    plt.plot(text_types, avg_data, label='Average Change', color='black', marker='D', linestyle='-', linewidth=2)

    # Customize the graph
    plt.title(f'Change in Thematic Distribution of Story Morals across Models\' Summary of: \n {title}')
    plt.xlabel('Language Model')
    plt.ylabel('Average Difference Percentage')
    plt.xticks(rotation=45, ha='right')  # Rotate text type names for better visibility
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add grid for better readability
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place the legend outside the plot
    file_name = re.sub(r'[^\w\s]', '', file_name).replace(" ", "")
    file_name = file_name + ".png" if ".png" not in file_name else file_name
    plt.savefig(file_name, format='png', dpi=300, bbox_inches='tight')
    # plt.show()
    return file_name

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency
def compare_distributions(orig_dist, summ_dist, name,summary_type): 
        category_diffs=""
        for category in orig_dist.keys():
            orig_pct = orig_dist.get(category, 0)
            summ_pct = summ_dist.get(category, 0)
            category_diffs+= str(f"Category '{category.split()[0]}': Original = {orig_pct:.2f}%, {summary_type} {name} = {summ_pct:.2f}%")
            category_diffs+=f"\n<br>Percentage difference {(summ_pct - orig_pct):.2f}%\n<br>"
        results =   str(category_diffs)        
        minimum = .1+abs(min(min(orig_dist.values()), min(summ_dist.values()), 0) ) # normalize min negative value to 0
        # Chi square does not allow negative values; : If a category's count is zero, you can add a small constant (like 0.5 or 1) to all counts. This technique, known as the Yates Continuity Correction or Laplace smoothing, can help avoid zero expected frequencies without significantly distorting the data.
        orig_dist = {key:num + minimum for key, num in orig_dist.items()}
        summ_dist =  {key:num + minimum for key, num in summ_dist.items()}
        results += "\n" 
        results += perform_chi_square_test(orig_dist, summ_dist, name, summary_type)

        return str(results)

def perform_chi_square_test(original_counts, summarized_counts, name="", summary_type=""):
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
    results += f"<br>Chi-square Statistic: {chi2}\n<br>"
    results += f"Degrees of Freedom: {dof}\n<br>"
    results += f"P-value: {p}\n<br>"
    results += f"Expected Frequencies:\n<br>"
    results += str(expected) +"\n<br>"

    # Interpret p-value
    alpha = 0.05  # significance level
    if p < alpha:

        results += str("Significant differences exist between the distributions (reject H0)\n<br>")
        results += f"\n\n<br> Generated a heatmap of Statistical Significance <br>\n ![Chi-Square Test Results for {summary_type} in {name}](" 
        results += generate_chi_square_heatmap(original_counts, summarized_counts, name=name,summary_type = summary_type) + ")\n\n<br><br>"
        return results
    else:
        results += str("<br>No significant differences exist between the distributions (H0) null hypothesis\n<br><br>")
        return results


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
    plt.figure(figsize=(16, 6))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap='YlGnBu', xticklabels=list(all_categories), yticklabels=["Original", "Summarized", "Expected Original", "Expected Summarized", "Chi-Square", "P-Value"])
    plt.title(f'Thematic Preferences in {summary_type} model retellings of {name}\n Chi-Square Heatmap')
    plt.yticks(rotation=0)  # Rotate labels
    plt.xticks(rotation=45, ha='right')  # Rotate labels
    plt.tight_layout()
    # Save or display the heatmap
    file_name = re.sub(r'[^\w\s]', '', name+summary_type).replace(" ", "").replace("'","").replace(",","").replace("&","")
    file_name += '_chi_square_heatmap.png'
    plt.savefig(file_name, dpi=300)
    # plt.show()
    return file_name

# Helper Functions

def chunk_sents_from_text(text: str, max_chunk_len: int) -> list[str]:
    all_tokens = BERT_tokenizer.tokenize(text)
    # Check if the text length is less than or equal to the chunk size

    # Initialize variables
    if len(all_tokens) <= max_chunk_len:
        print("Skip chunking")
        return [text]

    sents =  [sent.text for sent in nlp(text).sents] if isinstance(text,str) else text

    # Calculate target chunk length 
    target_num_chunks = math.ceil(len(all_tokens) / max_chunk_len)
    target_chunk_len = math.ceil(len(all_tokens) / max(1,target_num_chunks))
    curr_chunk_tokens = 0

    torch.cuda.empty_cache()
    gc.collect()
    curr_chunk_sents = []
    chunks = []
    for i, sent in enumerate(sents):
        sent_tokens = sent.split(" ")#BERT_tokenizer.tokenize(sent)
        # Check if adding current sentence would exceed target length


        if curr_chunk_tokens + len(sent_tokens) > max_chunk_len:
                # Current sentence cannot be added without exceeding max_chunk_len
                # Finalize current chunk and start a new one with the current sentence
                chunks.append(". ".join(curr_chunk_sents))
                curr_chunk_sents = [sent]
                curr_chunk_tokens = len(sent_tokens)

        else:
                # Add sentence to current chunk
                curr_chunk_sents.append(sent)
                curr_chunk_tokens += len(sent_tokens)
                # Check if the current chunk has reached or exceeded the target length
                if curr_chunk_tokens >= target_chunk_len:
                    # Finalize current chunk
                    chunks.append(". ".join(curr_chunk_sents).replace(". .",". ").replace("..","."))
                    curr_chunk_sents = []
                    curr_chunk_tokens = 0

    # Add any remaining sentences in final chunk
    if curr_chunk_sents:
        chunks.append(". ".join(curr_chunk_sents).replace(". .",". ").replace("..","."))
    return chunks


def get_embedding(text):
    # Ensure model is in evaluation mode and to the right device
    BERT_embedding_model.eval()
    device='cuda'
    BERT_embedding_model.to(device)  # Or 'cuda' if GPU is available and preferable
#    BERT_tokenizer.to(device)
    with torch.no_grad():  # Ensures no gradients are calculated to save memory
        inputs = BERT_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=510)
        inputs = {key: val.to(device) for key, val in inputs.items()}  # Or 'cuda'
        outputs = BERT_embedding_model(**inputs)

    # Detach and move to CPU if necessary, then convert to numpy
    embedding = outputs.pooler_output.squeeze().detach().cpu().numpy()

    return embedding

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

# pip install youtube_transcript_api
from youtube_transcript_api import YouTubeTranscriptApi
def load_and_preprocess_srt(file_path):
        subs = pysrt.open(file_path, encoding='iso-8859-1')
        texts = [' '.join(sub.text_without_tags.split()) for sub in subs]
        preprocessed_text = ' '.join(texts)
        return preprocessed_text

if __name__ == '__main__':

    research_question = "The Moral of the Story is "
    if False: # These are the tests we've passed:
        codebook = [    
            ("Peace", "Conflict"), 
            ("Altruism ", "Individualism"),     
            ("Spiritualism", "Materialism") 
        ]  
        video_id="F3-2kkD1LNU" # Moral Childrens Stories https://www.youtube.com/watch?v=F3-2kkD1LNU
        name="ChildrensStories"
        out = analyze_summarization_effects(name=name, content_id=video_id, research_question=research_question,codebook=codebook, summarization_ratio=.1)
        codebook = [ #zen koans codebook
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

        name = "101 Zen Stories and Koans"
        video_id= "Y0p663Ot8mo" # zen koans 1hr  "https://youtu.be/Y0p663Ot8mo \n"
        out = analyze_summarization_effects(name=name, content_id=video_id, research_question=research_question,codebook=codebook, summarization_ratio=.1)
        print(out)

        gc.collect()

        codebook = [ #gospels_codebook
            # Theme: Compassion vs. Apathy
            ("Compassion: Acts of kindness, teachings on loving one’s neighbor", "Apathy: Indifference or lack of concern towards others' suffering"),
            # Theme: Forgiveness vs. Resentment
            ("Forgiveness: Narratives or teachings specifically mentioning forgiveness, stories of reconciliation", "Resentment: Holding grudges or refusing to forgive"),
            # Theme: Justice vs. Injustice
            ("Justice: Teachings on fairness, parables involving justice, critiques of hypocrisy", "Injustice: Actions or policies that perpetuate inequality or unfair treatment"),
            # Theme: Humility vs. Pride
            ("Humility: Teachings that promote modesty, narratives showcasing lowering of oneself", "Pride: Arrogant behavior, refusal to acknowledge one’s flaws"),
            # Theme: Faith vs. Doubt
            ("Faith: Passages encouraging trust in God, examples of individuals displaying strong faith", "Doubt: Lack of faith or trust, skepticism without seeking truth"),
            # Theme: Repentance vs. Stubbornness
            ("Repentance: Stories or teachings highlighting change of heart or turning away from sin", "Stubbornness: Refusal to change one’s ways or acknowledge wrongdoing"),
            # Theme: Sacrifice vs. Selfishness
            ("Sacrifice: Examples of self-giving love, teachings on serving others, selflessness", "Selfishness: Focusing solely on one’s own interests without regard for others"),
            # Theme: Truth vs. Deception
            ("Truth: Teachings against lying or deceit, parables valuing truthfulness", "Deception: Engaging in lies, misleading others for personal gain"),
            # Theme: Peace vs. Conflict
            ("Peace: Teachings promoting peace, narratives resolving conflicts without violence", "Conflict: Inciting or engaging in violence, exacerbating disagreements"),
            # Theme: Stewardship vs. Exploitation
            ("Stewardship: Passages discussing care for the Earth, responsible management of resources", "Exploitation: Misusing resources or harming the environment for selfish ends")
        ]


        name = "The Parables of Jesus Christ"
        video_id = "Ed41paFWSKM" # 60min Parables "https://www.youtube.com/watch?v=Ed41paFWSKM \n"
        out = analyze_summarization_effects(name=name, content_id=video_id, research_question=research_question, codebook=codebook)
        print(out)

        gc.collect()
        torch.cuda.empty_cache()

    codebook = [ #aesops_fables_codebook
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
    name = "Aesop's Fables"
    video_id =  "https://ia801403.us.archive.org/26/items/aesopsfables00028gut/aesopa10.txt" #"aaMLVsH6ikE" #aesops fables 3hr   "https://youtu.be/aaMLVsH6ikE \n"
    out = analyze_summarization_effects(name=name, content_id=video_id, research_question=research_question,codebook=codebook, summarization_ratio=.05)
    print(out)
    input("Models analyzed against Aesop's Fables. Continue to Plato's Republic?")
    codebook = [ #platos_republic_codebook
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

    name = "Plato's Republic \n"
    video_id = "CqGsg01ycpk" #5hr audiobook of the Republic #"https://youtu.be/CqGsg01ycpk \n"
    out = analyze_summarization_effects(name=name, content_id=video_id, research_question=research_question,codebook=codebook, summarization_ratio=.05)

    print(out)

    gc.collect()

    

    gc.collect()
    codebook = [ #panchatantra_codebook
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
    name = "The Fables of the Panchatantra"
    video_id = "https://archive.org/stream/ryder-1925-panchatantra-english/Ryder_1925_Panchatantra_English_djvu.txt" #"lWnJem4hKxY" # 8 hours of the Panchatantra  "https://youtu.be/lWnJem4hKxY \n"
    out = analyze_summarization_effects(name=name, content_id=video_id, research_question=research_question,codebook=codebook, summarization_ratio=.05)
    print(out)
    print("SUCCESS! ALL CONTENT HAS BEEN ANALYZED")
    
    if False: #separate gospels? To test
        name = "The Gospel of Matthew"
        out = name + "\n"
        video_id = "matthew.txt"#"https://openbible.com/textfiles/akjv.txt" 
        out += analyze_summarization_effects(name=name, content_id=video_id, research_question=research_question,codebook=gospels_codebook, summarization_ratio=1/20) 

        name=name.replace(" ", "").replace(",","").replace("&","")+ "_model_moralsum_results.txt"
        with open(name , 'w') as f:
            f.write(out)
        print(name)
        input("Continue to the Gospel of Mark?")
        name = "The Gospel of Mark"
        out = name + "\n"
        video_id = "mark.txt"#"https://openbible.com/textfiles/akjv.txt" 
        out += analyze_summarization_effects(name=name, content_id=video_id, research_question=research_question,codebook=gospels_codebook, summarization_ratio=1/20) 

        name=name.replace(" ", "").replace(",","").replace("&","")+ "_model_moralsum_results.txt"
        with open(name , 'w') as f:
            f.write(out)
        print(name)
        input("Continue to the Gospel of Luke?")
        name = "The Gospel of Luke"
        out = name + "\n"
        video_id = "luke.txt"#"https://openbible.com/textfiles/akjv.txt"
        out += analyze_summarization_effects(name=name, content_id=video_id, research_question=research_question,codebook=gospels_codebook, summarization_ratio=1/20) 

        name=name.replace(" ", "").replace(",","").replace("&","")+ "_model_moralsum_results.txt"
        with open(name , 'w') as f:
            f.write(out)

        print(name)
        input("Continue to the Gospel of John?")
        name = "The Gospel of John"
        out = name + "\n"
        video_id = "john.txt"#"https://openbible.com/textfiles/akjv.txt"
        out += analyze_summarization_effects(name=name, content_id=video_id, research_question=research_question,codebook=gospels_codebook, summarization_ratio=1/20) 

        name=name.replace(" ", "").replace(",","").replace("&","")+ "_model_moralsum_results.txt"
        with open(name , 'w') as f:
            f.write(out)

        print(name)
