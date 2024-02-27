# Summarization of Symbolic Stories for Evaluating Moral Comprehension of Language Models
A Python script for benchmarking Language Models on their narrative comprehension of the moral of the story in culturally significant texts

## Overview

The Moral Summarization Analysis Tool is an advanced Python-based framework designed to analyze, evaluate, and compare the thematic fidelity of various state-of-the-art AI summarization models against original texts. This tool is pivotal for understanding how different AI models like LLAMA, Mistral, BART, T5, and RoBERTa process and convey the themes and morals embedded in extensive texts, such as stories, documentaries, and lectures. It leverages advanced NLP techniques to generate, analyze, and visualize the thematic distributions, offering deep insights into the interpretive biases and abstract comprehension capabilities of each summarization approach.

### Features

- **Summarization**: Utilizes leading AI models for comprehensive text analysis.
- **Theme Analysis**: Detailed examination of theme distributions compared to original texts. 
- **Interactive Visualization**: Provides fidelity scores, chi-square tests, and interactive graphs for in-depth comparative analysis.
- **Media Format Support**: Compatible with plain texts, subtitles (.srt), documents (.txt), YouTube video IDs, and direct web content.

### Analytical Dimensions

- **Explicit Interpretation**: Measures explicit representation of specified themes.
- **Overall Fidelity**: Assesses accuracy against original theme distributions. 
- **Implicit Perspectives**: Detects nuanced interpretations and potential biases using statistical methods.

### Summarization

The script invokes several state-of-the-art transformer models from BERT to LLAMA to analyze texts:

- LLAMA-2-7b: Large language model featuring chain-of-thought aptitude; Compresses information into high-level insights with abstractions and deductions  
- Mistral-7b: An instructable 7B parameter autoregressive language model by Anthropic designed for controllable generations 
- Phi2: Abstractive summarizer finetuned on self-supervised data designed by Microsoft  
- Gemma-2b: Abstractive summarizer finetuned on self-supervised data designed by Microsoft  
- BART: Massive denoising bidirectional transformer, strong on summarization tasks
- T5: Large pretrained encoder-decoder model   
- GPT-2: Seminal transformer language model from OpenAI (non-encoder)  
- RoBERTa: Robustly optimized BERT architecture with improved training methodology   
- Kmeans-BERT: Uses BERT sentence embeddings with k-means clustering to summarize  
- MiniLM Agglomerative: Clustering approach with MiniLM sentence vectors
- Option for Claude or GPT4 with API key (not tested in example)

These models cover a range of approaches, architectures, and modalities – from clustering encoded representations to end-to-end supervised sequence generation architectures. Two ensemble methods combining sentences from other summaries are also provided.

## Analysis  

The comparative analytic capabilities to derive insights across models include:   

**Sentence Classification**   

- NSP (BERT)   
- NLL (GPT-2) Classify sentences from summaries and original texts with a user-specified “codebook” of textual categories and themes. By default provides positive and negative variations of each theme. A research question template guides classification decisions.   

**Model Calibration Analysis**   

- Sentence-level precision: Word Mover’s Distance between matched codebook/summary pairs  
- Corpus-level distributions: Chi-squared statistical tests measure significance of differences in category distributions for each model  
- Mean difference scores convey model uncertainty in making classification decisions, codifying only above-average theme categorizations

**Quantification of Thematic Fidelity in Summarization**   

- Percentage theme distribution similarity to original scope
- Distance between scope and summary
- Generated difference between codebook and summary

### Visual Comparisons   

![Interactive line graphs](/StoryAlignmentEval_TheParablesofJesusChrist/TheParablesofJesusChrist_net_theme_polarity_lines.png)   

![Scatterplots](/StoryAlignmentEval_101ZenStories/101ZenStoriesandKoans_model_meandiff_scatterplot.png)   

![Heatmaps](/StoryAlignmentEval_TheAllergoryOfTheCave/TheAllegoryoftheCave_theme_scores_heatmap.png)   

## Usage   

The primary function for analysis is analyze_summarization_effects. It accepts these parameters:   

- name: Name of content   
- content_id: Text, subtitles, video ID, etc.  
- research_question: Template text for guiding classification decisions when comparing each sentence to moral themes  
- codebook: List of textual themes/categories to track, ideally as positive and negative pairs covering multiple axes of potential moral perspectives. For example, peace vs. conflict.  
- summarization_ratio: Decimal ratio of summary length to original length, typically 0.1 – 0.3.  

**Example Call**  

```python
codebook = [    
    ("Peace", "Conflict"), 
    ("Altruism ", "Individualism"),     
    ("Spiritualism", "Materialism") 
]  

video_id="F3-2kkD1LNU" # Moral Childrens Stories https://www.youtube.com/watch?v=F3-2kkD1LNU  
name="ChildrensStories"  

research_question = "The moral of the story is "  

out = analyze_summarization_effects(name=name, content_id=video_id, research_question=research_question,codebook=codebook, summarization_ratio=.1)  
```

This generates a Name_ModelMoralSum_results.md markdown file analyzing the summaries, along with heatmap and graph visualizations of the distributions.  

## Output Files   

- Analysis report (markdown)
- Category distribution heatmap
- Theme preference line graph  
- Model accuracy scatterplot  
   

## Interpreting Outputs  

The core output is the markdown report that overviews the classified theme distributions in both the original text and across all model summaries evaluated. It includes:  

- The average uncertainty in sentence classifications as mean difference score
- A matrix of theme distribution percentages  
- Statistical significance testing between original and summarized scopes  
- Fidelity scores conveying how precisely each summary maintained scope distributions  
- Display of the most representative quotes captured for each theme by each model    
- Full generated summaries for qualitative review   

The interactive graph visualizations reveal how each summarization technique differently filters, focuses, or distorts the scope distribution when condensing the original material.   

Heatmaps convey where specific categories may be over or under represented in aggregate across all models, indicating potential alignment or gaps in the themes resonating across AI summarizers.  

Line graphs demonstrate the shifts in distribution from original to summarized scopes, highlighting outliers where models strongly gravitated to or avoided themes dominant in the original context.   

Scatterplots identify summarizers that introduced the greatest distribution skews on average across all categories.   

This multilayer quantitative view, combined with sampling representative sentences, supports an understanding of the ways AI models reliably succeed or fail in distilling themes proportionally and accurately when reinterpreting a complex context.

## Requirements   

**Python Version**: The script requires Python 3.7 or later to leverage transformer libraries.  

**Main Libraries**:   

- Transformers
- Torch  
- SciPy  
- NLTK  
- Matplotlib  
- Seaborn   
- SpaCy   
- PsychoPy  

**GPU Requirements**: Testing utilized consumer GPUs including Nvidia RTX 3060. Performance is feasible on most mid-range modern discrete GPUs. Summarization of long transcripts may require proportional VRAM capacity.  

To manage memory, batch chunking is dynamically calibrated within the maximum attention window of each transformer model. This prevents truncation but does require temporarily higher utilization during peak Steps.

## Applications   

This framework provides both qualitative and quantitative insights into how well AI summarization models can capture, convey, and contextualize the essence of stories in ways aligned with human values.  

As neural language systems progress in reasoning capacity, it is crucial that their knowledge representations align with moral perspectives that maintain fidelity to source contexts. Measuring current systemic gaps is essential for further progress in human-centered design.  

### Symbolic Stories as Language Model Evaluations:  

- Auditing summarization systems for potential harms or biases resulting from distortion or imbalance of themes derived from source materials that may shape perspectives. 
- Evaluating how well AI narrative comprehension captures student developmental learning themes to potentially guide interactive dialogues or assessments.   
- Studying bidirectional impacts between literary exposure and moral reasoning by assessing theme resonances and qualitative analysis of moral comprehension in traditional literature by adapting codebooks as exploratory tools for analyzing model attitudes, tones, and values.
