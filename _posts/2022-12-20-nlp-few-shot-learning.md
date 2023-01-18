---
layout: post
usemathjax: true
title: "Few Shot Learning"
subtitle: "Exploring the Power of Few Shot Learning in the Natural Language Processing World"
date: 2022-10-08 23:45:13 -0400
background: '/img/posts/model-calibration.jpeg'
---

<h1 style="text-align: center;">Intro</h1>

Since the arrival of gpt-3 the areas of few shot learning has been more popular.
Providing a few examples of a specific task the model can auto complete the prediction from the prompt.

 1. <b>Task description</b>
	 - *Translate English to French*

 2. <b>Examples</b>
	 - *sky -> ciel*
	 - *moon -> lune*
	 - *cloud -> nuage*

 3. <b>Prompt</b>
	 - *wind ->*
	 
Giving few examples are the reason for the name few-shot learning.
Actually there is no learning in the inference time, in the case of GPT-3 there are no gradient updates performed on the model.
GPT-3 is a pre-trained model, and during inference, it utilizes the knowledge that it learned during training to generate outputs based on input data, but it does not update it's weight as it's not being trained anymore.

<h2 style="text-align: center;">Limitations</h2>
<b></b>
 - The performance of the model rely on good prompts. For example adding
   the prompt “lets take it step by step” to make high order reasoning
   gave much better results.
 - Deployment of huge models isn’t an easy task on edge devices.
 - Relies only on the pre-trained knowledge.
 - Limitation of input length and quadratically scaling attention.

![add image](/img/posts/few-shot-learning/Aggregate Performance Across Benchmarks.png)

<b>How to make it more efficient?</b>

<h2 style="text-align: center;">PET (“pattern exploiting training”)</h2>
[Exploiting Cloze Questions for Few Shot Text Classification and Natural Language Inference](https://arxiv.org/abs/2001.07676)

*Some NLP tasks can be solved in a fully unsupervised fashion by providing a pretrained language model with “task descriptions” in natural language (e.g., Radford et al., 2019). While this approach underperforms its supervised counterpart, we show in this work that the two ideas can be combined: We introduce Pattern-Exploiting Training (PET), a semi-supervised training procedure that reformulates input examples as cloze-style phrases to help language models understand a given task. These phrases are then used to assign soft labels to a large set of unlabeled examples. Finally, standard supervised training is performed on the resulting training set. For several tasks and languages, PET outperforms supervised training and strong semi-supervised approaches in low-resource settings by a large margin.*

Shows that any LM, not only huge ones, is able to learn new tasks from few learning examples.
Instead of doing “in-context prompt” like in GPT-3, convert our problem into task specific prompt learning.
Task specific prompt learning converts the downstream task into a masked language model problem.

Input: “Best pizza ever!”
PLM task: “It was ____.”
Output:

 - Great -> 1: 0.8
 - Bad -> 0: 0.2

The two ingredients, beside some labeled data, that we need are:
 - Pattern - predefined template
 - Verbalizer - translate the answer to a vocabulary token

Now we can train for each template a model (or a few) and ensemble them as teachers at the end to label unseen data as silver labels. We are doing ensemble because each model alone is unstable for be trained on few little samples. These labels are fed to another model as a distillation process. Doing this iteratively we are having the I-PET.

Cons:
 - Needs lots of prompts templates
 - Requires multi-step training including adapting and an ensemble of several PLMs
 - Will update gradient only for the label tokens
	 - Question: **Are workouts healthy?** Answer: **Yes**
		 - <b style="color:black;">Workout</b>
		 - <b style="color:black;">Health</b>
		 - <b style="color:green;">Yes</b>
		 - <b style="color:red;">No</b>

![add image](/img/posts/few-shot-learning/Amount if task-specific data used.png)

<h2 style="text-align: center;">ADAPET</h2>
[Improving and Simplifying Pattern Exploiting Training](https://arxiv.org/abs/2103.11955)

*Recently, pre-trained language models (LMs) have achieved strong performance when fine-tuned on difficult benchmarks like SuperGLUE. However, performance can suffer when there are very few labeled examples available for fine-tuning. Pattern Exploiting Training (PET) is a recent approach that leverages patterns for few-shot learning. However, PET uses task-specific unlabeled data. In this paper, we focus on few-shot learning without any unlabeled data and introduce ADAPET, which modifies PET’s objective to provide denser supervision during fine-tuning. As a result, ADAPET outperforms PET on SuperGLUE without any task-specific unlabeled data.*

 - The model consider every word of the vocabulary of the blank:
    - Question: **Are workouts healthy?** Answer: **Yes**
   	 - <b style="color:red;">Workout</b>
   	 - <b style="color:red;">Health</b>
   	 - <b style="color:green;">Yes</b>
   	 - <b style="color:red;">No</b>
 - The model mask some parts of the question:
	 - Question: **Are [MASK] healthy?** Answer: **Yes**

<h4 style="text-align: center;">Summary</h4>
PET enables few-shot learning by distillation from an ensemble of models trained on patterns capturing tasks as a closed tasks (filling in the gaps).
iPET enables the patterns to capture more tokens than just only one and then ADAPET simplified PET  by still using patterns, but replacing distillation with better losses.

<h2 style="text-align: center;">PEFT (“parameter efficient fine-tuning”)</h2>
[Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning](https://arxiv.org/abs/2205.05638)

*Few-shot in-context learning (ICL) enables pre-trained language models to perform a previously-unseen task without any gradient-based training by feeding a small number of training examples as part of the input. ICL incurs substantial computational, memory, and storage costs because it involves processing all of the training examples every time a prediction is made. Parameter-efficient fine-tuning (PEFT) (e.g. adapter modules, prompt tuning, sparse update methods, etc.) offers an alternative paradigm where a small set of parameters are trained to enable a model to perform the new task. In this paper, we rigorously compare few-shot ICL and PEFT and demonstrate that the latter offers better accuracy as well as dramatically lower computational costs. Along the way, we introduce a new PEFT method called (IA)3 that scales activations by learned vectors, attaining stronger performance while only introducing a relatively tiny amount of new parameters. We also propose a simple recipe based on the T0 model called T-Few that can be applied to new tasks without task-specific tuning or modifications. We validate the effectiveness of T-Few on completely unseen tasks by applying it to the RAFT benchmark, attaining super-human performance for the first time and outperforming the state-of-the-art by 6% absolute. All of the code used in our experiments is publicly available.*

T-Few combines adapter layers with a multi task model (T0). T0 trained on many different tasks in parallel. If we can find ways to update the adapters layers inside T0 we can do a few-shot learning.
Image [T-Few paper (flops per example)]

Cons:

 - High sensitive to prompt engineering
 - Large backbone model on edge devices

<h2 style="text-align: center;">SetFit</h2>
[Efficient Few-Shot Learning Without Prompts](https://arxiv.org/pdf/2209.11055.pdf)

*Recent few-shot methods, such as parameterefficient fine-tuning (PEFT) and pattern exploiting training (PET), have achieved impressive results in label-scarce settings. However, they are difficult to employ since they are subject to high variability from manually crafted prompts, and typically require billionparameter language models to achieve high accuracy. To address these shortcomings, we propose SETFIT (Sentence Transformer Finetuning), an efficient and prompt-free framework for few-shot fine-tuning of Sentence Transformers (ST). SETFIT works by first finetuning a pretrained ST on a small number of text pairs, in a contrastive Siamese manner. The resulting model is then used to generate rich text embeddings, which are used to train a classification head. This simple framework requires no prompts or verbalizers, and achieves high accuracy with orders of magnitude less parameters than existing techniques. Our experiments show that SETFIT obtains comparable results with PEFT and PET techniques, while being an order of magnitude faster to train. We also show that SETFIT can be applied in multilingual settings by simply switching the ST body.*

Some benchmarks dataset on few-shot learning:

 - RAFT (Alex et al. 2021)
 - FLEX (Bragg et al. 2021)
 - CLUES (Mukherjee et al. 2021)
[raft-leaderboard](https://huggingface.co/spaces/ought/raft-leaderboard)

<p style="text-align: center;">F1 comparison: SetFit vs. GPT-3 on RAFT benchmark</p>
![add image](/img/posts/few-shot-learning/F1 comparison- SetFit vs. GPT-3 on RAFT benchmark.png)

<h2 style="text-align: center;">Resources</h2>
+ [Huggingface Set-Fit blogpost](https://towardsdatascience.com/sentence-transformer-fine-tuning-setfit-outperforms-gpt-3-on-few-shot-text-classification-while-d9a3788f0b4e)
