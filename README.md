# Teeny-Tiny Castle 🏰

[![DOI](https://zenodo.org/badge/524324724.svg)](https://zenodo.org/badge/latestdoi/524324724)[
![made-with-python](https://camo.githubusercontent.com/f9010d0d18143896d2e496fe0e0c89056acab8229dbdf169f1d3a4759567fe63/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f4d616465253230776974682d507974686f6e2d3166343235662e737667)](https://www.python.org/)

![python](https://i.gifer.com/origin/25/25dcee6c68fdb7ac42019def1083b2ef_w200.gif)

AI Ethics and Safety are (_relatively_) new fields, and their tools (and how to handle them) are still **not known to most of the development community**. To address this problem, we created the `Teeny-Tiny Castle`, an open-source repository containing "_Educational tools for AI Ethics and Safety Research_." There, the developer can find many examples of how to use programming tools (functions, classes, libraries, etc.) to work with and deal with various problems raised in the literature (e.g., Algorithmic Discrimination, Model Opacity, etc.). At the moment, our repository has several examples of how to work ethically and safely with AI, using one of the most widely used languages in the community (`Python`). Our lines of focus are on issues related to "**Accountability & Sutentability**" "**Interpretability**" "**Robustness/Adversarial**" "**Fairness**" and "**Cybersecurity**", all being worked through examples that refer to some of the most common contemporary AI applications (e.g., _Computer Vision, Natural language Processing, Synthetic Data Generation, Classification & Forecasting_, etc.).

You can also find an **introductory course on ML** organized by the [AIRES at PUCRS](https://www.airespucrs.org/). To run the notebooks just open them in your Google Drive as a **Colab Notebook**, or as a **Jupyter notebook**. You can also follow our [Python and VS Code installation tutorial](python_VS-code_installation.html) if you want to run these notebooks on your own workstation. All requirements are specified in the [requirements.txt](requirements.txt) file. Python version is `3.9.13`.

If you would like to enable your GPU to follow our notebooks, you will need to install the **drivers related to your NVIDIA/Cuda/TPU/GPU**. If you use NVIDIA, you will also need the NVIDIA Toolkit and cudaDNN. You can find a comprehensive guide on how to configure your NVIDIA card in [this tutorial](https://www.youtube.com/watch?v=IubEtS2JAiY) (_- by [deeplizard](https://www.youtube.com/c/deeplizard) -_).

Join [AIRES at PUCRS](https://en.airespucrs.org/contato).

## AI Ethics ⚖️🤖⚖️

- Learn about the **state-of-the-art in AI Ethics** by browsing our [Dash](https://aires-worldwide-ai-ethics-en.onrender.com/) or [Power BI](https://en.airespucrs.org/worldwide-ai-ethics) dashboard;
- Learn about the **[major organizations pushing AI R&D](https://gcrinstitute.org/2020-survey-of-artificial-general-intelligence-projects-for-ethics-risk-and-policy/)** and their commitment to **AI Safety research** in this [dasboard](https://aires-risk-monitoring.onrender.com/);
- Track the current developments in the AI field with the **[AI Tracker](https://www.aitracker.org/)** tool, an online table that lists the largest (and most proficient) models ever created, along with an analysis of the potential risks associated with each model being tracked (developed by _[Gladstone AI](https://www.gladstone.ai/)_);
- **Interested in AI regulation?** On September 29, 2021, the **Chamber of Deputies of the Federative Republic of Brazil** approved _Bill n. 21/2020_, which establishes foundations and principles for the development and application of artificial intelligence (AI) in Brazil (together with _Bills 5051/2019_ and _872/2021_). Here you can find a [technical report](https://en.airespucrs.org/nota-tecnica-aires) (_Portuguese only...sorry_) structured from the main topics addressed in the three bills. This report was made by a collaboration between [AIRES at PUCRS](https://en.airespucrs.org/) and the Post-Graduate Program in Law (PPGD), linked to the PUCRS School of Law.

## Introduction Course on ML 📈

- If you want to learn how to **build your own workstation**, check this [Python and VS Code installation guide](xxx);
- Here you can find a [Basic Python Tutorial](xxx) to **get you started on the language and syntax** we are using in the available notebooks in this repository;
- Some of the **most utilized libraries in ML and Data Science** (if you are a 🐍) are `Pandas`, `Scikit-learn`, and `Numpy`. [Here you can find a brief tutorial](xxx) on how to use some of the (many) features and functions of these libraries;
- For many practitioners, **the "Hello World" of Deep Learning is classifying one of the MNIST datasets**. Here you can find a Basic `Keras` and `Tensor Flow` tutorial using the [Digit-MNIST](xxx) and [Fashion-MNIST](xxx) datasets;
- But if you want to check the `Pytorch` implementation of the same algorithm we built in the last Keras/TensorFlow tutorial, go to [this notebook](xxx);
- One of the most famous problems in ML (_- actually Neural Networks -_) history is the [`XOR` Problem](xxx). Here you can find an example of how to [implement a Multi-Layer Perceptron using only NumPy](xxx) to **solve the XOR problem**;
- Here you can find a notebook showing how to implement a **Feed-Forward Neural Network** using `NumPy`. You will [learn the inner workings of many of the pre-built functions available in libraries like `TensorFlow` and `Pytorch`](xxx);
- One of **the most basic ML models you can build is a Linear Regression (LR) model**. Here you will find how to [build an LR model from scratch, implementing the `Gradient Descent` algorithm using only `NumPy`](xxx);
- Here you can find a **visual representation** of how the [Gradient Descent algorithm works](xxx) while trying to find the correct model that approximates a `∿ mystery function ∿`;
- Among the different techniques that we find within Machine Learning, **Reinforcement Learning deserves some introduction**. As much as this methodology is _not yet mainstream_, RL was the paradigm responsible for creating some of the most _[general](https://arxiv.org/abs/2003.13350) and [proficient](https://arxiv.org/abs/1712.01815) agents of [today](https://www.nature.com/articles/s41586-021-03819-2)_. If you want to understand some of the basics behind RL, we have provided two notebooks where we trained an agent to deal with the [n-armed-bandit problem](xxx) (_a classic RL introduction problem_), and OpenAI Gym's [FrozenLake](xxx) environment (a simple environment to learn the basics of $Q$-learning);
- One of the most common applications of AI systems in industry is in creating **recommendation systems**. Learn how to [build three different types of recommendation systems with this notebook](xxx);
- Nowadays, (_almost_) everything is a transformer model, and one of the areas that have taken the most advantage of this is NLP (_Natural Language Processing_). [Create your own encoder-transformer model](xxx) to tackle a problem of text classification (more specifically, _toxicity detection_).
- Learn how to work with sequence-to-sequence tasks using RNNs and the [original (encoder-decoder) version of the transformer architecture](xxx).
- Learn how to keep track of your experiments, and analyze your model training with [TensorBoard](xxx);
- **Hyperparameter optimization** is crucial for boosting the performance of deep learning models. Tune them right to improve accuracy, prevent overfitting, and get the most out of your model. [Learn how to perform hyperparameter optimization](xxx) using `Hyperopt` and `Hyperas`.
- Get access to **all the datasets you might need** during the beginning of your ML journey by using the **[TensorFlow Datasets](xxx)**;
- Training and testing ML models are only a part of the [MLOps](https://en.wikipedia.org/wiki/MLOps) cycle. **Learn to [deploy simple AI applications](xxx) using `gradio`;**
- One of the main operations computed during training and inference of an ML model is the **dot product between vectors/matrices**. Here you can find an explanation of [what the `doot product` is](xxx) and why it is a useful piece of linear algebra;
- **Tensor manipulation** is a big part of what we do (_- the algorithms and processors do -_) in ML. Learn about tensors and tensor operations in [this notebook](xxx) using the `Pytorch` library;
- If you are **lost in the jargon**, this [glossary](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/87b1e85fa62074af1459e863cac986dc973b4666/ML%20Intro%20Course/glossary.md) can help you get familiar with some of the main terms we use in ML;
- To understand **how gradient descent updates the parameters of an ML model** using the opposite direction of the gradient of the loss function, understanding [what is a `derivative`](xxx) can help you a lot;
- Here are all the [`requirements`](xxx) (_and how to install them_) needed for the notebooks in this repository.

## Accountability & Sutentability ♻️

- **Model cards** offer a simple implementation to ensure **transparency and accountability** in the AI community. Learn [how to generate model cards in this notebook](xxx);
- Here you can find a notebook showing how to turn the [CO2 emission data generated by `CodeCarbon`](xxx) into a **Markdown CO2 emission report**.

## Interpretability with CV 🔎🖼️

- If you want to [create your own CNN to explore](xxx) (using the below notebooks), you can use this simple `CNN_model_maker` for the **CIFAR-10 dataset** (_this may take a while if you don't have a GPU. If you don't have GPU, limit your training phase to < 50 epochs_);
- Learn to explore and **interpret the inner workings of a CNN** by using [feature visualization techniques](xxx);
- Learn to interpret the output of CNN models by using [saliency mapping techniques](xxx);
- Learn to [interpret the inner workings of a CNN](xxx) using the `LIME` (_Local Interpretable Model-Agnostic Explanations_) library.
- _Diffusion models_ are one of the current **paradigms in synthetic data generation** (especially when it comes to photorealism). Learn how to [interpret diffusion models](xxx) with the `diffusers-interpret` library.

## Interpretability with NLP 🔎📚

- If you want to [create your own NLP model to explore](xxx) (using the below notebooks), you can use this simple `NLP_model_maker` (_available in [Portuguese](xxx) and [English](xxx)_) to **create your own language model**. We also provide a simple [GUI interface](xxx), created using `Dash` and `Flask` to allow you to easily interact with your model;
- Learn to **explore and interpret the outputs of a Language Model** using the `LIME` (_Local Interpretable Model-Agnostic Explanations_) library (_available in [Portuguese](xxx) and [English](xxx)_);
- Learn to interpret the output of Keras NLP models (like RNNs) using the [Integrated Gradients](xxx) method (_available in [Portuguese](xxx) and [English](xxx)_);
- Learn to explore text datasets using **Text Mining**. Here you will learn simple techniques to create visualization tools to interpret the distribution of patterns (_e.g., sentiment, word recurrence_) in a text corpus (_available in [Portuguese](xxx) and [English](xxx)_);
- **Ever wanted to build your own _Open-AI-style-NLP playground?_** Here you can find a working [dashboard that allows you to interact with LM and Report your findings](xxx). The GUI was created using `Dash` and `Flask`. You can [download different models](xxx) from the `Hugging Face` library and feed this Dashboard/Playground with the model you so choose (_[DistillGPT2](https://huggingface.co/distilgpt2) is a great choice for people without GPU access_). This playground was specifically created for experiments involving prompt engineering (_available in [English](xxx)_). When running the `my_app.py` file, you will run this app on your browser (hosted on your local machine).

## Interpretability in Classification & Prediction with Tabular Data 🔎📊

- **Many of the models used in commercial applications of ML are basically classification and regression models** that work with `tabular data`. Here you can find examples of how to [interpret divergent classifications using `LIME` and `DALEX`](xxx), two libraries aimed and providing explainable AI (XAI);
- Investigate the **COMPAS Recidivism Racial Bias dataset**, and use **ML Fairness** and **XAI tools** to (1) [create an interpretable model](xxx), and (2) [understand the classifications of the algorithms created](xxx).

## Machine Learning Fairness ⚖️

- **Learn how to measure the "fairness" of an ML model** by using [fairness metrics in the _Credit Cart Dataset_](xxx);
- Use the [AIF360](https://aif360.mybluemix.net/) library to **correct a dataset** (_- the Hogwarts case -_) using the [Disparate Impact Remover](xxx);
- Learn how to apply the [Ceteris paribus principle](xxx) to create a "**_What-if_**" model. Evaluate a classifier using **Counterfactual Fairness**.

## Cybersecurity 👾

- Here you can find an example of a _[really bad PHP](xxx)_ forum to **practice XSS** (and learn how not to build a PHP backend server, i.e., - _without escaping your strings_ - ). _Try to get to the users_passwords_dump file using only your browser!_ To be able to run and attack this page, you will need to host it on your own server (_localhost_). We recommend [Wampserver](https://www.wampserver.com/en/) for this job (**YOU SHOULD NOT PERFORM XSS IN WEBSITES YOU DON'T HAVE EXPLICIT PERMISSION**);
- Learn a little bit about **password security** by [cracking bad passwords encrypted whit unsafe hashes](xxx) (**YOU SHOULD NOT PERFORM PASSWORD CRACKING AGAINST HASHES YOU HAVE NO BUSINESS OWNING**);
- Learn how to use **_Deep Leaning_** to [classify/detect malware](xxx), among other techniques to explore and evaluate possibly malicious software (_PE analysis, Reverse engineering, Automated Virus Scanning_).

## Adversarial Machine Learning 🐱‍💻

- Learn about the [pickle exploit](xxx), and how it can be used to **embed malicious** code into ML models (**YOU SHOULD NEVER CREATE MALICIOUS CODE TO BE USED AGAINST OTHERS**);
- **Evasion attacks** are perhaps the best-known type of attack in ML safety. Also known as "_adversarial examples_", [these attacks are carefully perturbed input samples that can completely throw off an ML model](xxx). Learn to craft adversarial examples using the `SecML` toolkit;
- Learn to [craft adversarial examples against CNNs through the Fast Sign Gradient Method](xxx) using nothing but the utilities of the Keras/TensorFlow libraries;
- Language models are the cornerstone of many commercial applications. Learn how to [generate adversarial examples against language models](xxx) using the `textattack` library;
- **Model extraction attacks** pose a threat to intellectual property and privacy. Taking a proactive and adversarial approach to protecting ML systems, in this notebook, we [illustrate the inner workings of an extraction/cloning attack](xxx);
- Data poisoning attacks, like _[label-flipping](xxx)_ and _[backdoor attacks](xxx)_, can severely degrade model performance while giving an attacker the chance to introduce hidden functionalities and features into ML models. In [this notebook](xxx), you can learn about these types of attacks.
- Adversarial training is one of the strategies that ML model developers can use to make their models more robust. In this notebook, you will [learn how to perform adversarial training](xxx) (using _FGSM_) with CNNs;
- And in this notebook, you will [learn how to perform adversarial training on language models](xxx) using `textattack`.

## How to cite this repository 😊

```Markdown

@misc{teenytinycastle,
  doi = {10.5281/zenodo.7112065},
  url = {https://github.com/Nkluge-correa/teeny-tiny_castle},
  author = {Nicholas Kluge Corr{\^e}a},
  title = {Teeny-Tiny Castle},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  note = {Last updated 26 December 2022},
}

```

---

This repository was built as part of the RAIES ([Rede de Inteligência Artificial Ética e Segura](https://www.raies.org/)) initiative, a project supported by FAPERGS - ([Fundação de Amparo à Pesquisa do Estado do Rio Grande do Sul](https://fapergs.rs.gov.br/inicial)), Brazil.
