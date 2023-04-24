# Teeny-Tiny Castle 🏰

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7485126.svg)](https://doi.org/10.5281/zenodo.7485126)[
![made-with-python](https://camo.githubusercontent.com/f9010d0d18143896d2e496fe0e0c89056acab8229dbdf169f1d3a4759567fe63/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f4d616465253230776974682d507974686f6e2d3166343235662e737667)](https://www.python.org/)

![python](https://i.gifer.com/origin/25/25dcee6c68fdb7ac42019def1083b2ef_w200.gif)

AI Ethics and Safety are (_relatively_) new fields, and their tools (and how to handle them) are still _not known to most of the development community_. To address this problem, we created the `Teeny-Tiny Castle`, an open-source repository containing "_Educational tools for AI Ethics and Safety Research_." There, the developer can find many examples of how to use programming tools (like functions, classes, libraries, etc.) to work with and deal with various problems raised in the literature (e.g., algorithmic discrimination, model opacity, etc.).

At the moment, our repository has several examples of how to work ethically and safely with AI, using one of the most widely used languages in the community (`Python`). Our lines of focus are on issues related to `Accountability & Sustainability`, `Interpretability`, `Robustness/Adversarial`, `Fairness`, and `Cybersecurity`, all being worked through examples that refer to some of the most common contemporary AI applications (e.g., _Computer Vision, Natural language Processing, Synthetic Data Generation, Classification & Forecasting_, etc.).

You can also find an _introductory course on ML_ organized by the [`AIRES at PUCRS`](https://www.airespucrs.org/). To run the notebooks just open them in your Google Drive as a `Colab Notebook`, or as a `Jupyter Notebook`. You can also follow our [Python and VS Code installation tutorial](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/bbe9c0a77499fa68de7c6d53bf5ef7e0b43a25e0/ML%20Intro%20Course/python_VS-code_installation.md) if you want to run these notebooks on your own workstation. All requirements are specified in the [`requirements.txt`](requirements.txt) file. Python version is `3.9.13`.

If you would like to enable your GPU to follow our notebooks, you will need to install the drivers related to your NVIDIA/Cuda/TPU/GPU. If you use NVIDIA, you will also need the `NVIDIA Toolkit` and `cudaDNN`. Windows users can find a comprehensive guide on how to configure your NVIDIA card in [this tutorial](https://www.youtube.com/watch?v=IubEtS2JAiY) (_- by [deeplizard](https://www.youtube.com/c/deeplizard) -_).

Join [AIRES at PUCRS](https://en.airespucrs.org/contato).

## AI Ethics ⚖️🤖⚖️

- Learn about the _state-of-the-art in AI Ethics_ by browsing our [Dash](https://aires-playground.herokuapp.com/worldwide-ai-ethics) or [Power BI](https://en.airespucrs.org/worldwide-ai-ethics) dashboard.
- Learn about the most recent published AI models, like "_what are their capabilities and potential risks?_", by accessing our [model library](https://aires-playground.herokuapp.com/model-library).
- Interested in _AI regulation?_ On September 29, 2021, the _Chamber of Deputies of the Federative Republic of Brazil_ approved `Bill n. 21/2020`, which establishes foundations and principles for the development and application of artificial intelligence (AI) in Brazil (together with `Bills 5051/2019` and `872/2021`). Here you can find a [technical report](https://en.airespucrs.org/nota-tecnica-aires) (_Portuguese only_) structured from the main topics addressed in the three bills. This report was made by a collaboration between [AIRES at PUCRS](https://en.airespucrs.org/) and the _Post-Graduate Program in Law_, linked to the PUCRS School of Law.
- And if you are interested in the ethical, legal, and technical problems related to the fair, responsible, and transparent use of `facial recognition technologies`, [AIRES at PUCRS](https://en.airespucrs.org/) and [RAIES](https://www.raies.org/en) have made available [this technical note](https://www.airespucrs.org/nota-tecnica-frt) on the matter (_Portuguese only_).

## Introduction Course on ML 📈

- If you want to learn how to build your own workstation, check this [Python and VS Code installation guide](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/fa17764aa8800c388d0d298b750c686757e0861e/ML%20Intro%20Course/1_python_VS_code_installation.md).
- Here you can find a [Basic Python Tutorial](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/fa17764aa8800c388d0d298b750c686757e0861e/ML%20Intro%20Course/2_Basic_Python_Tutorial.ipynb) to get you started on the language and syntax we are using in the available notebooks in this repository.
- Some of the _most utilized libraries in ML and Data Science_ (if you are a 🐍) are `Pandas`, `Scikit-learn`, and `Numpy`. [Here you can find a brief tutorial](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/fa17764aa8800c388d0d298b750c686757e0861e/ML%20Intro%20Course/3_Basic_Pandas_Scikit_learn_NumPy_Tutorial.ipynb) on how to use some of the (many) features and functions of these libraries.
- If you are lost in the jargon, this [`glossary`](https://aires-playground.herokuapp.com/glossary) can help you get familiar with some of the main terms we use in ML.
- Gradient Descent is one of the foundations of ML. Here you can find a tutorial on how to implement the [Gradient Descent algorithm](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/fa17764aa8800c388d0d298b750c686757e0861e/ML%20Intro%20Course/4_gradient_descent.ipynb) while trying to find the correct model that approximates a `∿ mystery function ∿`.
- One of the most basic ML models you can build is a `Linear Regression` model. Here you will find how to [build an LR model from scratch, implementing the `Gradient Descent` algorithm using only `NumPy`](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/fa17764aa8800c388d0d298b750c686757e0861e/ML%20Intro%20Course/5_linear_regression_numpy.ipynb).
- One of the most famous problems in ML (_- actually Neural Networks -_) history is the `XOR` Problem. Here you can find an example of how to [implement a Multi-Layer Perceptron using only NumPy](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/fa17764aa8800c388d0d298b750c686757e0861e/ML%20Intro%20Course/6_xor_problem.ipynb) to _solve the XOR problem_.
- Here you can find a notebook showing how to implement a `Feed-Forward Neural Network` using `NumPy`. You will [learn the inner workings of many of the pre-built functions available in libraries like `TensorFlow` and `Pytorch`](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/fa17764aa8800c388d0d298b750c686757e0861e/ML%20Intro%20Course/7_MNIST_numpy.ipynb).
- For many practitioners, the "_Hello World!_" of Deep Learning is classifying one of the `MNIST` datasets. Here you can find a Basic `Keras` and `Tensor Flow` tutorial using the [Fashion-MNIST](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/fa17764aa8800c388d0d298b750c686757e0861e/ML%20Intro%20Course/8_Fashion_MNIST.ipynb) dataset.
- But if you want to check the `Pytorch` implementation of the same algorithm we built in the last Keras/TensorFlow tutorial, go to [this notebook](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/fa17764aa8800c388d0d298b750c686757e0861e/ML%20Intro%20Course/9_MNIST_torch.ipynb).
- `Hyperparameter optimization` is crucial for boosting the performance of deep learning models. Tune them right to improve accuracy, prevent overfitting, and get the most out of your model. [Learn how to perform hyperparameter optimization](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/fa17764aa8800c388d0d298b750c686757e0861e/ML%20Intro%20Course/10_hyperparameter_tuning.ipynb) using `Hyperopt` and `Hyperas`.
- Get access to _all the datasets you might need_ during the beginning of your ML journey by using the [TensorFlow Datasets](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/fa17764aa8800c388d0d298b750c686757e0861e/ML%20Intro%20Course/11_using_tfds.ipynb).
- Learn to keep track of your ML experiments using [`Tensorboard`](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/fa17764aa8800c388d0d298b750c686757e0861e/ML%20Intro%20Course/12_tensorboard_visualization.ipynb).
- One of the most common applications of AI systems in industry is in creating `recommendation systems`. Learn how to [build three different types of recommendation systems with this notebook](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/fa17764aa8800c388d0d298b750c686757e0861e/ML%20Intro%20Course/13_recommender_systems.ipynb).
- Time Series Forecasting is one of the major applications of ML. Learn how to [create forecasting algorithms](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/fa17764aa8800c388d0d298b750c686757e0861e/ML%20Intro%20Course/14_time_series_forecasting.ipynb) with `XGBoost`.
- Nowadays, (_almost_) everything is a transformer model, and one of the areas that have taken the most advantage of this is NLP (_Natural Language Processing_). [Create your own `encoder-transformer` model](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/fa17764aa8800c388d0d298b750c686757e0861e/ML%20Intro%20Course/15_toxicity_detection.ipynb) to tackle a problem of text classification (more specifically, _toxicity detection_).
- Learn how to work with sequence-to-sequence tasks using RNNs and the [original (`encoder-decoder`) version of the `transformer` architecture](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/fa17764aa8800c388d0d298b750c686757e0861e/ML%20Intro%20Course/16_seuqnece_to_sequence.ipynb).
- Generative models have been a focus in ML in recent years. [Create your own `language model`](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/fa17764aa8800c388d0d298b750c686757e0861e/ML%20Intro%20Course/17_text_generation.ipynb) using a miniature version of the GPT architecture.
- Among the different techniques that we find within Machine Learning, `Reinforcement Learning` deserves some introduction. As much as this methodology is _not yet mainstream_, RL was the paradigm responsible for creating some of the most _[general](https://arxiv.org/abs/2003.13350) and [proficient](https://arxiv.org/abs/1712.01815) agents of [today](https://www.nature.com/articles/s41586-021-03819-2)_. If you want to understand some of the basics behind RL, we have provided two notebooks where we trained an agent to deal with the [`FrozenLake`](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/fa17764aa8800c388d0d298b750c686757e0861e/ML%20Intro%20Course/18_Q_learning.ipynb).
- Training and testing ML models are only a part of the [`MLOps`](https://en.wikipedia.org/wiki/MLOps) cycle. Learn to [deploy simple AI applications](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/fa17764aa8800c388d0d298b750c686757e0861e/ML%20Intro%20Course/19_quick_AI_app.ipynb) using `gradio`.
- Master the basics of `MLOps` by [deploying our `XGBoost` regression model as an API](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/fa17764aa8800c388d0d298b750c686757e0861e/ML%20Intro%20Course/20_ML_api_deployment.ipynb) using `FastAPI`.

## Accountability & Sutentability ♻️

- `Model cards` offer a simple implementation to ensure `transparency` and `accountability` in the AI community. Learn [how to generate model cards in this notebook](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/fa17764aa8800c388d0d298b750c686757e0861e/ML%20Accountability/Model%20Cards/model_card_generator.ipynb).
- Here you can find a notebook showing how to turn the [CO2 emission data generated by `CodeCarbon`](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/fa17764aa8800c388d0d298b750c686757e0861e/ML%20Accountability/CO2%20Emission%20Report/emission_tracker.ipynb) into a _Markdown CO2 emission report_.
- How to deal with the [tradeoff between accuracy and sustainability](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/a57a675d3e32593382f4dfb4c7a8deb331b9dc18/ML%20Accountability/CO2%20Emission%20Report/carbon_emission_cv.ipynb)? Deal with this dilemma while training `CNNs` for medical applications.

## Interpretability with CV 🔎🖼️

- If you want to [create your own CNN to explore](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/fa17764aa8800c388d0d298b750c686757e0861e/ML%20Explainability/CV%20Interpreter/CNN_model_maker.ipynb) (using the below notebooks), you can use this simple `CNN_model_maker` for the `CIFAR-10` dataset (_this may take a while if you don't have a GPU. If you don't have GPU, limit your training phase to < 50 epochs_).
- Learn to explore and interpret the inner workings of a `CNN` by using [feature visualization techniques](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/fa17764aa8800c388d0d298b750c686757e0861e/ML%20Explainability/CV%20Interpreter/CNN_feature_visualization.ipynb).
- Create [maximum activations for the imagnet classes](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/b54bee4a191d151d682d380886deb01fc21eb19d/ML%20Explainability/CV%20Interpreter/CNN_activation_maximization.ipynb) using `MobileNetV2` and feature extraction techniques.
- Learn to interpret the output of `CNN` models by using [saliency mapping techniques](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/fa17764aa8800c388d0d298b750c686757e0861e/ML%20Explainability/CV%20Interpreter/CNN_attribution_maps.ipynb).
- Learn to [interpret the inner workings of a `CNN`](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/fa17764aa8800c388d0d298b750c686757e0861e/ML%20Explainability/CV%20Interpreter/CNN_attribution_maps_with_LIME.ipynb) using the `LIME` (_Local Interpretable Model-Agnostic Explanations_) library.
- _Diffusion models_ are one of the current _paradigms in synthetic data generation_ (especially when it comes to photorealism). Learn how to [interpret diffusion models](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/fa17764aa8800c388d0d298b750c686757e0861e/ML%20Explainability/CV%20Interpreter/diffusion_interpreter.ipynb) with the `diffusers-interpret` library.

## Interpretability with NLP 🔎📚

- If you want to create your own NLP model to explore (using the below notebooks), you can use this simple `NLP_model_maker` (_available in datasets come in Portuguese and in English_) to create your own `language model`. We also provide a simple [UI interface](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/fa17764aa8800c388d0d298b750c686757e0861e/ML%20Explainability/NLP%20Interpreter/model_maker.ipynb), created using `Dash` and `Flask` to allow you to easily interact with your model.
- Learn to explore and interpret the outputs of a `language model` using the `LIME` (_[Local Interpretable Model-Agnostic Explanations](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/fa17764aa8800c388d0d298b750c686757e0861e/ML%20Explainability/NLP%20Interpreter/lime_for_NLP.ipynb)_) library.
- Learn to interpret the output of `Keras` NLP models (like `RNNs`) using the [`Integrated Gradients`](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/fa17764aa8800c388d0d298b750c686757e0861e/ML%20Explainability/NLP%20Interpreter/integrated_gradients_in%20_keras_nlp.ipynb) method.
- Learn about how `word embeddings` are created, and how can you use them to perform interpretability analysis on text data/language models, by [implementing a `word2vector` model](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/e5ec7cc503d532c44ecb29df1583d4e3cca4cb81/ML%20Explainability/NLP%20Interpreter/word2vec.ipynb).
- Discover how language models embed meaning into vectors by [exploring the similarities and the geometric landscape](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/2da13285e775560b432a1e3d3560a49e37db31d0/ML%20Explainability/NLP%20Interpreter/investigating_word_embeddings.ipynb) of `word embeddings` and `embedding layers`.
- Learn to [explore text datasets using `Text Mining`](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/fa17764aa8800c388d0d298b750c686757e0861e/ML%20Explainability/NLP%20Interpreter/text_mining.ipynb). Here you will learn simple techniques to create visualization tools to interpret the distribution of patterns (_e.g., sentiment, word recurrence_) in a text corpus.
- Ever wanted to build your own _language model playground?_ Here you can find a working [dash.app that allows you to interact with LM and Report your findings](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/fa17764aa8800c388d0d298b750c686757e0861e/ML%20Explainability/NLP%20Playgroung/playground.py). The UI was created using `Dash` and `Flask`. You can [download different models](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/fa17764aa8800c388d0d298b750c686757e0861e/ML%20Explainability/NLP%20Playgroung/get_transformer.ipynb) from the `Hugging Face` library and feed this Playground with the model you so choose. This playground was specifically created for experiments involving `prompt engineering` and `human feedback`. You can find an online working example of this application in [this link](https://playground.airespucrs.org/language-model-playground).

## Interpretability in Classification & Prediction with Tabular Data 🔎📊

- Many of the models used in commercial applications of ML are basically classification and regression models that work with `tabular data`. Here you can find examples of how to [interpret divergent classifications using `LIME` and `DALEX`](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/fa17764aa8800c388d0d298b750c686757e0861e/ML%20Explainability/Tabular%20Interpreter/interpreter_for_tabular.ipynb), two libraries aimed and providing explainable AI (XAI).
- Investigate the COMPAS Recidivism Racial Bias dataset, and use ML Fairness and XAI tools to (1) [create an interpretable model](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/fa17764aa8800c388d0d298b750c686757e0861e/ML%20Explainability/Tabular%20Interpreter/fairness_xai_COMPAS.ipynb), and (2) [understand the classifications of the algorithms created](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/fa17764aa8800c388d0d298b750c686757e0861e/ML%20Explainability/Tabular%20Interpreter/fairness_xai_COMPAS.ipynb).

## Machine Learning Fairness ⚖️

- Learn how to measure the "_fairness_" of an ML model by using [fairness metrics in the `Credit Cart Dataset`](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/fa17764aa8800c388d0d298b750c686757e0861e/ML%20Fairness/fair_metrics_Credit_card_approval.ipynb).
- When dealing with certain datasets, like the `Adult Census Dataset` we learn that [interpreting fairness metrics, and choosing which one to use](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/fa17764aa8800c388d0d298b750c686757e0861e/ML%20Fairness/fairness_income.ipynb), can be a challenge.
- Use the [`AIF360`](https://aif360.mybluemix.net/) library to correct a dataset (_- the Hogwarts case -_) using the [Disparate Impact Remover](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/fa17764aa8800c388d0d298b750c686757e0861e/ML%20Fairness/disparate_impact_remove_Hogwarts.ipynb).
- Learn how to apply the [`Ceteris paribus`](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/fa17764aa8800c388d0d298b750c686757e0861e/ML%20Fairness/ceteris_paribus_profiles.ipynb) principle to create a "_What-if_" model. Evaluate a classifier using `Counterfactual Fairness`.

## Cybersecurity 👾

- Learn a little bit about `password security` by [cracking bad passwords encrypted whit unsafe hashes](https://github.com/Nkluge-correa/password_cracking_dash) (⚠️YOU SHOULD NOT PERFORM PASSWORD CRACKING AGAINST HASHES YOU HAVE NO BUSINESS OWNING⚠️).
- Learn how to use _Deep Learning_ to [classify/detect malware](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/fa17764aa8800c388d0d298b750c686757e0861e/ML%20&%20Cybersecurity/Malware_detection.ipynb), among other techniques to explore and evaluate possibly malicious software (_PE analysis, Reverse engineering, Automated Virus Scanning_).

## Adversarial Machine Learning 🐱‍💻

- Learn about the [pickle exploit](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/fa17764aa8800c388d0d298b750c686757e0861e/ML%20Adversarial/the_pickle_exploit.ipynb), and how it can be used to embed malicious code into ML models (⚠️YOU SHOULD NEVER CREATE MALICIOUS CODE TO BE USED AGAINST OTHERS⚠️).
- Evasion attacks are perhaps the best-known type of attack in ML safety. Also known as `adversarial examples`, [these attacks are carefully perturbed input samples that can completely throw off an ML model](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/fa17764aa8800c388d0d298b750c686757e0861e/ML%20Adversarial/evasion_attacks.ipynb). Learn to craft adversarial examples using the `SecML` toolkit.
- Learn to [craft `adversarial examples` against `CNNs` through the `Fast Sign Gradient Method`](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/fa17764aa8800c388d0d298b750c686757e0861e/ML%20Adversarial/evasion_attacks_FGSM.ipynb) using nothing but the utilities of the Keras/TensorFlow libraries.
- Language models are the cornerstone of many commercial applications. Learn how to [generate adversarial examples against language models](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/fa17764aa8800c388d0d298b750c686757e0861e/ML%20Adversarial/adversarial_text_attack.ipynb) using the `textattack` library.
- Model `extraction attacks` pose a threat to intellectual property and privacy. Taking a proactive and adversarial approach to protecting ML systems, in this notebook, we [illustrate the inner workings of an extraction/cloning attack](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/fa17764aa8800c388d0d298b750c686757e0861e/ML%20Adversarial/model_extraction_nlp.ipynb).
- Data `poisoning attacks`, like _label-flipping_ and _backdoor attacks_, can severely degrade model performance while giving an attacker the chance to introduce hidden functionalities and features into ML models. In [this notebook](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/fa17764aa8800c388d0d298b750c686757e0861e/ML%20Adversarial/data_poisoning_attacks.ipynb), you can learn about these types of attacks.
- Adversarial training is one of the strategies that ML model developers can use to make their models more robust. In this notebook, you will [learn how to perform `adversarial training`](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/fa17764aa8800c388d0d298b750c686757e0861e/ML%20Adversarial/adversarial_training_cv.ipynb) (using _FGSM_) with `CNNs`.
- And in this notebook, you will [learn how to perform `adversarial training` on language models](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/fa17764aa8800c388d0d298b750c686757e0861e/ML%20Adversarial/adversarial_training_nlp.ipynb) using `textattack`.

## Cite as 🤗

```latex
@misc{teenytinycastle,
    doi = {10.5281/zenodo.7112065},
    url = {https://github.com/Nkluge-correa/teeny-tiny_castle},
    author = {Nicholas Kluge Corr{\^e}a},
    title = {Teeny-Tiny Castle},
    year = {2023},
    publisher = {GitHub},
    journal = {GitHub repository},
    note = {Last updated 24 April 2023},
}
```

---

This repository was built as part of the RAIES ([Rede de Inteligência Artificial Ética e Segura](https://www.raies.org/)) initiative, a project supported by FAPERGS - ([Fundação de Amparo à Pesquisa do Estado do Rio Grande do Sul](https://fapergs.rs.gov.br/inicial)), Brazil.
