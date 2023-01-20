# Teeny-Tiny Castle 🏰

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7485126.svg)](https://doi.org/10.5281/zenodo.7485126)[
![made-with-python](https://camo.githubusercontent.com/f9010d0d18143896d2e496fe0e0c89056acab8229dbdf169f1d3a4759567fe63/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f4d616465253230776974682d507974686f6e2d3166343235662e737667)](https://www.python.org/)

![python](https://i.gifer.com/origin/25/25dcee6c68fdb7ac42019def1083b2ef_w200.gif)

AI Ethics and Safety are (_relatively_) new fields, and their tools (and how to handle them) are still **not known to most of the development community**. To address this problem, we created the `Teeny-Tiny Castle`, an open-source repository containing "_Educational tools for AI Ethics and Safety Research_." There, the developer can find many examples of how to use programming tools (like functions, classes, libraries, etc.) to work with and deal with various problems raised in the literature (e.g., algorithmic discrimination, model opacity, etc.).

At the moment, our repository has several examples of how to work ethically and safely with AI, using one of the most widely used languages in the community (`Python`). Our lines of focus are on issues related to "**Accountability & Sutentability**" "**Interpretability**" "**Robustness/Adversarial**" "**Fairness**" and "**Cybersecurity**", all being worked through examples that refer to some of the most common contemporary AI applications (e.g., _Computer Vision, Natural language Processing, Synthetic Data Generation, Classification & Forecasting_, etc.).

You can also find an **introductory course on ML** organized by the [AIRES at PUCRS](https://www.airespucrs.org/). To run the notebooks just open them in your Google Drive as a **Colab Notebook**, or as a **Jupyter notebook**. You can also follow our [Python and VS Code installation tutorial](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/bbe9c0a77499fa68de7c6d53bf5ef7e0b43a25e0/ML%20Intro%20Course/python_VS-code_installation.md) if you want to run these notebooks on your own workstation. All requirements are specified in the [requirements.txt](requirements.txt) file. Python version is `3.9.13`.

If you would like to enable your GPU to follow our notebooks, you will need to install the **drivers related to your NVIDIA/Cuda/TPU/GPU**. If you use NVIDIA, you will also need the NVIDIA Toolkit and cudaDNN. You can find a comprehensive guide on how to configure your NVIDIA card in [this tutorial](https://www.youtube.com/watch?v=IubEtS2JAiY) (_- by [deeplizard](https://www.youtube.com/c/deeplizard) -_).

Join [AIRES at PUCRS](https://en.airespucrs.org/contato).

## AI Ethics ⚖️🤖⚖️

- Learn about the **state-of-the-art in AI Ethics** by browsing our [Dash](https://aires-worldwide-ai-ethics-en.onrender.com/) or [Power BI](https://en.airespucrs.org/worldwide-ai-ethics) dashboard;
- Learn about the **most recent published AI models**, like "_what are their capabilities and potential risks?_", by accessing our [model library](https://aires-risk-monitoring.onrender.com/) (_upcomming..._);
- **Interested in AI regulation?** On September 29, 2021, the **Chamber of Deputies of the Federative Republic of Brazil** approved _Bill n. 21/2020_, which establishes foundations and principles for the development and application of artificial intelligence (AI) in Brazil (together with _Bills 5051/2019_ and _872/2021_). Here you can find a [technical report](https://en.airespucrs.org/nota-tecnica-aires) (_Portuguese only...sorry_) structured from the main topics addressed in the three bills. This report was made by a collaboration between [AIRES at PUCRS](https://en.airespucrs.org/) and the Post-Graduate Program in Law (PPGD), linked to the PUCRS School of Law.
- And if you are interested in the ethical, legal, and technical problems related to the fair, responsible, and transparent use of **facial recognition technologies**, [AIRES at PUCRS](https://en.airespucrs.org/) and [RAIES](https://www.raies.org/en) have made available this technical note on the matter (_again...Portuguese only_).

## Introduction Course on ML 📈

- If you want to learn how to **build your own workstation**, check this [Python and VS Code installation guide](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/bbe9c0a77499fa68de7c6d53bf5ef7e0b43a25e0/ML%20Intro%20Course/python_VS-code_installation.md);
- Here you can find a [Basic Python Tutorial](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/bbe9c0a77499fa68de7c6d53bf5ef7e0b43a25e0/ML%20Intro%20Course/Basic_Python_Tutorial.ipynb) to **get you started on the language and syntax** we are using in the available notebooks in this repository;
- Some of the **most utilized libraries in ML and Data Science** (if you are a 🐍) are `Pandas`, `Scikit-learn`, and `Numpy`. [Here you can find a brief tutorial](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/bbe9c0a77499fa68de7c6d53bf5ef7e0b43a25e0/ML%20Intro%20Course/Basic_Pandas_Scikit-learn_NumPy_Tutorial.ipynb) on how to use some of the (many) features and functions of these libraries;
- For many practitioners, **the "_Hello World!_" of Deep Learning is classifying one of the MNIST datasets**. Here you can find a Basic `Keras` and `Tensor Flow` tutorial using the [Digit-MNIST](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/bbe9c0a77499fa68de7c6d53bf5ef7e0b43a25e0/ML%20Intro%20Course/MNIST_digit.ipynb) and [Fashion-MNIST](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/bbe9c0a77499fa68de7c6d53bf5ef7e0b43a25e0/ML%20Intro%20Course/Fashion_MNIST.ipynb) datasets;
- But if you want to check the `Pytorch` implementation of the same algorithm we built in the last Keras/TensorFlow tutorial, go to [this notebook](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/bbe9c0a77499fa68de7c6d53bf5ef7e0b43a25e0/ML%20Intro%20Course/MNIST_torch.ipynb);
- One of the most famous problems in ML (_- actually Neural Networks -_) history is the [`XOR` Problem](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/bbe9c0a77499fa68de7c6d53bf5ef7e0b43a25e0/ML%20Intro%20Course/xor.ipynb). Here you can find an example of how to [implement a Multi-Layer Perceptron using only NumPy](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/bbe9c0a77499fa68de7c6d53bf5ef7e0b43a25e0/ML%20Intro%20Course/xor.ipynb) to **solve the XOR problem**;
- Here you can find a notebook showing how to implement a **Feed-Forward Neural Network** using `NumPy`. You will [learn the inner workings of many of the pre-built functions available in libraries like `TensorFlow` and `Pytorch`](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/bbe9c0a77499fa68de7c6d53bf5ef7e0b43a25e0/ML%20Intro%20Course/MNIST_numpy.ipynb);
- One of **the most basic ML models you can build is a Linear Regression (LR) model**. Here you will find how to [build an LR model from scratch, implementing the `Gradient Descent` algorithm using only `NumPy`](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/bbe9c0a77499fa68de7c6d53bf5ef7e0b43a25e0/ML%20Intro%20Course/linear_regression_numpy.ipynb);
- Here you can find a **visual representation** of how the [Gradient Descent algorithm works](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/bbe9c0a77499fa68de7c6d53bf5ef7e0b43a25e0/ML%20Intro%20Course/gradient_descent.ipynb) while trying to find the correct model that approximates a `∿ mystery function ∿`;
- Among the different techniques that we find within Machine Learning, **Reinforcement Learning deserves some introduction**. As much as this methodology is _not yet mainstream_, RL was the paradigm responsible for creating some of the most _[general](https://arxiv.org/abs/2003.13350) and [proficient](https://arxiv.org/abs/1712.01815) agents of [today](https://www.nature.com/articles/s41586-021-03819-2)_. If you want to understand some of the basics behind RL, we have provided two notebooks where we trained an agent to deal with the [n-armed-bandit problem](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/a5a51e46dc0e04cb4cb795503f14200374f62e52/ML%20Intro%20Course/n-armed_bandit.ipynb) (_a classic RL introduction problem_), and OpenAI Gym's [FrozenLake](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/master/ML%20Intro%20Course/Q-learning.ipynb);
- One of the most common applications of AI systems in industry is in creating **recommendation systems**. Learn how to [build three different types of recommendation systems with this notebook](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/bbe9c0a77499fa68de7c6d53bf5ef7e0b43a25e0/ML%20Intro%20Course/recommender.ipynb);
- Nowadays, (_almost_) everything is a transformer model, and one of the areas that have taken the most advantage of this is NLP (_Natural Language Processing_). [Create your own encoder-transformer model](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/bbe9c0a77499fa68de7c6d53bf5ef7e0b43a25e0/ML%20Intro%20Course/toxicity_detection.ipynb) to tackle a problem of text classification (more specifically, _toxicity detection_).
- Learn how to work with sequence-to-sequence tasks using RNNs and the [original (encoder-decoder) version of the transformer architecture](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/bbe9c0a77499fa68de7c6d53bf5ef7e0b43a25e0/ML%20Intro%20Course/seuqnece-to-sequence.ipynb).
- Generative models have been a focus in ML in recent years. [Create your own language model](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/f0d2fb3d6f875b84b9108f6f62dc550766302bcc/ML%20Intro%20Course/text_generation.ipynb) using a miniature version of the GPT architecture.
- Learn how to keep track of your experiments, and analyze your model training with [TensorBoard](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/bbe9c0a77499fa68de7c6d53bf5ef7e0b43a25e0/ML%20Intro%20Course/seuqnece-to-sequence.ipynb);
- **Hyperparameter optimization** is crucial for boosting the performance of deep learning models. Tune them right to improve accuracy, prevent overfitting, and get the most out of your model. [Learn how to perform hyperparameter optimization](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/bbe9c0a77499fa68de7c6d53bf5ef7e0b43a25e0/ML%20Intro%20Course/hype.ipynb) using `Hyperopt` and `Hyperas`.
- Get access to **all the datasets you might need** during the beginning of your ML journey by using the [TensorFlow Datasets](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/bbe9c0a77499fa68de7c6d53bf5ef7e0b43a25e0/ML%20Intro%20Course/using_tfds.ipynb);
- Training and testing ML models are only a part of the [MLOps](https://en.wikipedia.org/wiki/MLOps) cycle. Learn to [deploy simple AI applications](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/bbe9c0a77499fa68de7c6d53bf5ef7e0b43a25e0/ML%20Intro%20Course/quick_AI_app.ipynb) using `gradio`;
- One of the main operations computed during training and inference of an ML model is the **dot product between vectors/matrices**. Here you can find an explanation of [what the `doot product` is](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/bbe9c0a77499fa68de7c6d53bf5ef7e0b43a25e0/ML%20Intro%20Course/dot_product.ipynb) and why it is a useful piece of linear algebra;
- **Tensor manipulation** is a big part of what we do (_- the algorithms and processors do -_) in ML. Learn about tensors and tensor operations in [this notebook](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/bbe9c0a77499fa68de7c6d53bf5ef7e0b43a25e0/ML%20Intro%20Course/working_with_tensors.ipynb) using the `Pytorch` library;
- If you are **lost in the jargon**, this [glossary](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/87b1e85fa62074af1459e863cac986dc973b4666/ML%20Intro%20Course/glossary.md) can help you get familiar with some of the main terms we use in ML;
- To understand **how gradient descent updates the parameters of an ML model** using the opposite direction of the gradient of the loss function, understanding [what is a `derivative`](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/bbe9c0a77499fa68de7c6d53bf5ef7e0b43a25e0/ML%20Intro%20Course/derivative.ipynb) can help you a lot;
- Here are all the [`requirements`](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/bbe9c0a77499fa68de7c6d53bf5ef7e0b43a25e0/requirements.md) (_and how to install them_) needed for the notebooks in this repository.

## Accountability & Sutentability ♻️

- **Model cards** offer a simple implementation to ensure **transparency and accountability** in the AI community. Learn [how to generate model cards in this notebook](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/bbe9c0a77499fa68de7c6d53bf5ef7e0b43a25e0/ML%20Accountability/Model%20Cards/model_card_generator.ipynb);
- Here you can find a notebook showing how to turn the [CO2 emission data generated by `CodeCarbon`](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/bbe9c0a77499fa68de7c6d53bf5ef7e0b43a25e0/ML%20Accountability/CO2%20Emission%20Report/emission_tracker.ipynb) into a **Markdown CO2 emission report**.

## Interpretability with CV 🔎🖼️

- If you want to [create your own CNN to explore](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/bbe9c0a77499fa68de7c6d53bf5ef7e0b43a25e0/ML%20Explainability/CV%20Interpreter/CNN_model_maker.ipynb) (using the below notebooks), you can use this simple `CNN_model_maker` for the **CIFAR-10 dataset** (_this may take a while if you don't have a GPU. If you don't have GPU, limit your training phase to < 50 epochs_);
- Learn to explore and **interpret the inner workings of a CNN** by using [feature visualization techniques](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/bbe9c0a77499fa68de7c6d53bf5ef7e0b43a25e0/ML%20Explainability/CV%20Interpreter/CNN_feature_visualization.ipynb);
- Learn to interpret the output of CNN models by using [saliency mapping techniques](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/bbe9c0a77499fa68de7c6d53bf5ef7e0b43a25e0/ML%20Explainability/CV%20Interpreter/CNN_attribution_maps.ipynb);
- Learn to [interpret the inner workings of a CNN](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/bbe9c0a77499fa68de7c6d53bf5ef7e0b43a25e0/ML%20Explainability/CV%20Interpreter/CNN_attribution_maps_with_LIME.ipynb) using the `LIME` (_Local Interpretable Model-Agnostic Explanations_) library.
- _Diffusion models_ are one of the current **paradigms in synthetic data generation** (especially when it comes to photorealism). Learn how to [interpret diffusion models](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/bbe9c0a77499fa68de7c6d53bf5ef7e0b43a25e0/ML%20Explainability/CV%20Interpreter/diffusion_interpreter.ipynb) with the `diffusers-interpret` library.

## Interpretability with NLP 🔎📚

- If you want to create your own NLP model to explore (using the below notebooks), you can use this simple `NLP_model_maker` (_available in [Portuguese](<https://github.com/Nkluge-correa/teeny-tiny_castle/blob/bbe9c0a77499fa68de7c6d53bf5ef7e0b43a25e0/ML%20Explainability/NLP%20Interpreter%20(pt)/model_maker_pt.ipynb>) and [English](<https://github.com/Nkluge-correa/teeny-tiny_castle/blob/bbe9c0a77499fa68de7c6d53bf5ef7e0b43a25e0/ML%20Explainability/NLP%20Interpreter%20(en)/model_maker_en.ipynb>)_) to **create your own language model**. We also provide a simple [UI interface](<https://github.com/Nkluge-correa/teeny-tiny_castle/blob/bbe9c0a77499fa68de7c6d53bf5ef7e0b43a25e0/ML%20Explainability/NLP%20Interpreter%20(en)/senti_dash_en.py>), created using `Dash` and `Flask` to allow you to easily interact with your model;
- Learn to **explore and interpret the outputs of a Language Model** using the `LIME` (_Local Interpretable Model-Agnostic Explanations_) library (_available in [Portuguese](<https://github.com/Nkluge-correa/teeny-tiny_castle/blob/bbe9c0a77499fa68de7c6d53bf5ef7e0b43a25e0/ML%20Explainability/NLP%20Interpreter%20(pt)/lime_for_NLP_pt.ipynb>) and [English](<https://github.com/Nkluge-correa/teeny-tiny_castle/blob/bbe9c0a77499fa68de7c6d53bf5ef7e0b43a25e0/ML%20Explainability/NLP%20Interpreter%20(en)/lime_for_NLP_en.ipynb>)_);
- Learn to interpret the output of Keras NLP models (like RNNs) using the **Integrated Gradients** method (_available in [Portuguese](<https://github.com/Nkluge-correa/teeny-tiny_castle/blob/bbe9c0a77499fa68de7c6d53bf5ef7e0b43a25e0/ML%20Explainability/NLP%20Interpreter%20(pt)/integrated_gradients_in%20_keras_nlp_pt.ipynb>) and [English](<https://github.com/Nkluge-correa/teeny-tiny_castle/blob/bbe9c0a77499fa68de7c6d53bf5ef7e0b43a25e0/ML%20Explainability/NLP%20Interpreter%20(en)/integrated_gradients_in%20_keras_nlp_en.ipynb>)_);
- Learn to explore text datasets using **Text Mining**. Here you will learn simple techniques to create visualization tools to interpret the distribution of patterns (_e.g., sentiment, word recurrence_) in a text corpus (_available in [Portuguese](<https://github.com/Nkluge-correa/teeny-tiny_castle/blob/bbe9c0a77499fa68de7c6d53bf5ef7e0b43a25e0/ML%20Explainability/NLP%20Interpreter%20(pt)/text_mining_pt.ipynb>) and [English](<https://github.com/Nkluge-correa/teeny-tiny_castle/blob/bbe9c0a77499fa68de7c6d53bf5ef7e0b43a25e0/ML%20Explainability/NLP%20Interpreter%20(en)/text_mining_en.ipynb>)_);
- **Ever wanted to build your own _Open-AI-style-NLP playground?_** Here you can find a working [dash.app that allows you to interact with LM and Report your findings](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/09375f6328dd45abfbea78c506a094291eeea6f6/ML%20Explainability/NLP%20Playgroung/playground.py). The UI was created using `Dash` and `Flask`. You can [download different models](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/09375f6328dd45abfbea78c506a094291eeea6f6/ML%20Explainability/NLP%20Playgroung/get_transformer.ipynb) from the `Hugging Face` library and feed this Playground with the model you so choose (_[DistillGPT2](https://huggingface.co/distilgpt2) is a great choice for people without GPU access_). This playground was specifically created for experiments involving prompt engineering and human feedback.

## Interpretability in Classification & Prediction with Tabular Data 🔎📊

- **Many of the models used in commercial applications of ML are basically classification and regression models** that work with `tabular data`. Here you can find examples of how to [interpret divergent classifications using `LIME` and `DALEX`](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/bbe9c0a77499fa68de7c6d53bf5ef7e0b43a25e0/ML%20Explainability/Tabular%20Interpreter/interpreter_for_tabular.ipynb), two libraries aimed and providing explainable AI (XAI);
- Investigate the **COMPAS Recidivism Racial Bias dataset**, and use **ML Fairness** and **XAI tools** to (1) [create an interpretable model](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/bbe9c0a77499fa68de7c6d53bf5ef7e0b43a25e0/ML%20Explainability/Tabular%20Interpreter/fairness_xai_COMPAS.ipynb), and (2) [understand the classifications of the algorithms created](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/bbe9c0a77499fa68de7c6d53bf5ef7e0b43a25e0/ML%20Explainability/Tabular%20Interpreter/fairness_xai_COMPAS.ipynb).

## Machine Learning Fairness ⚖️

- **Learn how to measure the "fairness" of an ML model** by using [fairness metrics in the _Credit Cart Dataset_](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/bbe9c0a77499fa68de7c6d53bf5ef7e0b43a25e0/ML%20Fairness/fair_metrics_Credit_card_approval.ipynb);
- Use the [AIF360](https://aif360.mybluemix.net/) library to **correct a dataset** (_- the Hogwarts case -_) using the [Disparate Impact Remover](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/bbe9c0a77499fa68de7c6d53bf5ef7e0b43a25e0/ML%20Fairness/disparate_impact_remove_Hogwarts.ipynb);
- Learn how to apply the [Ceteris paribus principle](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/bbe9c0a77499fa68de7c6d53bf5ef7e0b43a25e0/ML%20Fairness/ceteris_paribus_profiles.ipynb) to create a "**_What-if_**" model. Evaluate a classifier using **Counterfactual Fairness**.

## Cybersecurity 👾

- Learn a little bit about **password security** by [cracking bad passwords encrypted whit unsafe hashes](https://github.com/Nkluge-correa/password_cracking_dash) (**YOU SHOULD NOT PERFORM PASSWORD CRACKING AGAINST HASHES YOU HAVE NO BUSINESS OWNING**);
- Learn how to use **_Deep Learning_** to [classify/detect malware](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/bbe9c0a77499fa68de7c6d53bf5ef7e0b43a25e0/ML%20&%20Cybersecurity/ML%20Malware%20Analysis/Malware_detection.ipynb), among other techniques to explore and evaluate possibly malicious software (_[PE analysis, Reverse engineering, Automated Virus Scanning](https://github.com/Nkluge-correa/teeny-tiny_castle/tree/master/ML%20%26%20Cybersecurity/ML%20Malware%20Analysis)_).

## Adversarial Machine Learning 🐱‍💻

- Learn about the [pickle exploit](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/bbe9c0a77499fa68de7c6d53bf5ef7e0b43a25e0/ML%20Adversarial/the_pickle_exploit.ipynb), and how it can be used to **embed malicious** code into ML models (**YOU SHOULD NEVER CREATE MALICIOUS CODE TO BE USED AGAINST OTHERS**);
- **Evasion attacks** are perhaps the best-known type of attack in ML safety. Also known as "_adversarial examples_", [these attacks are carefully perturbed input samples that can completely throw off an ML model](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/bbe9c0a77499fa68de7c6d53bf5ef7e0b43a25e0/ML%20Adversarial/evasion_attacks.ipynb). Learn to craft adversarial examples using the `SecML` toolkit;
- Learn to [craft adversarial examples against CNNs through the Fast Sign Gradient Method](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/bbe9c0a77499fa68de7c6d53bf5ef7e0b43a25e0/ML%20Adversarial/evasion_attacks_FGSM.ipynb) using nothing but the utilities of the Keras/TensorFlow libraries;
- Language models are the cornerstone of many commercial applications. Learn how to [generate adversarial examples against language models](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/bbe9c0a77499fa68de7c6d53bf5ef7e0b43a25e0/ML%20Adversarial/adversarial_text_attack.ipynb) using the `textattack` library;
- **Model extraction attacks** pose a threat to intellectual property and privacy. Taking a proactive and adversarial approach to protecting ML systems, in this notebook, we [illustrate the inner workings of an extraction/cloning attack](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/bbe9c0a77499fa68de7c6d53bf5ef7e0b43a25e0/ML%20Adversarial/model_extraction_nlp.ipynb);
- Data poisoning attacks, like _label-flipping_ and _backdoor attacks_, can severely degrade model performance while giving an attacker the chance to introduce hidden functionalities and features into ML models. In [this notebook](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/bbe9c0a77499fa68de7c6d53bf5ef7e0b43a25e0/ML%20Adversarial/data_poisoning_attacks.ipynb), you can learn about these types of attacks.
- Adversarial training is one of the strategies that ML model developers can use to make their models more robust. In this notebook, you will [learn how to perform adversarial training](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/bbe9c0a77499fa68de7c6d53bf5ef7e0b43a25e0/ML%20Adversarial/adversarial_training_cv.ipynb) (using _FGSM_) with CNNs;
- And in this notebook, you will [learn how to perform adversarial training on language models](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/bbe9c0a77499fa68de7c6d53bf5ef7e0b43a25e0/ML%20Adversarial/adversarial_training_nlp.ipynb) using `textattack`.

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
  note = {Last updated 20 January 2023},
}

```

---

This repository was built as part of the RAIES ([Rede de Inteligência Artificial Ética e Segura](https://www.raies.org/)) initiative, a project supported by FAPERGS - ([Fundação de Amparo à Pesquisa do Estado do Rio Grande do Sul](https://fapergs.rs.gov.br/inicial)), Brazil.
