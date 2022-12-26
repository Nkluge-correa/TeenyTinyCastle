# **Machine Learning Glossary**

Return to the [castle](https://github.com/Nkluge-correa/teeny-tiny_castle).

If you want an online-AI-like version of this glossary, talk to [Ai.ra](http://aira-expert-en.airespucrs.org/).

Most of theses definitions were gather from the [Google Developers Machine Learning Glossary](https://developers.google.com/machine-learning/glossary). All hyperlinks will guide you to their page

---

## Accuracy

The fraction of [predictions](https://developers.google.com/machine-learning/glossary#prediction) that a [classification model](https://developers.google.com/machine-learning/glossary#classification_model) got right.

---

## Action

In [reinforcement learning](https://developers.google.com/machine-learning/glossary#reinforcement-learning-rl), an action is the mechanism by which the [agent](https://developers.google.com/machine-learning/glossary#agent) transitions between [states](https://developers.google.com/machine-learning/glossary#state) of the [environment](https://developers.google.com/machine-learning/glossary#environment). The agent chooses the action by using a [policy](https://developers.google.com/machine-learning/glossary#policy).

---

## Activation function

A function (e.g., [ReLU](https://developers.google.com/machine-learning/glossary#ReLU) or [sigmoid](https://developers.google.com/machine-learning/glossary#sigmoid_function)) that takes in the weighted sum of all of the inputs from the previous layer and then generates and passes an output value (typically nonlinear) to the next layer.

---

## Active learning

A [training](https://developers.google.com/machine-learning/glossary#training) approach in which the algorithm _chooses_ some of the data it learns from. Active learning is particularly valuable when [labeled examples](https://developers.google.com/machine-learning/glossary#labeled_example) are scarce or expensive to obtain. Instead of blindly seeking a diverse range of labeled examples, an active learning algorithm selectively seeks the particular range of examples it needs for learning.

---

## AdaGrad

A sophisticated gradient descent algorithm that rescales the gradients of each [parameter](https://developers.google.com/machine-learning/glossary#parameter), effectively giving each parameter an independent [learning rate](https://developers.google.com/machine-learning/glossary#learning_rate). For a full explanation, see [this paper](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf).

---

## Agent

In [reinforcement learning](https://developers.google.com/machine-learning/glossary#reinforcement-learning-rl), the entity that uses a [policy](https://developers.google.com/machine-learning/glossary#policy) to maximize the expected [return](https://developers.google.com/machine-learning/glossary#return) gained from transitioning between [states](https://developers.google.com/machine-learning/glossary#state) of the [environment](https://developers.google.com/machine-learning/glossary#environment).

---

## Anomaly detection

The process of identifying [outliers](https://developers.google.com/machine-learning/glossary#outliers). For example, if the mean for a certain [feature](https://developers.google.com/machine-learning/glossary#feature) is 100 with a standard deviation of 10, then anomaly detection should flag a value of 200 as suspicious.

---

## Artificial General Intelligence

A non-human mechanism that demonstrates a _broad range_ of problem solving, creativity, and adaptability. For example, a program demonstrating artificial general intelligence could translate text, compose symphonies, _and_ excel at games that have not yet been invented.

---

## Artificial Intelligence

A non-human program or [model](https://developers.google.com/machine-learning/glossary#model) that can solve sophisticated tasks. For example, a program or model that translates text or a program or model that identifies diseases from radiologic images both exhibit artificial intelligence.

Formally, [machine learning](https://developers.google.com/machine-learning/glossary#machine_learning) is a sub-field of artificial intelligence. However, in recent years, some organizations have begun using the terms _artificial intelligence_ and _machine learning_ interchangeably.

---

## Attention

Any of a wide range of [neural network](https://developers.google.com/machine-learning/glossary#neural_network) architecture mechanisms that aggregate information from a set of inputs in a data-dependent manner. A typical attention mechanism might consist of a weighted sum over a set of inputs, where the [weight](https://developers.google.com/machine-learning/glossary#weight) for each input is computed by another part of the neural network.

Refer also to [self-attention](https://developers.google.com/machine-learning/glossary#self-attention) and [multi-head self-attention](https://developers.google.com/machine-learning/glossary#multi-head-self-attention), which are the building blocks of [Transformers](https://developers.google.com/machine-learning/glossary#Transformer).

---

## Attribute

Synonym for [feature](https://developers.google.com/machine-learning/glossary#feature). In fairness, attributes often refer to characteristics pertaining to individuals.

---

## AUC (Area under the ROC Curve)

An evaluation metric that considers all possible [classification thresholds](https://developers.google.com/machine-learning/glossary#classification_threshold).

The Area Under the [ROC curve](https://developers.google.com/machine-learning/glossary#ROC) is the probability that a classifier will be more confident that a randomly chosen positive example is actually positive than that a randomly chosen negative example is positive.

---

## Automation bias

When a human decision maker favors recommendations made by an automated decision-making system over information made without automation, even when the automated decision-making system makes errors.

---

## Average Precision

A metric for summarizing the performance of a ranked sequence of results. Average precision is calculated by taking the average of the [precision](https://developers.google.com/machine-learning/glossary#precision) values for each relevant result (each result in the ranked list where the recall increases relative to the previous result).

---

## Backpropagation

The primary algorithm for performing [gradient descent](https://developers.google.com/machine-learning/glossary#gradient_descent) on [neural networks](https://developers.google.com/machine-learning/glossary#neural_network). First, the output values of each node are calculated (and cached) in a forward pass. Then, the [partial derivative](https://developers.google.com/machine-learning/glossary#partial_derivative) of the error with respect to each parameter is calculated in a backward pass through the graph.

---

## Baseline

A [model](https://developers.google.com/machine-learning/glossary#model) used as a reference point for comparing how well another model (typically, a more complex one) is performing. For example, a [logistic regression model](https://developers.google.com/machine-learning/glossary#logistic_regression) might serve as a good baseline for a [deep model](https://developers.google.com/machine-learning/glossary#deep_model).

For a particular problem, the baseline helps model developers quantify the minimal expected performance that a new model must achieve for the new model to be useful.

---

## Batch

The set of [examples](https://developers.google.com/machine-learning/glossary#example) used in one [iteration](https://developers.google.com/machine-learning/glossary#iteration) (that is, one [gradient](https://developers.google.com/machine-learning/glossary#gradient) update) of [model training](https://developers.google.com/machine-learning/glossary#model_training).

See also [batch size](https://developers.google.com/machine-learning/glossary#batch_size).

---

## Batch Size

The number of [examples](https://developers.google.com/machine-learning/glossary#example) in a [batch](https://developers.google.com/machine-learning/glossary#batch). For example, the batch size of [SGD](https://developers.google.com/machine-learning/glossary#SGD) is 1, while the batch size of a [mini-batch](https://developers.google.com/machine-learning/glossary#mini-batch) is usually between 10 and 1000. Batch size is usually fixed during [training](https://developers.google.com/machine-learning/glossary#training) and [inference](https://developers.google.com/machine-learning/glossary#inference); however, [TensorFlow](https://developers.google.com/machine-learning/glossary#TensorFlow) does permit dynamic batch sizes.

---

## Bayesian Neural Network

A probabilistic [neural network](https://developers.google.com/machine-learning/glossary#neural_network) that accounts for uncertainty in [weights](https://developers.google.com/machine-learning/glossary#weight) and outputs. A standard neural network regression model typically [predicts](https://developers.google.com/machine-learning/glossary#prediction) a scalar value; for example, a model predicts a house price of 853,000. By contrast, a Bayesian neural network predicts a distribution of values; for example, a model predicts a house price of 853,000 with a standard deviation of 67,200. A Bayesian neural network relies on [Bayes' Theorem](https://betterexplained.com/articles/an-intuitive-and-short-explanation-of-bayes-theorem/) to calculate uncertainties in weights and predictions. A Bayesian neural network can be useful when it is important to quantify uncertainty, such as in models related to pharmaceuticals. Bayesian neural networks can also help prevent [overfitting](https://developers.google.com/machine-learning/glossary#overfitting).

---

## Bayesian Optimization

A [probabilistic regression model](https://developers.google.com/machine-learning/glossary#probabilistic-regression-model) technique for optimizing computationally expensive [objective functions](https://developers.google.com/machine-learning/glossary#objective_function) by instead optimizing a surrogate that quantifies the uncertainty via a Bayesian learning technique. Since Bayesian optimization is itself very expensive, it is usually used to optimize expensive-to-evaluate tasks that have a small number of parameters, such as selecting [hyperparameters](https://developers.google.com/machine-learning/glossary#hyperparameter).

---

## Bellman Equation

In reinforcement learning, the following identity satisfied by the optimal [Q-function](https://developers.google.com/machine-learning/glossary#q-function). [Reinforcement learning](https://developers.google.com/machine-learning/glossary#reinforcement_learning) algorithms apply this identity to create [Q-learning](https://developers.google.com/machine-learning/glossary#q-learning) via the following update rule.

Beyond reinforcement learning, the Bellman equation has applications to dynamic programming. See the [Wikipedia entry for Bellman Equation](https://wikipedia.org/wiki/Bellman_equation).

---

## BERT (Bidirectional Encoder Representations from Transformers)

A model architecture for text [representation](https://developers.google.com/machine-learning/glossary#representation). A trained BERT model can act as part of a larger model for text classification or other ML tasks.

BERT has the following characteristics:

-Uses the [Transformer](https://developers.google.com/machine-learning/glossary#Transformer) architecture, and therefore relies on [self-attention](https://developers.google.com/machine-learning/glossary#self-attention). -Uses the [encoder](https://developers.google.com/machine-learning/glossary#encoder) part of the Transformer. The encoder's job is to produce good text representations, rather than to perform a specific task like classification. -Is [bidirectional](https://developers.google.com/machine-learning/glossary#bidirectional). -Uses [masking](https://developers.google.com/machine-learning/glossary#masked-language-model) for [unsupervised training](https://developers.google.com/machine-learning/glossary#unsupervised_machine_learning).

---

## Bias (ethics/fairness)

Stereotyping, prejudice or favoritism towards some things, people, or groups over others. These biases can affect collection and interpretation of data, the design of a system, and how users interact with a system.

Not to be confused with the [bias term](https://developers.google.com/machine-learning/glossary#bias) in machine learning models or [prediction bias](https://developers.google.com/machine-learning/glossary#prediction_bias).

---

## Bias (math)

An intercept or offset from an origin. Bias (also known as the **bias term**) is referred to as _b_ or _w0_ in machine learning models.

Not to be confused with [bias in ethics and fairness](https://developers.google.com/machine-learning/glossary#bias_ethics) or [prediction bias](https://developers.google.com/machine-learning/glossary#prediction_bias).

---

## Binary classification

A type of [classification](https://developers.google.com/machine-learning/glossary#classification_model) task that outputs one of two mutually exclusive [classes](https://developers.google.com/machine-learning/glossary#class). For example, a machine learning model that evaluates email messages and outputs either "spam" or "not spam" is a [binary classifier](https://developers.google.com/machine-learning/glossary#binary_classification).

Contrast with [multi-class classification](https://developers.google.com/machine-learning/glossary#multi-class-classification).

---

## Categorical Data

[Features](https://developers.google.com/machine-learning/glossary#feature) having a discrete set of possible values (e.g., eye color). Categorical features are sometimes called [discrete features](https://developers.google.com/machine-learning/glossary#discrete_feature). Contrast with [numerical data](https://developers.google.com/machine-learning/glossary#numerical_data).

---

## Class

One of a set of enumerated target values for a [label](https://developers.google.com/machine-learning/glossary#label). For example, in a [binary classification](https://developers.google.com/machine-learning/glossary#binary_classification) model that detects spam, the two classes are _spam_ and _not spam_. In a [multi-class classification](https://developers.google.com/machine-learning/glossary#multi-class) model that identifies dog breeds, the classes would be _poodle_, _beagle_, _pug_, and so on.

---

## Classification Model

A type of [model](https://developers.google.com/machine-learning/glossary#model) that distinguishes among two or more discrete classes. For example, a natural language processing classification model could determine whether an input sentence was in French, Spanish, or Italian.

---

## Continuous Feature

A floating-point [feature](https://developers.google.com/machine-learning/glossary#feature) with an infinite range of possible values. Contrast with [discrete feature](https://developers.google.com/machine-learning/glossary#discrete_feature).

---

## Convex function

A function in which the region above the graph of the function is a [convex set](https://developers.google.com/machine-learning/glossary#convex_set). The prototypical convex function is shaped something like the letter **U**. For example, the following are all convex functions:

![U-shaped curves, each with a single minimum point.](https://developers.google.com/static/machine-learning/glossary/images/convex_functions.png)

A **strictly convex function** has exactly one local minimum point, which is also the global minimum point. The classic U-shaped functions are strictly convex functions. However, some convex functions (for example, straight lines) are not U-shaped.

Many variations of [gradient descent](https://developers.google.com/machine-learning/glossary#gradient_descent) are guaranteed to find a point close to the minimum of a strictly convex function. Similarly, many variations of [stochastic gradient descent](https://developers.google.com/machine-learning/glossary#SGD) have a high probability (though, not a guarantee) of finding a point close to the minimum of a strictly convex function.

[Deep models](https://developers.google.com/machine-learning/glossary#deep_model) are never convex functions. Remarkably, algorithms designed for [convex optimization](https://developers.google.com/machine-learning/glossary#convex_optimization) tend to find reasonably good solutions on deep networks anyway, even though those solutions are not guaranteed to be a global minimum.

---

## Convex Optimization

The process of using mathematical techniques such as [gradient descent](https://developers.google.com/machine-learning/glossary#gradient_descent) to find the minimum of a [convex function](https://developers.google.com/machine-learning/glossary#convex_function). A great deal of research in machine learning has focused on formulating various problems as convex optimization problems and in solving those problems more efficiently. For complete details, see Boyd and Vandenberghe, [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf).

---

## Convolution

In mathematics, casually speaking, a mixture of two functions. In machine learning, a convolution mixes the convolutional filter and the input matrix in order to train [weights](https://developers.google.com/machine-learning/glossary#weight).

The term "convolution" in machine learning is often a shorthand way of referring to either [convolutional operation](https://developers.google.com/machine-learning/glossary#convolutional_operation) or [convolutional layer](https://developers.google.com/machine-learning/glossary#convolutional_layer).

Without convolutions, a machine learning algorithm would have to learn a separate weight for every cell in a large [tensor](https://developers.google.com/machine-learning/glossary#tensor). For example, a machine learning algorithm training on 2K x 2K images would be forced to find 4M separate weights. Thanks to convolutions, a machine learning algorithm only has to find weights for every cell in the [convolutional filter](https://developers.google.com/machine-learning/glossary#convolutional_filter), dramatically reducing the memory needed to train the model. When the convolutional filter is applied, it is simply replicated across cells such that each is multiplied by the filter.

---

## Convolutional Neural Network

A [neural network](https://developers.google.com/machine-learning/glossary#neural_network) in which at least one layer is a [convolutional layer](https://developers.google.com/machine-learning/glossary#convolutional_layer). A typical convolutional neural network consists of some combination of the following layers:

-[convolutional layers](https://developers.google.com/machine-learning/glossary#convolutional_layer) -[pooling layers](https://developers.google.com/machine-learning/glossary#pooling) -[dense layers](https://developers.google.com/machine-learning/glossary#dense_layer)

Convolutional neural networks have had great success in certain kinds of problems, such as image recognition.

---

## Cross-entropy

A generalization of [Log Loss](https://developers.google.com/machine-learning/glossary#Log_Loss) to [multi-class classification problems](https://developers.google.com/machine-learning/glossary#multi-class). Cross-entropy quantifies the difference between two probability distributions. See also [perplexity](https://developers.google.com/machine-learning/glossary#perplexity).

---

## Data set or Dataset

A collection of [examples](https://developers.google.com/machine-learning/glossary#example).

---

## Decision Forest

A model created from multiple [decision trees](https://developers.google.com/machine-learning/glossary#decision-tree). A decision forest makes a prediction by aggregating the predictions of its decision trees. Popular types of decision forests include [random forests](https://developers.google.com/machine-learning/glossary#random-forest) and [gradient boosted trees](https://developers.google.com/machine-learning/glossary#gbt).

---

## Deep Model

A type of [neural network](https://developers.google.com/machine-learning/glossary#neural_network) containing multiple [hidden layers](https://developers.google.com/machine-learning/glossary#hidden_layer).

---

## Demographic Parity

A [fairness metric](https://developers.google.com/machine-learning/glossary#fairness_metric) that is satisfied if the results of a model's classification are not dependent on a given [sensitive attribute](https://developers.google.com/machine-learning/glossary#sensitive_attribute).

---

## Dense Layer

Synonym for [fully connected layer](https://developers.google.com/machine-learning/glossary#fully_connected_layer).

---

## Depth

The number of [layers](https://developers.google.com/machine-learning/glossary#layer) (including any [embedding](https://developers.google.com/machine-learning/glossary#embeddings) layers) in a [neural network](https://developers.google.com/machine-learning/glossary#neural_network) that learn [weights](https://developers.google.com/machine-learning/glossary#weight). For example, a neural network with 5 [hidden layers](https://developers.google.com/machine-learning/glossary#hidden_layer) and 1 [output layer](https://developers.google.com/machine-learning/glossary#output-layer) has a depth of 6.

---

## Disparate Impact

Making decisions about people that impact different population subgroups disproportionately. This usually refers to situations where an algorithmic decision-making process harms or benefits some subgroups more than others.

Contrast with [disparate treatment](https://developers.google.com/machine-learning/glossary#disparate_treatment), which focuses on disparities that result when subgroup characteristics are explicit inputs to an algorithmic decision-making process.

---

## Disparate treatment

Factoring subjects' [sensitive attributes](https://developers.google.com/machine-learning/glossary#sensitive_attribute) into an algorithmic decision-making process such that different subgroups of people are treated differently.

Contrast with [disparate impact](https://developers.google.com/machine-learning/glossary#disparate_impact), which focuses on disparities in the societal impacts of algorithmic decisions on subgroups, irrespective of whether those subgroups are inputs to the model.

**Warning:** Because sensitive attributes are almost always correlated with other features the data may have, explicitly removing sensitive attribute information does not guarantee that subgroups will be treated equally. For example, removing sensitive demographic attributes from a training data set that still includes postal code as a feature may address disparate treatment of subgroups, but there still might be disparate impact upon these groups because postal code might serve as a [proxy](https://developers.google.com/machine-learning/glossary#proxy_sensitive_attributes) for other demographic information.

---

## Dropout Regularization

A form of [regularization](https://developers.google.com/machine-learning/glossary#regularization) useful in training [neural networks](https://developers.google.com/machine-learning/glossary#neural_network). Dropout regularization removes a random selection of a fixed number of the units in a network layer for a single gradient step. The more units dropped out, the stronger the regularization. This is analogous to training the network to emulate an exponentially large [ensemble](https://developers.google.com/machine-learning/glossary#ensemble) of smaller networks. For full details, see [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf).

---

## Dynamic Model

A [model](https://developers.google.com/machine-learning/glossary#model) that is trained online in a continuously updating fashion. That is, data is continuously entering the model.

---

## Empirical Risk Minimization (ERM)

Choosing the function that minimizes loss on the training set. Contrast with [structural risk minimization](https://developers.google.com/machine-learning/glossary#SRM).

---

## Environment

In reinforcement learning, the world that contains the [agent](https://developers.google.com/machine-learning/glossary#agent) and allows the agent to observe that world's [state](https://developers.google.com/machine-learning/glossary#state). For example, the represented world can be a game like chess, or a physical world like a maze. When the agent applies an [action](https://developers.google.com/machine-learning/glossary#action) to the environment, then the environment transitions between states.

---

## Episode

In reinforcement learning, each of the repeated attempts by the [agent](https://developers.google.com/machine-learning/glossary#agent) to learn an [environment](https://developers.google.com/machine-learning/glossary#environment).

---

## Epoch

A full training pass over the entire dataset such that each example has been seen once. Thus, an epoch represents `N`/[batch size](https://developers.google.com/machine-learning/glossary#batch_size) training [iterations](https://developers.google.com/machine-learning/glossary#iteration), where `N` is the total number of examples.

---

## Equality of Opportunity

A [fairness metric](https://developers.google.com/machine-learning/glossary#fairness_metric) that checks whether, for a preferred [label](https://developers.google.com/machine-learning/glossary#label) (one that confers an advantage or benefit to a person) and a given [attribute](https://developers.google.com/machine-learning/glossary#attribute), a classifier predicts that preferred label equally well for all values of that attribute. In other words, equality of opportunity measures whether the people who should qualify for an opportunity are equally likely to do so regardless of their group membership.

---

## Equalized Odds

A [fairness metric](https://developers.google.com/machine-learning/glossary#fairness_metric) that checks if, for any particular label and attribute, a classifier predicts that label equally well for all values of that attribute.

---

## Fairness Metric

A mathematical definition of “fairness” that is measurable. Many fairness metrics are mutually exclusive; see [incompatibility of fairness metrics](https://developers.google.com/machine-learning/glossary#incompatibility_of_fairness_metrics).

---

## False Negative (FN)

An example in which the model mistakenly predicted the [negative class](https://developers.google.com/machine-learning/glossary#negative_class). For example, the model inferred that a particular email message was not spam (the negative class), but that email message actually was spam.

---

## False Positive (FP)

An example in which the model mistakenly predicted the [positive class](https://developers.google.com/machine-learning/glossary#positive_class). For example, the model inferred that a particular email message was spam (the positive class), but that email message was actually not spam.

---

## Feature

An input variable used in making [predictions](https://developers.google.com/machine-learning/glossary#prediction).

---

## Feature Engineering

The process of determining which [features](https://developers.google.com/machine-learning/glossary#feature) might be useful in training a model, and then converting raw data from log files and other sources into said features. In TensorFlow, feature engineering often means converting raw log file entries to [tf.Example](https://developers.google.com/machine-learning/glossary#tf.Example) protocol buffers. See also [tf.Transform](https://github.com/tensorflow/transform).

Feature engineering is sometimes called **feature extraction**.

---

## Feedforward Neural Network (FFN)

A neural network without cyclic or recursive connections. For example, traditional [deep neural networks](https://developers.google.com/machine-learning/glossary#deep_neural_network) are feedforward neural networks. Contrast with [recurrent neural networks](https://developers.google.com/machine-learning/glossary#recurrent_neural_network), which are cyclic.

---

## Few-shot Learning

A machine learning approach, often used for object classification, designed to learn effective classifiers from only a small number of training examples. See also [one-shot learning](https://developers.google.com/machine-learning/glossary#one-shot_learning).

---

## Fine Tuning

Perform a secondary optimization to adjust the parameters of an already trained [model](https://developers.google.com/machine-learning/glossary#model) to fit a new problem. Fine tuning often refers to refitting the weights of a trained [unsupervised](https://developers.google.com/machine-learning/glossary#unsupervised_machine_learning) model to a [supervised](https://developers.google.com/machine-learning/glossary#supervised_machine_learning) model.

---

## Fully Connected Layer

A [hidden layer](https://developers.google.com/machine-learning/glossary#hidden_layer) in which each [node](https://developers.google.com/machine-learning/glossary#node) is connected to _every_ node in the subsequent hidden layer. A fully connected layer is also known as a [dense layer](https://developers.google.com/machine-learning/glossary#dense_layer).

---

## Generalization Curve

A [loss curve](https://developers.google.com/machine-learning/glossary#loss_curve) showing both the [training set](https://developers.google.com/machine-learning/glossary#training_set) and the [validation set](https://developers.google.com/machine-learning/glossary#validation_set). A generalization curve can help you detect possible [overfitting](https://developers.google.com/machine-learning/glossary#overfitting). For example, the following generalization curve suggests overfitting because loss for the validation set ultimately becomes significantly higher than for the training set.

---

## Generative Adversarial Network (GAN)

A system to create new data in which a [generator](https://developers.google.com/machine-learning/glossary#generator) creates data and a [discriminator](https://developers.google.com/machine-learning/glossary#discriminator) determines whether that created data is valid or invalid.

---

## GPT (Generative Pre-trained Transformer)

A family of [Transformer](https://developers.google.com/machine-learning/glossary#Transformer)-based [large language models](https://developers.google.com/machine-learning/glossary#large-language-model) developed by [OpenAI](https://openai.com/).

---

## Gradient

The vector of [partial derivatives](https://developers.google.com/machine-learning/glossary#partial_derivative) with respect to all of the independent variables. In machine learning, the gradient is the vector of partial derivatives of the model function.

---

## Gradient Descent

A technique to minimize [loss](https://developers.google.com/machine-learning/glossary#loss) by computing the gradients of loss with respect to the model's parameters, conditioned on training data. Informally, gradient descent iteratively adjusts parameters, gradually finding the best combination of [weights](https://developers.google.com/machine-learning/glossary#weight) and bias to minimize loss.

---

## Ground Truth

The correct answer. Reality. Since reality is often subjective, expert [raters](https://developers.google.com/machine-learning/glossary#rater) typically are the proxy for ground truth.

---

## Hidden Layer

A synthetic layer in a [neural network](https://developers.google.com/machine-learning/glossary#neural_network) between the [input layer](https://developers.google.com/machine-learning/glossary#input_layer) (that is, the features) and the [output layer](https://developers.google.com/machine-learning/glossary#output_layer) (the prediction). Hidden layers typically contain an [activation function](https://developers.google.com/machine-learning/glossary#activation_function) (such as [ReLU](https://developers.google.com/machine-learning/glossary#ReLU)) for training. A [deep neural network](https://developers.google.com/machine-learning/glossary#deep_neural_network) contains more than one hidden layer.

---

## Hyperparameter

The "knobs" that you tweak during successive runs of training a model. For example, [learning rate](https://developers.google.com/machine-learning/glossary#learning_rate) is a hyperparameter. Contrast with [parameter](https://developers.google.com/machine-learning/glossary#parameter).

---

## Hyperplane

A boundary that separates a space into two subspaces. For example, a line is a hyperplane in two dimensions and a plane is a hyperplane in three dimensions. More typically in machine learning, a hyperplane is the boundary separating a high-dimensional space. [Kernel Support Vector Machines](https://developers.google.com/machine-learning/glossary#KSVMs) use hyperplanes to separate positive classes from negative classes, often in a very high-dimensional space.

---

## Incompatibility of Fairness Metrics

The idea that some notions of fairness are mutually incompatible and cannot be satisfied simultaneously. As a result, there is no single universal [metric](https://developers.google.com/machine-learning/glossary#fairness_metric) for quantifying fairness that can be applied to all ML problems. While this may seem discouraging, incompatibility of fairness metrics doesn’t imply that fairness efforts are fruitless. Instead, it suggests that fairness must be defined contextually for a given ML problem, with the goal of preventing harms specific to its use cases. See ["On the (im)possibility of fairness"](https://arxiv.org/pdf/1609.07236.pdf) for a more detailed discussion of this topic.

---

## Independently and Identically Distributed (i.i.d)

Data drawn from a distribution that doesn't change, and where each value drawn doesn't depend on values that have been drawn previously. An i.i.d. is the [ideal gas](https://wikipedia.org/wiki/Ideal_gas) of machine learning—a useful mathematical construct but almost never exactly found in the real world. For example, the distribution of visitors to a web page may be i.i.d. over a brief window of time; that is, the distribution doesn't change during that brief window and one person's visit is generally independent of another's visit. However, if you expand that window of time, seasonal differences in the web page's visitors may appear.

---

## Inference

In machine learning, often refers to the process of making predictions by applying the trained model to [unlabeled examples](https://developers.google.com/machine-learning/glossary#unlabeled_example). In statistics, inference refers to the process of fitting the parameters of a distribution conditioned on some observed data. (See the [Wikipedia article on statistical inference](https://wikipedia.org/wiki/Statistical_inference).)

---

## Input Layer

The first layer (the one that receives the input data) in a [neural network](https://developers.google.com/machine-learning/glossary#neural_network).

---

## Interpretability

The ability to explain or to present an ML model's reasoning in understandable terms to a human.

---

## Keras

A popular Python machine learning API. [Keras](https://keras.io/) runs on several deep learning frameworks, including TensorFlow, where it is made available as [tf.keras](https://www.tensorflow.org/api_docs/python/tf/keras).

---

## L1 loss

[Loss](https://developers.google.com/machine-learning/glossary#loss) function based on the absolute value of the difference between the values that a model is predicting and the actual values of the [labels](https://developers.google.com/machine-learning/glossary#label). L1 loss is less sensitive to outliers than [L2 loss](https://developers.google.com/machine-learning/glossary#squared_loss).

---

## L1 regularization

A type of [regularization](https://developers.google.com/machine-learning/glossary#regularization) that penalizes weights in proportion to the sum of the absolute values of the weights. In models relying on [sparse features](https://developers.google.com/machine-learning/glossary#sparse_features), L1 regularization helps drive the weights of irrelevant or barely relevant features to exactly 0, which removes those features from the model. Contrast with [L2 regularization](https://developers.google.com/machine-learning/glossary#L2_regularization).

---

## L2 loss

See [squared loss](https://developers.google.com/machine-learning/glossary#squared_loss).

---

## L2 regularization

A type of [regularization](https://developers.google.com/machine-learning/glossary#regularization) that penalizes [weights](https://developers.google.com/machine-learning/glossary#weight) in proportion to the sum of the _squares_ of the weights. L2 regularization helps drive [outlier](https://developers.google.com/machine-learning/glossary#outliers) weights (those with high positive or low negative values) closer to 0 but not quite to 0. (Contrast with [L1 regularization](https://developers.google.com/machine-learning/glossary#L1_regularization).) L2 regularization always improves generalization in linear models.

---

## Label

In supervised learning, the "answer" or "result" portion of an [example](https://developers.google.com/machine-learning/glossary#example). Each example in a labeled dataset consists of one or more features and a label. For instance, in a housing dataset, the features might include the number of bedrooms, the number of bathrooms, and the age of the house, while the label might be the house's price. In a spam detection dataset, the features might include the subject line, the sender, and the email message itself, while the label would probably be either "spam" or "not spam."

---

## Language model

A [model](https://developers.google.com/machine-learning/glossary#model) that estimates the probability of a [token](https://developers.google.com/machine-learning/glossary#token) or sequence of tokens occurring in a longer sequence of tokens.

---

## Layer

A set of [neurons](https://developers.google.com/machine-learning/glossary#neuron) in a [neural network](https://developers.google.com/machine-learning/glossary#neural_network) that process a set of input features, or the output of those neurons Also, an abstraction in TensorFlow. Layers are Python functions that take [Tensors](https://developers.google.com/machine-learning/glossary#tensor) and configuration options as input and produce other tensors as output.

---

## Learning Rate

A scalar used to train a model via gradient descent. During each iteration, the [gradient descent](https://developers.google.com/machine-learning/glossary#gradient_descent) algorithm multiplies the learning rate by the gradient. The resulting product is called the **gradient step**.

Learning rate is a key [hyperparameter](https://developers.google.com/machine-learning/glossary#hyperparameter).

---

## Linear Model

A [model](https://developers.google.com/machine-learning/glossary#model) that assigns one [weight](https://developers.google.com/machine-learning/glossary#weight) per [feature](https://developers.google.com/machine-learning/glossary#feature) to make [predictions](https://developers.google.com/machine-learning/glossary#prediction). (Linear models also incorporate a [bias](https://developers.google.com/machine-learning/glossary#bias).) By contrast, the relationship of weights to features in [deep models](https://developers.google.com/machine-learning/glossary#deep_model) is not one-to-one. [Linear regression](https://developers.google.com/machine-learning/glossary#linear_regression) and [logistic regression](https://developers.google.com/machine-learning/glossary#logistic_regression) are two types of linear models. Linear models include not only models that use the linear equation but also a broader set of models that use the linear equation as part of the formula. For example, logistic regression post-processes the raw prediction (y′) to calculate the prediction.

---

## Linear Regression

Using the raw output (y′) of a [linear model](https://developers.google.com/machine-learning/glossary#linear_model) as the actual prediction in a [regression model](https://developers.google.com/machine-learning/glossary#regression_model). The goal of a regression problem is to make a real-valued prediction. For example, if the raw output (y′) of a linear model is 8.37, then the prediction is 8.37. Contrast linear regression with [logistic regression](https://developers.google.com/machine-learning/glossary#logistic_regression). Also, contrast regression with [classification](https://developers.google.com/machine-learning/glossary#classification_model).

---

## Logistic Regression

A [classification model](https://developers.google.com/machine-learning/glossary#classification_model) that uses a [sigmoid function](https://developers.google.com/machine-learning/glossary#sigmoid_function) to convert a [linear model's](https://developers.google.com/machine-learning/glossary#linear_model) raw prediction (y′) into a value between 0 and 1.

---

## Loss

A measure of how far a model's [predictions](https://developers.google.com/machine-learning/glossary#prediction) are from its [label](https://developers.google.com/machine-learning/glossary#label). Or, to phrase it more pessimistically, a measure of how bad the model is. To determine this value, a model must define a loss function. For example, linear regression models typically use [mean squared error](https://developers.google.com/machine-learning/glossary#MSE) for a loss function, while logistic regression models use [Log Loss](https://developers.google.com/machine-learning/glossary#Log_Loss).

---

## Loss Surface

A graph of weight(s) vs. loss. [Gradient descent](https://developers.google.com/machine-learning/glossary#gradient_descent) aims to find the weight(s) for which the loss surface is at a local minimum.

---

## Machine Learning

A program or system that builds (trains) a predictive model from input data. The system uses the learned model to make useful predictions from new (never-before-seen) data drawn from the same distribution as the one used to train the model. Machine learning also refers to the field of study concerned with these programs or systems.

---

## Mean Squared Error (MSE)

The average squared loss per example. MSE is calculated by dividing the [squared loss](https://developers.google.com/machine-learning/glossary#squared_loss) by the number of [examples](https://developers.google.com/machine-learning/glossary#example).

---

## Model

The representation of what a machine learning system has learned from the training data.

## Multi-class Classification

---

Classification problems that distinguish among more than two classes. For example, there are approximately 128 species of maple trees, so a model that categorized maple tree species would be multi-class. Conversely, a model that divided emails into only two categories (_spam_ and _not spam_) would be a [binary classification model](https://developers.google.com/machine-learning/glossary#binary_classification).

---

## Multi-class Logistic Regression

Using [logistic regression](https://developers.google.com/machine-learning/glossary#logistic_regression) in [multi-class classification](https://developers.google.com/machine-learning/glossary#multi-class) problems.

---

## NaN trap

When one number in your model becomes a [NaN](https://wikipedia.org/wiki/NaN) during training, which causes many or all other numbers in your model to eventually become a NaN. NaN is an abbreviation for "Not a Number."

---

## Neural Network

A model that, taking inspiration from the brain, is composed of layers (at least one of which is [hidden](https://developers.google.com/machine-learning/glossary#hidden_layer)) consisting of simple connected units or [neurons](https://developers.google.com/machine-learning/glossary#neuron) followed by nonlinearities.

---

## Neuron

A node in a [neural network](https://developers.google.com/machine-learning/glossary#neural_network), typically taking in multiple input values and generating one output value. The neuron calculates the output value by applying an [activation function](https://developers.google.com/machine-learning/glossary#activation_function) (nonlinear transformation) to a weighted sum of input values.

---

## Normalization

The process of converting an actual range of values into a standard range of values, typically -1 to +1 or 0 to 1. For example, suppose the natural range of a certain feature is 800 to 6,000. Through subtraction and division, you can normalize those values into the range -1 to +1.

---

## NumPy

An [open-source math library](http://www.numpy.org/) that provides efficient array operations in Python. [pandas](https://developers.google.com/machine-learning/glossary#pandas) is built on NumPy.

---

## Objective

A metric that your algorithm is trying to optimize.

---

## Objective function

The mathematical formula or metric that a model aims to optimize. For example, the objective function for [linear regression](https://developers.google.com/machine-learning/glossary#linear_regression) is usually [squared loss](https://developers.google.com/machine-learning/glossary#squared_loss). Therefore, when training a linear regression model, the goal is to minimize squared loss. In some cases, the goal is to maximize the objective function. For example, if the objective function is accuracy, the goal is to maximize accuracy.

---

## Outlier Detection

The process of identifying [outliers](https://developers.google.com/machine-learning/glossary#outliers) in a [training set](https://developers.google.com/machine-learning/glossary#training_set).

---

## Outliers

Values distant from most other values.

---

## Output layer

The "final" layer of a neural network. The layer containing the answer(s).

---

## Overfitting

Creating a model that matches the [training data](https://developers.google.com/machine-learning/glossary#training_set) so closely that the model fails to make correct predictions on new data.

---

## Oversampling

Reusing the [examples](https://developers.google.com/machine-learning/glossary#example) of a [minority class](https://developers.google.com/machine-learning/glossary#minority_class) in a [class-imbalanced dataset](https://developers.google.com/machine-learning/glossary#class_imbalanced_data_set) in order to create a more balanced [training set](https://developers.google.com/machine-learning/glossary#training_set).

---

## Pandas

A column-oriented data analysis API. Many machine learning frameworks, including TensorFlow, support pandas data structures as input. See the [pandas documentation](http://pandas.pydata.org/) for details.

---

## Parameter

A variable of a model that the machine learning system trains on its own. For example, [weights](https://developers.google.com/machine-learning/glossary#weight) are parameters whose values the machine learning system gradually learns through successive training iterations. Contrast with [hyperparameter](https://developers.google.com/machine-learning/glossary#hyperparameter).

---

## Partial Derivative

A derivative in which all but one of the variables is considered a constant. For example, the partial derivative of _f(x, y)_ with respect to _x_ is the derivative of _f_ considered as a function of _x_ alone (that is, keeping _y_ constant). The partial derivative of _f_ with respect to _x_ focuses only on how _x_ is changing and ignores all other variables in the equation.

---

## Perceptron

A system (either hardware or software) that takes in one or more input values, runs a function on the weighted sum of the inputs, and computes a single output value. In machine learning, the function is typically nonlinear, such as [ReLU](https://developers.google.com/machine-learning/glossary#ReLU), [sigmoid](https://developers.google.com/machine-learning/glossary#sigmoid_function), or tanh.

Perceptrons are the ([nodes](https://developers.google.com/machine-learning/glossary#node)) in [deep neural networks](https://developers.google.com/machine-learning/glossary#deep_model). That is, a deep neural network consists of multiple connected perceptrons, plus a [backpropagation](https://developers.google.com/machine-learning/glossary#backpropagation) algorithm to introduce feedback.

---

## Perplexity

One measure of how well a [model](https://developers.google.com/machine-learning/glossary#model) is accomplishing its task. For example, suppose your task is to read the first few letters of a word a user is typing on a smartphone keyboard, and to offer a list of possible completion words. Perplexity, P, for this task is approximately the number of guesses you need to offer in order for your list to contain the actual word the user is trying to type.

---

## Policy

In reinforcement learning, an [agent's](https://developers.google.com/machine-learning/glossary#agent) probabilistic mapping from [states](https://developers.google.com/machine-learning/glossary#state) to [actions](https://developers.google.com/machine-learning/glossary#action).

---

## Precision

A metric for [classification models](https://developers.google.com/machine-learning/glossary#classification_model). Precision identifies the frequency with which a model was correct when predicting the [positive class](https://developers.google.com/machine-learning/glossary#positive_class).

---

## Prediction

A model's output when provided with an input [example](https://developers.google.com/machine-learning/glossary#example).

---

## Pre-Trained Model

Models or model components (such as [embeddings](https://developers.google.com/machine-learning/glossary#embeddings)) that have been already been trained. Sometimes, you'll feed pre-trained embeddings into a [neural network](https://developers.google.com/machine-learning/glossary#neural_network). Other times, your model will train the embeddings itself rather than rely on the pre-trained embeddings.

---

## Proxy (sensitive attributes)

An attribute used as a stand-in for a [sensitive attribute](https://developers.google.com/machine-learning/glossary#sensitive_attribute). For example, an individual's postal code might be used as a proxy for their income, race, or ethnicity.

---

## Q-function

In reinforcement learning, the function that predicts the expected [return](https://developers.google.com/machine-learning/glossary#return) from taking an [action](https://developers.google.com/machine-learning/glossary#action) in a [state](https://developers.google.com/machine-learning/glossary#state) and then following a given [policy](https://developers.google.com/machine-learning/glossary#policy).

Q-function is also known as **state-action value function**.

---

## Q-learning

In reinforcement learning, an algorithm that allows an [agent](https://developers.google.com/machine-learning/glossary#agent) to learn the optimal [Q-function](https://developers.google.com/machine-learning/glossary#q-function) of a [Markov decision process](https://developers.google.com/machine-learning/glossary#markov_decision_process) by applying the [Bellman equation](https://developers.google.com/machine-learning/glossary#bellman_equation). The Markov decision process models an [environment](https://developers.google.com/machine-learning/glossary#environment).

---

## Recall

A metric for [classification models](https://developers.google.com/machine-learning/glossary#classification_model) that answers the following question: Out of all the possible positive labels, how many did the model correctly identify?

---

## Rectified Linear Unit (ReLU)

An [activation function](https://developers.google.com/machine-learning/glossary#activation_function) with the following rules: -If input is negative or zero, output is 0; -If input is positive, output is equal to input.

---

## Recurrent Neural Network

A [neural network](https://developers.google.com/machine-learning/glossary#neural_network) that is intentionally run multiple times, where parts of each run feed into the next run. Specifically, hidden layers from the previous run provide part of the input to the same hidden layer in the next run. Recurrent neural networks are particularly useful for evaluating sequences, so that the hidden layers can learn from previous runs of the neural network on earlier parts of the sequence.

---

## Reinforcement Learning (RL)

A family of algorithms that learn an optimal [policy](https://developers.google.com/machine-learning/glossary#policy), whose goal is to maximize [return](https://developers.google.com/machine-learning/glossary#return) when interacting with an [environment](https://developers.google.com/machine-learning/glossary#environment). For example, the ultimate reward of most games is victory. Reinforcement learning systems can become expert at playing complex games by evaluating sequences of previous game moves that ultimately led to wins and sequences that ultimately led to losses.

---

## Representation

The process of mapping data to useful [features](https://developers.google.com/machine-learning/glossary#feature).

---

## Reward

In reinforcement learning, the numerical result of taking an [action](https://developers.google.com/machine-learning/glossary#action) in a [state](https://developers.google.com/machine-learning/glossary#state), as defined by the [environment](https://developers.google.com/machine-learning/glossary#environment).

---

## Scikit-learn

A popular open-source machine learning platform. See [scikit-learn.org](http://scikit-learn.org/).

---

## Self-attention (also called self-attention layer)

A neural network layer that transforms a sequence of [embeddings](https://developers.google.com/machine-learning/glossary#embeddings) (for instance, [token](https://developers.google.com/machine-learning/glossary#token) embeddings) into another sequence of embeddings. Each embedding in the output sequence is constructed by integrating information from the elements of the input sequence through an [attention](https://developers.google.com/machine-learning/glossary#attention) mechanism.

The **self** part of **self-attention** refers to the sequence attending to itself rather than to some other context. Self-attention is one of the main building blocks for [Transformers](https://developers.google.com/machine-learning/glossary#Transformer) and uses dictionary lookup terminology, such as “query”, “key”, and “value”.

A self-attention layer starts with a sequence of input representations, one for each word. The input representation for a word can be a simple embedding. For each word in an input sequence, the network scores the relevance of the word to every element in the whole sequence of words. The relevance scores determine how much the word's final representation incorporates the representations of other words.

---

## Self-supervised learning

A family of techniques for converting an [unsupervised machine learning](https://developers.google.com/machine-learning/glossary#unsupervised_machine_learning) problem into a [supervised machine learning](https://developers.google.com/machine-learning/glossary#supervised_machine_learning) problem by creating surrogate [labels](https://developers.google.com/machine-learning/glossary#label) from [unlabeled examples](https://developers.google.com/machine-learning/glossary#unlabeled_example). Some [Transformer](https://developers.google.com/machine-learning/glossary#Transformer)-based models such as [BERT](https://developers.google.com/machine-learning/glossary#BERT) use self-supervised learning. Self-supervised training is a [semi-supervised learning](https://developers.google.com/machine-learning/glossary#semi-supervised_learning) approach.

---

## Semi-supervised learning

Training a model on data where some of the training examples have labels but others don't. One technique for semi-supervised learning is to infer labels for the unlabeled examples, and then to train on the inferred labels to create a new model. Semi-supervised learning can be useful if labels are expensive to obtain but unlabeled examples are plentiful.

[Self-training](https://developers.google.com/machine-learning/glossary#self-training) is one technique for semi-supervised learning.

---

## Sensitive Attribute

A human attribute that may be given special consideration for legal, ethical, social, or personal reasons.

---

## Sentiment Analysis

Using statistical or machine learning algorithms to determine a group's overall attitude—positive or negative—toward a service, product, organization, or topic. For example, using [natural language understanding](https://developers.google.com/machine-learning/glossary#natural_language_understanding), an algorithm could perform sentiment analysis on the textual feedback from a university course to determine the degree to which students generally liked or disliked the course.

---

## Softmax

A function that provides probabilities for each possible class in a [multi-class classification model](https://developers.google.com/machine-learning/glossary#multi-class). The probabilities add up to exactly 1.0. For example, softmax might determine that the probability of a particular image being a dog at 0.9, a cat at 0.08, and a horse at 0.02. (Also called **full softmax**.)

---

## Squared loss

The [loss](https://developers.google.com/machine-learning/glossary#loss) function used in [linear regression](https://developers.google.com/machine-learning/glossary#linear_regression). (Also known as **L2 Loss**.) This function calculates the squares of the difference between a model's predicted value for a labeled [example](https://developers.google.com/machine-learning/glossary#example) and the actual value of the [label](https://developers.google.com/machine-learning/glossary#label). Due to squaring, this loss function amplifies the influence of bad predictions. That is, squared loss reacts more strongly to outliers than [L1 loss](https://developers.google.com/machine-learning/glossary#L1_loss).

---

## Step size

Synonym for [learning rate](https://developers.google.com/machine-learning/glossary#learning_rate).

---

## Stochastic Gradient Descent (SGD)

A [gradient descent](https://developers.google.com/machine-learning/glossary#gradient_descent) algorithm in which the batch size is one. In other words, SGD relies on a single example chosen uniformly at random from a dataset to calculate an estimate of the gradient at each step.

---

## Supervised Machine Learning

Training a [model](https://developers.google.com/machine-learning/glossary#model) from input data and its corresponding [labels](https://developers.google.com/machine-learning/glossary#label). Supervised machine learning is analogous to a student learning a subject by studying a set of questions and their corresponding answers. After mastering the mapping between questions and answers, the student can then provide answers to new (never-before-seen) questions on the same topic. Compare with [unsupervised machine learning](https://developers.google.com/machine-learning/glossary#unsupervised_machine_learning).

---

## Synthetic Feature

A [feature](https://developers.google.com/machine-learning/glossary#feature) not present among the input features, but created from one or more of them.

---

## Tensor

The primary data structure in TensorFlow programs. Tensors are N-dimensional (where N could be very large) data structures, most commonly scalars, vectors, or matrices. The elements of a Tensor can hold integer, floating-point, or string values.

---

## TensorFlow

A large-scale, distributed, machine learning platform. The term also refers to the base API layer in the TensorFlow stack, which supports general computation on dataflow graphs.

Although TensorFlow is primarily used for machine learning, you may also use TensorFlow for non-ML tasks that require numerical computation using dataflow graphs.

---

## Transformer

A [neural network](https://developers.google.com/machine-learning/glossary#neural_network) architecture developed at Google that relies on [self-attention](https://developers.google.com/machine-learning/glossary#self-attention) mechanisms to transform a sequence of input [embeddings](https://developers.google.com/machine-learning/glossary#embeddings) into a sequence of output embeddings without relying on [convolutions](https://developers.google.com/machine-learning/glossary#convolution) or [recurrent neural networks](https://developers.google.com/machine-learning/glossary#recurrent_neural_network). A Transformer can be viewed as a stack of self-attention layers.

A Transformer can include any of the following an [encoder](https://developers.google.com/machine-learning/glossary#encoder), a [decoder](https://developers.google.com/machine-learning/glossary#decoder), or both an encoder and decoder. An **encoder** transforms a sequence of embeddings into a new sequence of the same length. An encoder includes N identical layers, each of which contains two sub-layers. These two sub-layers are applied at each position of the input embedding sequence, transforming each element of the sequence into a new embedding. The first encoder sub-layer aggregates information from across the input sequence. The second encoder sub-layer transforms the aggregated information into an output embedding. A **decoder** transforms a sequence of input embeddings into a sequence of output embeddings, possibly with a different length. A decoder also includes N identical layers with three sub-layers, two of which are similar to the encoder sub-layers. The third decoder sub-layer takes the output of the encoder and applies the [self-attention](https://developers.google.com/machine-learning/glossary#self-attention) mechanism to gather information from it.

---

## True Positive (TP)

An example in which the model _correctly_ predicted the [positive class](https://developers.google.com/machine-learning/glossary#positive_class). For example, the model inferred that a particular email message was spam, and that email message really was spam.

---

## Unsupervised Machine Learning

Training a [model](https://developers.google.com/machine-learning/glossary#model) to find patterns in a dataset, typically an unlabeled dataset. The most common use of unsupervised machine learning is to cluster data into groups of similar examples. For example, an unsupervised machine learning algorithm can cluster songs together based on various properties of the music. The resulting clusters can become an input to other machine learning algorithms (for example, to a music recommendation service). Clustering can be helpful in domains where true labels are hard to obtain. For example, in domains such as anti-abuse and fraud, clusters can help humans better understand the data.

---

## Weight

A coefficient for a [feature](https://developers.google.com/machine-learning/glossary#feature) in a linear model, or an edge in a deep network. The goal of training a linear model is to determine the ideal weight for each feature. If a weight is 0, then its corresponding feature does not contribute to the model.

---

Return to the [castle](https://github.com/Nkluge-correa/teeny-tiny_castle).
