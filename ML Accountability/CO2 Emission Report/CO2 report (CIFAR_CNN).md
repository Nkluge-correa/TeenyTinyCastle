# $CO_2$ Emission Report

Generated at: _25/12/2022_

## CARBON FOOTPRINT

A carbon footprint is the total greenhouse gas (GHG) emissions caused by an individual, event, organization, service, place or product, expressed as carbon dioxide equivalent ($CO_2e$). Greenhouse gases, including the carbon-containing gases carbon dioxide and methane , can be emitted through the burning of fossil fuels , land clearance, and the production and consumption of food, manufactured goods, materials, wood, roads, buildings, transportation, and other services.

Modern AI models can consume a massive amount of energy during their training and fine-tuning phase, and these energy requirements are growing at a breathtaking rate. Researchers from the University of Massachusetts [[1](references)], Amherst, conducted a life cycle analysis for training several typical big AI models in a recent publication. They discovered that the procedure may produce almost $626,000$ pounds of $CO_2$ equivalent.

## $CO_2$ Emission Report with CodeCarbon

A $CO_2$ Emission Report is a simple transparency tool to help developers make public (and thus become accountable) the $CO_2$ production of an ML model.

This report is made possible by CodeCarbon [[2](references)] [[3](references)] [[4](references)] a lightweight software package that seamlessly integrates into your Python codebase. It estimates the amount of carbon dioxide ($CO_2$) produced by the cloud or personal computing resources used to execute the code.

## HOW TO USE CODECARBON

One can use the Code Carbon library by simply installing its dependencies with a `pip install codecarbon`, a using its tracker function to register the energy consumption of some costly computation.

```python

from codecarbon import EmissionsTracker

tracker = EmissionsTracker()
tracker.start()
expensive_computing_function_here()
tracker.stop()

```

## MODEL DETAILS

- Model developed by Nicholas Kluge (nicholas@airespucrs.org), researcher at the Pontifical Catholic University of Rio Grande do Sul (PUCRS), in October 2022;
- This model is a Convolutional Neural Network trained in multi-label classification;
- This model was trained solely for academic motivations, in order to explore interpretability techniques and adversarial example generation in ML;
- The dataset used to train this model was the CIFAR-10 dataset (Canadian Institute For Advanced Research);
- MIT License.

## $CO_2$ Emission Results

|**Duration (Seconds)**|**Emission (KgCO2)**|**Emission Rate (KtCO2/Year)**|**CPU Power (Watts)**|
|--------------------------------|-------------------------------------|------------------------------------|--------------------------------|
| 1192.956553|0.001056|0.000885|14.0|
|**GPU Power (Watts)**|**RAMPower (Watts)**|**Power Consumption (CPU - kWh)**|**Power Consumption (GPU - kWh)**|
|0.0| 11.905804|0.004639|0.0|
|**Power Consumption (RAM - kWh)**|**Total Consumption (kWh)**|**Country**| **ISO**|
|0.003942|0.008581|Brazil|BRA|
|**Region**| **Cloud Provider**| **Provider's Region**|**OS**|
|rio grande do sul| nan| nan|Windows-10-10.0.19041-SP0|
|**Python Version**| **No. of Processors**|**Provider's CPU Model**| **No. of GPUs**|
|3.8.0|8|11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz|1|
|**GPU Model**|**RAM Memory Size (GB)**| **Tracking Mode**|**Cloud-Processed**|
|1 x NVIDIA GeForce MX450| 31.748809814453125|machine| N|

## REFERENCES

[1] Karen Hao. Training a single ai model can emit as much carbon as five cars in their lifetimes. _MIT technology Review_, 2019.

[2] Alexandre Lacoste, Alexandra Luccioni, Victor Schmidt, and Thomas Dandres. Quantifying the carbon emissions of machine learning. _Workshop on Tackling Climate Change with Machine Learning at NeurIPS 2019_, 2019.

[3] Kadan Lottick, Silvia Susai, Sorelle A. Friedler, and Jonathan P. Wilson. Energy usage reports: Environmental awareness as part of algorithmicaccountability. _Workshop on Tackling Climate Change with Machine Learning at NeurIPS 2019_, 2019.

[4] Victor Schmidt, Kamal Goyal, Aditya Joshi, Boris Feld, Liam Conell, Nikolas Laskaris, Doug Blank, Jonathan Wilson, Sorelle Friedler, and Sasha Luccioni. CodeCarbon: _Estimate and Track Carbon Emissions from Machine Learning Computing_, 2021.
