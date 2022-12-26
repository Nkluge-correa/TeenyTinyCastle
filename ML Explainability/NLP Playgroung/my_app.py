from dash.dependencies import Input, Output, State
from dash import dcc, html, Output, Input, State
import dash_bootstrap_components as dbc
from transformers import AutoTokenizer
from transformers import pipeline
import transformers
import zipfile
import torch
import dash
import os
import io
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

name = 'distilgpt2'  # gpt2-large, distilgpt2, gpt2-medium

unzipped_model = zipfile.ZipFile(f'{name}.zip', 'r')
model_file = unzipped_model.read(f'{name}.pt')
stream = io.BytesIO(model_file)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(stream)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(f'{name}_tokenizer')
generator = pipeline('text-generation', model=model, tokenizer=tokenizer,
                     device=0 if torch.cuda.is_available() else -1)


def generate_text(string, length_choice, temp, respose_num, top, rep_penalty):
    text = string
    response = generator(text, pad_token_id=tokenizer.eos_token_id,
                         max_length=length_choice,
                         temperature=temp,
                         num_return_sequences=respose_num,
                         top_k=top,
                         repetition_penalty=rep_penalty)
    output = []
    output_2 = []
    for i in range(0, len(response)):
        x = response[i]['generated_text']
        output_2.append(x)
        output_2.append('\n')
    for i in range(0, len(response)):
        x = response[i]['generated_text']
        output.append(f'{i+1}# Model Response:\n\n')
        output.append(f'{x}')
        output.append(dcc.Markdown('___'))

    return output, output_2


modal = html.Div(
    [
        dbc.Button('Report', id='report-button', n_clicks=0,
                   outline=True, color='warning', disabled=True),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle(
                    dcc.Markdown('### **Report 📝**'), style={})),
                dbc.ModalBody([dcc.Markdown('#### Your prompt was:'), html.Br(),
                               dcc.Markdown('', id='prompt', style={
                                            'font-size': 25}),
                               dcc.Markdown('___'),
                               dcc.Markdown(
                                   '#### The model response(s) was:'), html.Br(),
                               dcc.Markdown('', id='response',
                                            style={'font-size': 25}),
                               dcc.Markdown('___'),
                               dcc.Markdown(
                                   "#### Evaluate the model's response(s) in the form below"),
                               dcc.Markdown('___'),
                               html.Iframe(src="https://forms.gle/sdAqtx3QHGf6YAVE9", style={
                                           "width": "100%", 'height': 1360, 'overflowY': 'scroll'})
                               ], style={
                    'text-align': 'justify',
                    'font-size': 20,
                    'text-justify': 'inter-word'}),
                dbc.ModalFooter(
                    html.Div([dbc.Button('Close', id='close-body-scroll', className='ms-auto',
                             n_clicks=0, color='warning', outline=False)], style={'display': 'inline-block'})
                ),
            ],
            id='modal-body-scroll',
            scrollable=True,
            size='lg',
            is_open=False,
        ),
    ], style={
        'text-align': 'right',
    },

)

modal_info = html.Div(
    [
        dbc.Button('Information', id='info-button',
                   n_clicks=0, outline=True, color='warning'),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle(dcc.Markdown(
                    '### **Language Model Alignment 🧑‍🏫**'), style={})),
                dbc.ModalBody([dcc.Markdown('''###### Since the transformer architecture was proposed by **Vaswani and collaborators** (Google) in their seminal paper "**[Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)**," deep learning applied to NLP tasks has entered a new era: *the era of **large** language models.*''', style={'font-size': 24}), html.Br(),
                               dcc.Markdown(
                                   '''###### Models such as **BERT** (110M), **GPT** (175B), and the mysterious **Wu Dao 2.0** (1.75T) are examples of large language models capable of solving many kinds of tasks without the need for fine-tuning.'''), html.Br(),
                               dcc.Markdown(
                                   '''###### However, such models can generate *false, toxic, or simply useless content for their controller.* Following in the line of research from similar works (Kenton et al., **[2021](https://arxiv.org/pdf/2103.14659.pdf)**; Ziegler et al., **[2022](https://arxiv.org/pdf/2205.01663.pdf)**; Ouyang et al., **[2022](https://arxiv.org/pdf/2203.02155.pdf)**; Romal et al., **[2022](https://arxiv.org/abs/2201.08239)**), this research seeks to evaluate and improve the **alignment of language models**, i.e., *how well the responses of such models are aligned with the instructions of a human controller.*'''), html.Br(),
                               dcc.Markdown(
                                   '''###### In the quest to create models that are **better [aligned](https://arxiv.org/abs/1906.01820) with their controllers**, we hope to be able to create **safer** and more **efficient** models. In this playground, you can submit instructions to a GPT model in the form of *"prompts,"* and then evaluate the model's performance.'''), html.Br(),
                               dcc.Markdown(
                                   '''###### This is what we call "**Prompt engineering**," i. e., *when the description of the task is embedded in the input given to the model*. For example:'''),
                               dcc.Markdown('___'),
                               dcc.Markdown("'''"),
                               dcc.Markdown(
                                   '''###### **Recipe for chocolate cake that does not use eggs and milk:**'''), html.Br(),
                               dcc.Markdown('''###### ***1 cup of all-purpose flour, 1/2 cup of cocoa powder, 1/2 teaspoon of baking soda, 1/2 teaspoon of baking powder, 1/4 teaspoon of salt, 1/2 cup of sugar, 1/4 cup of vegetable oil, 1/4 cup of water, 1 teaspoon of vanilla extract.***'''), html.Br(),
                               dcc.Markdown(
                                   '''###### **Recipe for a strawberry cake that does not use eggs and milk:**'''),
                               dcc.Markdown("'''"),
                               dcc.Markdown('___'),
                               dcc.Markdown(
                                   '''###### An aligned language model would produce a cake recipe (even if not very appetizing) **without using eggs and milk.**'''), html.Br(),
                               dcc.Markdown('''###### In this research, we look for examples of **misalignment** and **corrections**, i.e.,* if the model produces a response that is misaligned with its instruction, we want the controller to generate an aligned response.* The evaluations and responses generated will be used to **fine-tune a future model**, where we expect it to be more aligned with human intentions and instructions.'''),  html.Br(),
                               dcc.Markdown(
                                   '''###### For more information about this study, or if you would like to join our research team, please **[contact us](https://en.airespucrs.org/contato)**.'''),
                               dcc.Markdown('___'),
                               dcc.Markdown(
                                   '### **Sampling Parameters ⚙️**'), html.Br(),
                               dcc.Markdown('''###### Language models usually generate text through **greedy search**, i.e., *selecting the highest probability word at each step.* However, not always the *"best"* sentence is generated by the consecutive iteration of a greedy policy (sometimes the algorithm may end up **stuck in a loop of repeating sentences**). That's where sampling parameters come to aid us (the dials and knobs on the left portion of this page).'''), html.Br(),
                               dcc.Markdown('''###### With sampling, **every token with a non-zero probability has a chance of being selected**, and the different sampling parameters tweak the output generated by a model, e.g., by **increasing** the **temperature parameter**, which increases the **entropy** of the resulting **Softmax output**, we can get a more *diverse output.*'''), html.Br(),
                               dcc.Markdown(
                                   '''###### Let's have a quick and holist review of what every parameter controls:'''), html.Br(),
                               dcc.Markdown(
                                   '''###### ✔️ Response Length: *the maximum length of the response being generated by the model (the longer the response length, the longer the inference time).*'''),
                               dcc.Markdown(
                                   '''###### ✔️ Nº of Responses: *number of responses generated by the model.*'''),
                               dcc.Markdown('''###### ✔️ Temperature: *a hyperparameter that is applied to logits to affect the final probabilities from the softmax. A higher temperature will make the model output seem more "creative" and less deterministic (helps avoid repeating sentences).*'''),
                               dcc.Markdown('''###### ✔️ Top-K: *a hyperparameter that is applied to ensure that the less probable words (the tail of the distribution) should not have any chance at all. Only top K (10 = 10%) probable tokens should be considered, which helps a model "stay on track" and not deviate from the topic.*'''),
                               dcc.Markdown(
                                   '''###### ✔️ Repetition penalty: *as the name suggests, this hyperparameter is meant to avoid sentences that repeat themselves without adding context to the output.*'''), html.Br(),
                               dcc.Markdown(
                                   '''###### You can teak these parameters however you like to generate responses from the model. Have fun!'''),
                               dcc.Markdown('___'),
                               dcc.Markdown('### **Model Card**'), html.Br(),
                               dcc.Markdown(
                                   '#### **DistilGPT2 🤖**'), html.Br(),
                               dcc.Markdown('''###### The model used in this study is **[DistilGPT2](https://huggingface.co/distilgpt2)** (short for Distilled-GPT2), an English-language model pre-trained with the supervision of the smallest version of **Generative Pre-trained Transformer 2** (GPT-2). The GPT-2 was introduced in **[this paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)** and first released at **[this page](https://openai.com/blog/better-language-models/)**. DistilGPT2 has **82 million parameters**, and was developed using **[knowledge distillation](https://huggingface.co/distilgpt2#knowledge-distillation)**, being a faster, lighter version of GPT-2. DistilGPT2 was trained using **[OpenWebTextCorpus](https://skylion007.github.io/OpenWebTextCorpus/)**, an open-source reproduction of OpenAI’s WebText dataset, which was used to train GPT-2.'''), html.Br(),
                               dcc.Markdown(
                                   '#### **Model Details 📊**'), html.Br(),
                               dcc.Markdown(
                                   '###### ✔️ Developed by: *[Hugging Face](https://huggingface.co/);*'),
                               dcc.Markdown(
                                   '###### ✔️ Model type: *Transformer-based Language Model;*'),
                               dcc.Markdown('###### ✔️ Language: *English;*'),
                               dcc.Markdown(
                                   '###### ✔️ License: *Apache 2.0;*'),
                               dcc.Markdown(
                                   '''###### ✔️ Resources for more information: *See this **[repository](https://github.com/huggingface/transformers/tree/main/examples/research_projects/distillation)**, Sanh et al. (**[2019](https://arxiv.org/abs/1910.01108)**) publication, and this page for more about **[GPT-2](https://openai.com/blog/better-language-models/)**.*'''), html.Br(),
                               dcc.Markdown(
                                   '### **Limitations and Risks ☣️**'), html.Br(),
                               dcc.Markdown(
                                   '''###### **CONTENT WARNING:** *this section contains content that is disturbing, offensive, and can propagate historical and current stereotypes.* DistilGPT2 suffers from persistent bias issues, as highlighted in the demonstrative examples below. These issues are an inheritance of GPT-2.  Issues related to GPT-2 are documented in this **[publication](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)** and this **[model card](https://github.com/openai/gpt-2/blob/master/model_card.md)**.  '''), html.Br(),
                               dcc.Markdown('''
                                ```
                                from transformers import pipeline, set_seed
                                generator = pipeline('text-generation', model='distilgpt2')
                                set_seed(48)
                                generator("The White man worked as a", max_length=20, num_return_sequences=3)
                                [{'generated_text': "The White man worked as a salesman at a McDonald's restaurant called Kia at the time of the"},
                                {'generated_text': 'The White man worked as a contractor in the Army in the late 1990s. He became a "'},
                                {'generated_text': 'The White man worked as a police spokesman to the US Navy in the 1930s.'}]
                                

                                generator("The Black man worked as a", max_length=20, num_return_sequences=3)
                                [{'generated_text': 'The Black man worked as a shop assistant for an hour at Wal-Mart at Wal-Mart in'},
                                {'generated_text': 'The Black man worked as a waiter in the hotel when he was assaulted when he got out of a'},
                                {'generated_text': 'The Black man worked as a police spokesman four months ago...'}]
                                ```
                                '''), html.Br(),
                               dcc.Markdown(
                                   '### **Potential Uses & Out-of-scope Uses 🚫**'), html.Br(),
                               dcc.Markdown(
                                   '''###### GPT-2 was developed to be used by researchers to better understand large-scale generative language models, with possible secondary use cases including: *writing assistance*, *creative writing* and *art*, and *entertainment*. OpenAI states in the GPT-2 **[model card](https://github.com/openai/gpt-2/blob/master/model_card.md)**: *“Because large-scale language models like GPT-2 do not distinguish fact from fiction, we don’t support use-cases that require the generated text to be true. Additionally, language models like GPT-2 reflect the biases inherent to the systems they were trained on, so we do not recommend that they be deployed into systems that interact with humans unless the deployers first carry out a study of biases relevant to the intended use-case.”*'''), html.Br(),
                               dcc.Markdown(
                                   '### **Environmental Impact 🌱**'), html.Br(),
                               dcc.Markdown(
                                   '''###### **Carbon emissions were estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute):**'''), html.Br(),
                               dcc.Markdown(
                                   '###### 🎯 Hardware Type: *8 16GB V100;*'),
                               dcc.Markdown(
                                   '###### 🎯 Hours used: *168 (1 week);*'),
                               dcc.Markdown(
                                   '###### 🎯 Cloud Provider: *Azure;*'),
                               dcc.Markdown(
                                   '###### 🎯 Compute Region: *assumed East US for calculations;*'),
                               dcc.Markdown(
                                   '###### 🎯 Carbon Emitted: *149.2 kg eq. CO2;*'), html.Br(),
                               ], style={
                    'text-align': 'justify',
                    'font-size': 20,
                    'text-justify': 'inter-word'}),
                dbc.ModalFooter(
                    html.Div([dbc.Button('Close', id='close-body-scroll-2', className='ms-auto',
                             n_clicks=0, color='warning', outline=False)], style={'display': 'inline-block'})
                ),
            ],
            id='modal-body-scroll-2',
            scrollable=True,
            fullscreen=True,
            is_open=False,
        ),
    ], style={
        'margin-right': '5px',
        'display': 'inline-block',
    },

)

app = dash.Dash(__name__,
                meta_tags=[
                    {'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0, maximum-scale=1.2, minimum-scale=0.5,'}],
                external_stylesheets=[dbc.themes.CYBORG])

server = app.server
app.title = 'NLP Playground 📚'

app.layout = dbc.Container(
    fluid=True,
    children=[
        html.H1('NLP Playground  📙 🤖 📚', style={
            'font-style': 'bold',
            'margin-top': '15px',
            'margin-left': '15px',
            'display': 'inline-block'}),
        html.Hr(),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    html.Br(),
                    dcc.Markdown('**Response Length**'),
                    dcc.Slider(100, 1000, 100,
                               value=100,
                               id='my-slider'),
                    html.Br(),
                    dcc.Markdown('**Nº of Responses**'),
                    dcc.Slider(1, 5, 1,
                               value=1,
                               id='my-slider_2'),
                    html.Br(),
                    dcc.Markdown('**Temperature**'),
                    dcc.Slider(0.1, 2.1, 0.1,
                               marks=None,
                               value=1.5,
                               tooltip={'placement': 'bottom',
                                        'always_visible': True},
                               id='my-slider_3'),
                    html.Br(),
                    dcc.Markdown('**Top-K**'),
                    dcc.Slider(10, 50, 10,
                               value=20,
                               id='my-slider_4'),
                    html.Br(),
                    dcc.Markdown('**Repetition penalty**'),
                    dcc.Slider(1.1, 2.1, 0.1,
                               marks=None,
                               value=1.5,
                               tooltip={'placement': 'bottom',
                                        'always_visible': True},
                               id='my-slider_5'),
                    html.Br(),
                ], color='warning', outline=True, style={'margin-left': '15px'})
            ], md=4),
            dbc.Col([
                dcc.Textarea(
                    id='textarea-state',
                    value='',
                    style={'width': '100%', 'height': 200},
                ),
                html.Div([modal_info, dbc.Button('Submit', id='submit-button', n_clicks=0,
                         outline=False, color='warning')], style={'text-align': 'right'}),
                dcc.Loading(id='loading_0', type='circle', children=[html.Div(id='textarea-output-state', style={'whiteSpace': 'pre-line',
                                                                                                                 'height': '200px',
                                                                                                                 'overflowY': 'scroll',
                                                                                                                 'margin-top': '5px',
                                                                                                                 'margin-bottom': '5px'})]),
                modal,
            ], md=8),
        ]),
        html.Hr(),

    ],
)


@app.callback(

    [Output('textarea-output-state', 'children'),
     Output('report-button', 'disabled'),
     Output('prompt', 'children'),
     Output('response', 'children')],

    Input('submit-button', 'n_clicks_timestamp'),

    [State('textarea-state', 'value'),
     State('my-slider', 'value'),
     State('my-slider_2', 'value'),
     State('my-slider_3', 'value'),
     State('my-slider_4', 'value'),
     State('my-slider_5', 'value'), ]
)
def update_output(click, value_txt, value_slider_1, value_slider_2, value_slider_3, value_slider_4, value_slider_5):
    if click is not None:
        x, y = generate_text(value_txt, value_slider_1, value_slider_3,
                             value_slider_2, value_slider_4, value_slider_5)
        return x, False, value_txt, y
    else:
        return '', True, '', ''


@app.callback(
    Output('modal-body-scroll', 'is_open'),
    [
        Input('report-button', 'n_clicks'),
        Input('close-body-scroll', 'n_clicks'),
    ],
    [State('modal-body-scroll', 'is_open')],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output('modal-body-scroll-2', 'is_open'),
    [
        Input('info-button', 'n_clicks'),
        Input('close-body-scroll-2', 'n_clicks'),
    ],
    [State('modal-body-scroll-2', 'is_open')],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


if __name__ == '__main__':
    app.run_server(debug=False)
