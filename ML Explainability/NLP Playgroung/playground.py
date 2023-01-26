from dash import dcc, html, Output, Input, State
from transformers import AutoTokenizer, pipeline
import dash_bootstrap_components as dbc
import torch
import dash

model = torch.load('models\Distilgpt2.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

tokenizer = AutoTokenizer.from_pretrained('models\Distilgpt2_tokenizer')
generator = pipeline('text-generation', model=model, tokenizer=tokenizer,
                     device=0 if torch.cuda.is_available() else -1)


def generate_text(string, length_choice, temp, respose_num, top, rep_penalty):
    generated_response = generator(string, pad_token_id=tokenizer.eos_token_id,
                                   max_length=length_choice,
                                   temperature=temp,
                                   num_return_sequences=respose_num,
                                   top_k=top,
                                   repetition_penalty=rep_penalty)
    output = []
    output_2 = []
    for i, _ in enumerate(generated_response):
        text = generated_response[i]['generated_text'].replace('\n', ' ')
        output_2.append(
            f"{i+1}. Generated Response: {text}\n")
    for i, _ in enumerate(generated_response):
        output.append(dcc.Markdown(f'## {i+1}. Generated Response\n\n'))
        output.append(generated_response[i]['generated_text'])
        output.append(dcc.Markdown('---'))

    return output, output_2


offcanvas_evaluation = html.Div(
    [
        dbc.Button(['Report ', html.I(className="bi bi-flag-fill")], id='evaluation-button', n_clicks=0,
                   disabled=True, outline=True, color='light', style={'width': '100%', 'border-radius': '5px'}),
        dbc.Offcanvas(
            [
                dcc.Markdown('#### Your prompt was:'), html.Br(),
                html.P('', id='prompt', style={
                    'font-size': 20}),
                dcc.Markdown('---'),
                dcc.Markdown(
                    '#### The generated response(s) was:'), html.Br(),
                dcc.Markdown('', id='response',
                             style={'font-size': 20}),
                dcc.Markdown('---'),
                dcc.Markdown(
                    """
                    #### Evaluate the model's response(s) in the form below
                    
                    ---
                    """),
                html.Iframe(src="https://forms.gle/sdAqtx3QHGf6YAVE9", style={
                            "width": "100%", 'height': 1360, 'overflowY': 'scroll'})
            ],
            id="evaluation-offcanvas",
            scrollable=True,
            title="Evaluation 📝",
            placement='end',
            is_open=False,
            style={'width': '50vw'}
        ),
    ])


modal_info = html.Div(
    [
        dbc.Button(['Information ', html.I(className='bi bi-robot')], id='info-button', size='xl',
                   n_clicks=0, outline=True, color='light'),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle(dcc.Markdown(
                    '## Language Model Alignment 🎯'))),
                dbc.ModalBody([dcc.Markdown('''**Since the transformer architecture was proposed by _Vaswani et al._ (Google) in their seminal paper "_[Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)_," deep learning applied to NLP tasks has entered a new era: _the era of Large Language Models_ (LLM). Models such as [BERT](https://huggingface.co/docs/transformers/model_doc/bert), [GPT-3](https://arxiv.org/abs/2005.14165), [LaMDA](https://arxiv.org/abs/2201.08239), and [PaLM](https://arxiv.org/abs/2204.02311), are examples of LLM capable of solving many kinds of tasks.**''', style={'font-size': 20,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             'text-align': 'justify',
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             'text-justify': 'inter-word'}), html.Br(),
                               dcc.Markdown(
                                   '''**However, when use as generative models, such systems can provide false, toxic, or simply useless content for their controller. Following in the line of research from similar works (Kenton et al. [2021](https://arxiv.org/pdf/2103.14659.pdf), Ziegler et al. [2022](https://arxiv.org/pdf/2205.01663.pdf), Ouyang et al. [2022](https://arxiv.org/pdf/2203.02155.pdf), Romal et al. [2022](https://arxiv.org/abs/2201.08239)), this research tool seeks to evaluate and improve the _alignment_ of language models, i.e., _how well the responses of such models are aligned with the intentions of a human controller_.**''', style={'font-size': 20,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 'text-align': 'justify',
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 'text-justify': 'inter-word'}), html.Br(),
                               dcc.Markdown(
                                   '''**In the quest to create models that are better [aligned](https://arxiv.org/abs/1906.01820) with human intentions, we hope to be able to create safer and more efficient models. In this playground, you can submit prompts to a GPT model, and then evaluate the model's performance.**''', style={'font-size': 20,
                                                                                                                                                                                                                                                                                                                                          'text-align': 'justify',
                                                                                                                                                                                                                                                                                                                                          'text-justify': 'inter-word'}), html.Br(),
                               dcc.Markdown(
                                   '''**For example, you can give a demonstration in the form of a prompt, and ask a complition for an uncomplited task (_prompt engenniring_):**''', style={'font-size': 20,
                                                                                                                                                                                             'text-align': 'justify',
                                                                                                                                                                                             'text-justify': 'inter-word'}),
                               dcc.Markdown("""
                               
                               **Recipe for chocolate cake that does not use eggs and milk:**

                                - **_1 cup of all-purpose flour, 1/2 cup of cocoa powder, 1/2 teaspoon of baking soda, 1/2 teaspoon of baking powder, 1/4 teaspoon of salt, 1/2 cup of sugar, 1/4 cup of vegetable oil, 1/4 cup of water, 1 teaspoon of vanilla extract_.**

                                **Recipe for a strawberry cake that does not use eggs and milk:**

                               """, style={'font-size': 20,
                                           'text-align': 'justify',
                                           'text-justify': 'inter-word'}),
                               dcc.Markdown(
                                   '''**An aligned language model would produce a cake recipe (_even if not very appetizing_) without using eggs and milk.**''', style={'font-size': 20,
                                                                                                                                                                        'text-align': 'justify',
                                                                                                                                                                        'text-justify': 'inter-word'}), html.Br(),
                               dcc.Markdown('''**Thus, with this tool, you can search for examples of misalignment and later correct the model. Corrections and evaluations can be later used to fine-tune the model with something like _[Reinforcement Learning trough Human Feedback](https://huggingface.co/blog/rlhf)_.**''', style={'font-size': 20,
                                                                                                                                                                                                                                                                                                                                          'text-align': 'justify',
                                                                                                                                                                                                                                                                                                                                          'text-justify': 'inter-word'}),  html.Br(),
                               dcc.Markdown('''

                               ## Working with Sampling Parameters ⚙️

                               ---

                               **Language models usually generate text through _greedy search_, i.e., selecting the highest probability token at each autoregressive step. However, human language is not the output of a policy consecutively iterates a greedy policy. That's where sampling parameters come to aid us. They are the dials and knobs on the left portion of this page.**''', style={'font-size': 20,
                                                                                                                                                                                                                                                                                                                                                                                                      'text-align': 'justify',
                                                                                                                                                                                                                                                                                                                                                                                                      'text-justify': 'inter-word'}), html.Br(),
                               dcc.Markdown('''**With the proper sampling strategy, every token with a non-zero probability has a chance of being selected, and the different sampling parameters tweak the output generated by a model, e.g., by increasing the temperature parameter, which increases the entropy of the resulting softmax output, we can get a more diverse output.**''', style={'font-size': 20,
                                                                                                                                                                                                                                                                                                                                                                                                    'text-align': 'justify',
                                                                                                                                                                                                                                                                                                                                                                                                    'text-justify': 'inter-word'}), html.Br(),
                               dcc.Markdown(
                                   '''**Let's have a quick and holist review of what every parameter on the left panel controls:**''', style={'font-size': 20,
                                                                                                                                              'text-align': 'justify',
                                                                                                                                              'text-justify': 'inter-word'}), html.Br(),
                               dcc.Markdown(
                                   '''
                                   - **Response Length**: the maximum length of the response being generated by the model.
                                   - **Nº of Responses**: number of responses generated by the model.                                                                                                    'text-align': 'justify',
                                   - **Temperature**: a hyperparameter that is applied to logits to affect the final probabilities from the softmax. A higher temperature will make the model output seem less deterministic.
                                   - **Top-K**: a hyperparameter that is applied to ensure that the less probable words (the tail of the distribution) should not have any chance at all. Only top K (10 = 10%) probable tokens should be considered. This helps a model "stay on track" and not deviate from the topic.
                                   - **Repetition penalty**: as the name suggests, this hyperparameter is meant to avoid sentences that repeat themselves without adding context to the output.
                                   ''', style={'font-size': 20,
                                               'text-justify': 'inter-word'}), html.Br(),
                               dcc.Markdown('''

                               ## Model Card (DistilGPT2) 🤖

                               ---

                               **The model used in this demo is [DistilGPT2](https://huggingface.co/distilgpt2) (short for Distilled-GPT2), an English-language model pre-trained with the supervision of the smallest version of Generative Pre-trained Transformer 2 (GPT-2). The GPT-2 was introduced in [this paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) and first released on [this page](https://openai.com/blog/better-language-models/). DistilGPT2 has 82 million parameters and was developed using [knowledge distillation](https://huggingface.co/distilgpt2#knowledge-distillation), being a faster, lighter version of GPT-2. DistilGPT2 was trained using the [OpenWebTextCorpus](https://skylion007.github.io/OpenWebTextCorpus/) dataset, an open-source reproduction of OpenAI's WebText dataset.**''', style={'font-size': 20,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 'text-align': 'justify',
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 'text-justify': 'inter-word'}), html.Br(),
                               dcc.Markdown(
                                   '''

                                   ### Model Details 📊

                                   ---

                                   - **Developed by**: [Hugging Face](https://huggingface.co/).
                                   - **Model type**: Transformer-based Language Model.
                                   - **Languages**: English.
                                   - **License**: Apache 2.0.
                                   - **Resources for more information**: See this [repository](https://github.com/huggingface/transformers/tree/main/examples/research_projects/distillation), Sanh et al. ([2019](https://arxiv.org/abs/1910.01108)) publication, and this page to learn more about [GPT-2](https://openai.com/blog/better-language-models/).
                                   ''', style={'font-size': 20,
                                               'text-align': 'justify',
                                               'text-justify': 'inter-word'}), html.Br(),
                               dcc.Markdown(
                                   '''

                                   ### Limitations and Risks ☣️

                                   ---

                                   **CONTENT WARNING: This section contains content that is disturbing and offensive. It can propagate historical and current stereotypes.**
                                   
                                   **DistilGPT2 suffers from persistent bias issues, as highlighted in the demonstrative examples below. These issues are an inheritance of GPT-2. Issues related to GPT-2 are documented in this [publication](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) and this [model card](https://github.com/openai/gpt-2/blob/master/model_card.md).**''', style={'font-size': 20,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                   'text-align': 'justify',
                                                                                                                                                                                                                                                                                                                                                                                                                                                                   'text-justify': 'inter-word'}), html.Br(),
                               dcc.Markdown('''
                                ```python
                                from transformers import AutoTokenizer, pipeline

                                from transformers import pipeline, set_seed
                                generator = pipeline('text-generation', model='distilgpt2')
                                set_seed(42)

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
                                   '''

                                   ### Potential Uses & Out-of-scope Uses 🚫

                                   ---
                                   
                                   **GPT-2 was developed to be used by researchers to better understand large-scale generative language models, with possible secondary use cases including writing assistance, creative writing and art, and entertainment. OpenAI states in the GPT-2 [model card](https://github.com/openai/gpt-2/blob/master/model_card.md): "_Because large-scale language models like GPT-2 do not distinguish fact from fiction, we don't support use-cases that require the generated text to be true. Additionally, language models like GPT-2 reflect the biases inherent to the systems they were trained on, so we do not recommend that they be deployed into systems that interact with humans unless the deployers first carry out a study of biases relevant to the intended use-case_."**''', style={'font-size': 20,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      'text-align': 'justify',
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      'text-justify': 'inter-word'}), html.Br(),
                               dcc.Markdown(
                                   '''

                                   ### Environmental Impact 🌱

                                   ---

                                   **Carbon emissions were estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute):**''', style={'font-size': 20,
                                                                                                                                                                          'text-align': 'justify',
                                                                                                                                                                          'text-justify': 'inter-word'}), html.Br(),
                               dcc.Markdown(
                                   '''
                                   - **Hardware Type**: 8 16GB V100.
                                   - **Hours used**: 168 (1 week).
                                   - **Cloud Provider**: Azure.
                                   - **Compute Region**: assumed East US for calculations.
                                   - **Carbon Emitted**: 149.2 KG CO2.
                                   ''', style={'font-size': 20,
                                               'text-align': 'justify',
                                               'text-justify': 'inter-word'}), html.Br(),
                               ]),
                dbc.ModalFooter(
                    dbc.Button(
                        html.I(className="bi bi-x-circle"),
                        id='close-info-scroll',
                        className='ms-auto',
                        n_clicks=0,
                        size='lg'
                    )
                ),
            ],
            id='modal-info-scroll',
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
                external_stylesheets=[dbc.themes.SLATE, dbc.icons.BOOTSTRAP])

server = app.server
app.title = 'Alignment Playground 🤖'

app.layout = dbc.Container(
    fluid=True,
    children=[
        html.Div([html.H1('Alignment Playground 🤖 🦾 📚'),],
                 style={'textAlign': 'center',
                        'margin-top': '20px'}),
        html.Div([modal_info], style={
                 'textAlign': 'center', 'margin-top': '25px', 'margin-bottom': '20px'}),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dcc.Markdown('**`Minimum Response Length`**',
                                 style={'margin-left': '15px', 'margin-top': '5px'}),
                    html.Div([
                        dcc.Slider(10, 50, 10,
                                   value=10,
                                   id='min-length-slider'),
                    ], style={'margin-left': ' 5px', 'margin-bottom': '5px'}),
                    dcc.Markdown('**`Maximum Response Length`**',
                                 style={'margin-left': '15px', }),
                    html.Div([
                        dcc.Slider(100, 500, 100,
                                   value=100,
                                   id='max-length-slider'),
                    ], style={'margin-left': '5px', 'margin-bottom': '5px'}),
                    dcc.Markdown('**`Number of Responses`**',
                                 style={'margin-left': '15px'}),
                    html.Div([
                        dcc.Slider(1, 5, 1,
                                   value=1,
                                   id='response-slider'),
                    ], style={'margin-left': '5px', 'margin-bottom': '5px'}),
                    dcc.Markdown('**`Temperature`**',
                                 style={'margin-left': '15px'}),
                    html.Div([
                        dcc.Slider(0.1, 2.1, 0.1,
                                   marks=None,
                                   value=0.1,
                                   tooltip={'placement': 'bottom',
                                            'always_visible': True},
                                   id='temperature-slider'),
                    ], style={'margin-left': '5px', 'margin-bottom': '5px'}),
                    dcc.Markdown('**`Top-K`**', style={'margin-left': '15px'}),
                    html.Div([
                        dcc.Slider(10, 50, 10,
                                   value=10,
                                   id='topk-slider'),
                    ], style={'margin-left': '5px', 'margin-bottom': '5px'}),
                    dcc.Markdown('**`Repetition Penalty`**',
                                 style={'margin-left': '15px'}),
                    html.Div([
                        dcc.Slider(1.1, 2.1, 0.1,
                                   marks=None,
                                   value=1.1,
                                   tooltip={'placement': 'bottom',
                                            'always_visible': True},
                                   id='repetition-slider'),
                    ], style={'margin-left': '5px', 'margin-bottom': '10px'}),
                ], color='dark', outline=False, style={'margin-left': '15px'})
            ], md=4),
            dbc.Col([
                dcc.Textarea(
                    id='textarea-state',
                    value='',
                    style={'width': '100%', 'height': 200},
                ),
                html.Div([dbc.Button(['Submit ', html.I(className="bi bi-send-fill")], id='submit-button', n_clicks=0,
                         outline=True, color='light', style={'width': '100%', 'border-radius': '5px'})]),
                dcc.Loading(id='loading_0', type='circle', children=[html.Div(id='textarea-output-state', style={'whiteSpace': 'pre-line',
                                                                                                                 'height': '200px',
                                                                                                                 'overflowY': 'scroll',
                                                                                                                 'margin-top': '5px',
                                                                                                                 'margin-bottom': '5px',
                                                                                                                 })]),
                offcanvas_evaluation
            ], md=8),
        ]),

    ],
)


@app.callback(

    [Output('textarea-output-state', 'children'),
     Output('evaluation-button', 'disabled'),
     Output('prompt', 'children'),
     Output('response', 'children')],

    Input('submit-button', 'n_clicks_timestamp'),

    [State('textarea-state', 'value'),
     State('min-length-slider', 'value'),
     State('response-slider', 'value'),
     State('temperature-slider', 'value'),
     State('topk-slider', 'value'),
     State('repetition-slider', 'value'), ]
)
def update_output(
        click, value_txt,
        length_slider,
        response_slider,
        temperature_slider,
        topk_slider,
        repetition_slider):

    if click is not None:
        x, y = generate_text(value_txt, length_slider, temperature_slider,
                             response_slider, topk_slider, repetition_slider)
        return x, False, value_txt, y
    else:
        return '', True, '', ''


@app.callback(
    Output("evaluation-offcanvas", "is_open"),
    Input("evaluation-button", "n_clicks"),
    State("evaluation-offcanvas", "is_open"),
)
def toggle_offcanvas_scrollable(n1, is_open):
    if n1:
        return not is_open
    return is_open


@app.callback(
    Output('modal-body-evaluation', 'is_open'),
    [
        Input('report-button', 'n_clicks'),
        Input('close-evaluation-modal', 'n_clicks'),
    ],
    [State('modal-body-evaluation', 'is_open')],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output('modal-info-scroll', 'is_open'),
    [
        Input('info-button', 'n_clicks'),
        Input('close-info-scroll', 'n_clicks'),
    ],
    [State('modal-info-scroll', 'is_open')],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


if __name__ == '__main__':
    app.run_server(debug=False)
