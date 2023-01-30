import json
import dash
import time
import string
import unidecode
from tensorflow import keras
import dash_bootstrap_components as dbc
from dash import dcc, html, Output, Input, State
from keras_preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer, tokenizer_from_json

model = keras.models.load_model('models\senti_model_sigmoid.h5')

with open('models\\tokenizer_senti_model_en.json') as fp:
    data = json.load(fp)
    tokenizer = tokenizer_from_json(data)
    word_index = tokenizer.word_index
    fp.close()


def textbox(text, box='other'):
    style = {
        'max-width': '55%',
        'width': 'max-content',
        'padding': '10px 15px',
        'border-radius': '25px',
    }

    if box == 'self':
        style['margin-left'] = 'auto'
        style['margin-right'] = 0

        color = 'primary'
        inverse = True

    elif box == 'other':
        style['margin-left'] = 0
        style['margin-right'] = 'auto'

        color = 'light'
        inverse = False

    else:
        raise ValueError('Incorrect option for `box`.')

    return dbc.Card(text, style=style, body=True, color=color, inverse=inverse)


conversation = html.Div(
    style={
        'width': '80%',
        'max-width': '600px',
        'height': '35vh',
        'margin': 'auto',
        'margin-top': '100px',
        'overflow-y': 'auto',
    },
    id='display-prediction',
)


controls = dbc.InputGroup(
    style={'width': '80%', 'max-width': '600px', 'margin': 'auto'},
    children=[
        dbc.Input(id='user-input',
                  placeholder='Write a comment...', type='text'),
        dbc.InputGroup(dbc.Button('Submit', size='lg', id='submit')),
    ],
)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server
app.title = 'Sentiment Classifier'

app.layout = dbc.Container(
    fluid=True,
    children=[
        html.Div([html.H1('Sentiment Classifier 🤖', style={
            'margin-top': '20px'}),], style={'textAlign': 'center'}),
        html.Hr(),
        dbc.Row([
            dbc.Col([
                dcc.Store(id='store-data', data=''),
                dcc.Loading(id='loading_0', type='circle',
                            children=[conversation]),
                controls,
            ], md=12),
        ]),
    ],
)


@app.callback(
    Output('display-prediction', 'children'),
    [
        Input('store-data', 'data')]
)
def update_display(sentiment_analysis):
    time.sleep(2)
    return [
        textbox(sentiment_analysis, box='self') if i % 2 == 0 else textbox(
            sentiment_analysis, box='other')
        for i, sentiment_analysis in enumerate(sentiment_analysis)
    ]


@app.callback(
    [
        Output('store-data', 'data'),
        Output('user-input', 'value')
    ],

    [
        Input('submit', 'n_clicks'),
        Input('user-input', 'n_submit')
    ],

    [
        State('user-input', 'value'),
        State('store-data', 'data')
    ]
)
def run_senti_model(n_clicks, n_submit, user_input, sentiment_analysis):
    sentiment_analysis = sentiment_analysis or []
    if n_clicks == 0:
        sentiment_analysis.append('🎭')
        sentiment_analysis.append('How do you feel?')
        return sentiment_analysis, ''

    if user_input is None or user_input == '':
        sentiment_analysis.append('🎭')
        sentiment_analysis.append('How do you feel?')
        return sentiment_analysis, ''

    else:
        texto = user_input
        texto = texto.translate(str.maketrans('', '', string.punctuation))
        texto = texto.lower()
        texto = unidecode.unidecode(texto)

        sequence = tokenizer.texts_to_sequences([texto])

        padded_text = keras.preprocessing.sequence.pad_sequences(
            sequence, maxlen=256, truncating='post')

        prediction = model.predict(padded_text, verbose=0)

        pred_neg = 1 - prediction[0][0]
        pred_neg = '{:,.2f}'.format(pred_neg * 100) + ' %'
        neg = f'Negative Sentiment 😔 {pred_neg}'

        pred_pos = prediction[0][0]
        pred_pos = '{:,.2f}'.format(pred_pos * 100) + ' %'
        pos = f'Positive Sentiment 😊 {pred_pos}'

        responde = f'''{neg}
        {pos}
        '''

        sentiment_analysis.append(user_input)
        sentiment_analysis.append(responde)

        return sentiment_analysis, ''


if __name__ == '__main__':
    app.run_server(debug=False)
