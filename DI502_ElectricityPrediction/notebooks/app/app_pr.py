import psutil
def pusage():
    process = psutil.Process()
    print( int(process.memory_info().rss ) / (1024*1024) )


import pandas as pd
import numpy as np
import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
import plotly.express as px
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from dash.exceptions import PreventUpdate
import os
import joblib
import zipfile



# Load data function
def load_data():
    # Specify the ZIP file name
    zip_filename = "filtered.zip"

    # Extract the model file from the ZIP archive
    with zipfile.ZipFile(zip_filename, "r") as archive:
        # Extract the model file (named "your_model.pkl" in this example)
        archive.extract("filtered.pkl")
        
    # Load the model
    df = joblib.load("filtered.pkl")  # Replace with "pickle.load" if you used pickle

    os.remove("filtered.pkl")

    return df



# Load data and perform log transformation
train_data = load_data()

train_data['log_meter_reading']=np.log1p(train_data['meter_reading'])
train_data['log_square_feet']=np.log1p(train_data['square_feet'])
train_data['timestamp'] = pd.to_datetime(train_data['timestamp'])

# Create new columns
train_data['hour'] = train_data['timestamp'].dt.hour.astype(np.uint8)
train_data['month'] = train_data['timestamp'].dt.month.astype(np.uint8)
train_data['week'] = train_data['timestamp'].dt.isocalendar().week.astype(np.uint8)
train_data[['year','weekofyear','dayofweek']]= np.uint16(train_data['timestamp'].dt.isocalendar())



XGBoost_MODEL_DESCRIPTION = '''
    The eXtreme Gradient Boosting (XGBoost) forecasting model was trained on historical meter readings, weather, and building data
    from 2016-2017. Meter readings are from the buildings in site_id - 1 and site_id - 6.
'''

Dataset_DESCRIPTION = '''
    The dataset is taken from [ASHRAE Great Energy Predictor III Competition](https://www.kaggle.com/c/ashrae-energy-prediction) .
    The dataset is self-contained and it has Competition Use , Non-Commercial Purposes & Academic Research, Commercial Purposes licenses.
'''

XGBresults = pd.read_csv('XGB_results.csv')

# Load results information from CSV
results_info = pd.read_csv('results_info.csv')


app = dash.Dash(
    external_stylesheets=[dbc.themes.LUX],
    suppress_callback_exceptions=True
)


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),    
    html.Div([
        dcc.Tabs(
            id='tabs',
            value='/',
            children=[
                dcc.Tab(
                    label= " Home",
                    value='/',
                    style={'background-color': '#F5F5F5', 'color': '#333333', 'border': 'none'},
                    selected_style={'background-color': '#F5F5F5', 'color': '#333333', 'border': 'none'}
                ),
                dcc.Tab(
                    label='Team',
                    value='/team',
                    style={'background-color': '#F5F5F5', 'color': '#333333', 'border': 'none'},
                    selected_style={'background-color': '#F5F5F5', 'color': '#333333', 'border': 'none'}
                ),
                dcc.Tab(
                    label='Dataset',
                    value='/dataset',
                    style={'background-color': '#F5F5F5', 'color': '#333333', 'border': 'none'},
                    selected_style={'background-color': '#F5F5F5', 'color': '#333333', 'border': 'none'}
                ),
                dcc.Tab(
                    label='Model',
                    value='/xgboost',
                    style={'background-color': '#F5F5F5', 'color': '#333333', 'border': 'none'},
                    selected_style={'background-color': '#F5F5F5', 'color': '#333333', 'border': 'none'}
                ),
            ],
            style={
                'background-color': '#F5F5F5',
                'padding': '10px',
                'flex-grow': 1,
                'margin': '0',
            },
            className='tabs',
        ),
    ],
    style={
        'opacity': 1,
        'transition': 'opacity 1s ease-in-out'
    },
    className='page-tab-container'),
    
    html.Div(id='page-transition-container', className='page-transition-container is-active'),
    
    html.Div(id='tabs-content', className='tabs-content-container fadeIn'),  # Apply the fadeIn class here
    html.Link(rel='stylesheet', href='/assets/styles.css'),
])

# Callback to update the page content based on the selected tab
@app.callback(
    Output('page-transition-container', 'className'),
    [Input('url', 'pathname')]
)
def update_page_container_class(pathname):
    return 'page-transition-container is-active'

# Callback to update the content of the selected tab with fade-in effect
@app.callback(
    Output('tabs-content', 'children'),
    [Input('tabs', 'value')]
)
def update_tabs_content(selected_tab):
    if selected_tab == '/':
        content = index_page
    elif selected_tab == '/team':
        content = team_layout
    elif selected_tab == '/dataset':
        content = dataset_layout
    elif selected_tab == '/xgboost':
        content = xgboost_layout
    else:
        content = html.Div()

    return content


index_page = html.Div([
    html.Div(id='page-content', className='page-content'),
    html.Div([
    dbc.Row([
        dbc.Col([
            html.H1(
                children="Welcome to Foresquad",
                className='fadeIn',  # Apply the fadeIn class
                style={'margin-bottom': '0'},
            ),
        ], width=5),
        dbc.Col(width=5),
    ], justify='center'),
    ],
    style={
        'background-image': f'url({app.get_asset_url("elec.png")})',
        'background-size': 'cover',
        'background-position': 'center',
        'background-repeat': 'no-repeat',
        'height': '60vh',  # Adjust the height to cover the entire viewport
        'display': 'flex',
        'opacity': 0.8,  # Adjust the opacity value (0.0 to 1.0)
        'flex-direction': 'column',
        'align-items': 'left',
        'justify-content': 'center',
        'margin': '0',  # Reset margin to zero
        'padding': '0',  # Reset padding to zero
    }),
    
    html.Br(),
    dbc.Row([
        dbc.Col(
            html.Div([
                html.H4(
                    children="To what extent do weather, weekday, and primary use determine total electricity consumption? Click a tab to explore."
                ),
            ]), width=7
        ),
        dbc.Col(width=3),
    ], justify="center"),
    html.Br(),
    
    # Add colored sections with sliding in animations here
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H3("Section 1 Title"),
                html.P("Section 1 content"),
            ], className='slideIn',  # Apply the fadeIn class
            style={'background-color': '#FADBD8', 'padding': '20px'}),
        ], width=4),
        dbc.Col([
            html.Div([
                html.H3("Section 2 Title"),
                html.P("Section 2 content"),
            ], className='slideIn',  # Apply the fadeIn class
            style={'background-color': '#85C1E9', 'padding': '20px'}),
        ], width=4),
        dbc.Col([
            html.Div([
                html.H3("Section 3 Title"),
                html.P("Section 3 content"),
            ], className='slideIn',  # Apply the fadeIn class
            style={'background-color': '#AED6F1', 'padding': '20px'}),
        ], width=4),
    ], justify='center'),
    

])
    


team_layout = html.Div([    
    html.H3('Meet Our Team',style={'margin-left': '20px','margin-top': '20px'}),
    html.Div([
        html.Link(
            href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css",
            rel="stylesheet",
            integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN"
        ),
    ]),
    html.Div([

        html.Div([
            html.Img(src='/assets/ali.jpeg', style={'width': '150px','padding-bottom': '10px' , 'filter': 'grayscale(100%)'}),
            html.H4('Ali Rifat Kulu'),
            html.P(style={'font-weight': 'bold'}, children=['Role:', ' Project Manager']),
            html.P('Ali is the driving force behind our projects, ensuring everything runs smoothly. With a passion for innovation, he leads the team to success.'),
            html.A(
                href='https://www.linkedin.com/in/alirifatkulu/',
                children=[html.I(className='fab fa-linkedin', style={'font-size': '20px', 'color': '#0077b5'}), " LinkedIn"],
                target='_blank',  # Open the link in a new tab
                style={'text-decoration': 'none', 'margin-top': '10px'}  # Adjust styling as needed
            ),
        ], style={'margin': '10px', 'padding': '10px', 'width':'23.5%', 'border': '1px solid #CCCCCC', 'border-radius': '5px', 'animation': 'fadeIn 1s ease-in-out'}),

        html.Div([
            html.Img(src='/assets/ezgi.jpeg', style={'width': '150px','padding-bottom': '10px' , 'filter': 'grayscale(100%)'}),
            html.H4('Ezgi Cavus'),
            html.P(style={'font-weight': 'bold'}, children=['Role:',  'Designer']),
            html.P('Ezgi is our creative genius. With an eye for aesthetics, she transforms ideas into visually stunning designs that captivate our audience.'),
            html.A(
                href='https://www.linkedin.com/in/ezgi-tuncay-çavuş-6bb43a207/',
                children=[html.I(className='fab fa-linkedin', style={'font-size': '20px', 'color': '#0077b5'}), " LinkedIn"],
                target='_blank',  # Open the link in a new tab
                style={'text-decoration': 'none', 'margin-top': '10px'}  # Adjust styling as needed
            ),
        ], style={'margin': '10px', 'padding': '10px', 'width':'23.5%', 'border': '1px solid #CCCCCC', 'border-radius': '5px', 'animation': 'fadeIn 2s ease-in-out'}),

        html.Div([
            html.Img(src='/assets/goksu.jpeg', style={'width': '150px','padding-bottom': '10px' , 'filter': 'grayscale(100%)'}),
            html.H4('Goksu Uzunturk'),
            html.P(style={'font-weight': 'bold'}, children=['Role:', 'Developer']),
            html.P('Goksu is our coding ninja. Armed with the latest technologies, she turns concepts into functional and efficient software solutions.'),
            html.A(
                href='https://www.linkedin.com/in/g%C3%B6ksu-uzunt%C3%BCrk?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app',
                children=[html.I(className='fab fa-linkedin', style={'font-size': '20px', 'color': '#0077b5'}), " LinkedIn"],
                target='_blank',  # Open the link in a new tab
                style={'text-decoration': 'none', 'margin-top': '10px'}  # Adjust styling as needed
            ),
        ], style={'margin': '10px', 'padding': '10px', 'width':'23.5%', 'border': '1px solid #CCCCCC', 'border-radius': '5px','animation': 'fadeIn 3s ease-in-out'}),

        html.Div([
            html.Img(src='/assets/ozge.jpeg', style={'width': '150px','padding-bottom': '10px' , 'filter': 'grayscale(100%)'}),
            html.H4('Ozge Ozkul'),
            html.P(style={'font-weight': 'bold'}, children=['Role:','Data Scientist']),
            html.P('Ozge is our data wizard. She navigates through data with precision, extracting insights that guide our decision-making process.'),
            html.A(                
                href='https://www.linkedin.com/in/ozge-ozkul?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app',             
                children=[html.I(className='fab fa-linkedin', style={'font-size': '20px', 'color': '#0077b5'}), " LinkedIn"],
                target='_blank',  # Open the link in a new tab
                style={'text-decoration': 'none', 'margin-top': '10px'}  # Adjust styling as needed
            ),
        ],style={'margin': '10px', 'padding': '10px', 'width':'23.5%', 'border': '1px solid #CCCCCC', 'border-radius': '5px', 'animation': 'fadeIn 4s ease-in-out'}),
    ], style={'display': 'flex', 'justify-content': 'space-between'}),

        html.Div([
        
        html.Form([
            html.H4('Contact Us', style={'margin-left': '30px', 'margin-top': '20px'}),
            
            html.Label('Name', style={'display': 'block'}),
            dcc.Input(type='text', id='name', style={'display': 'block'}),
            
            html.Label('Email', style={'display': 'block'}),
            dcc.Input(type='email', id='email', style={'display': 'block'}),
            
            html.Label('Message', style={'display': 'block'}),
            dcc.Textarea(id='message', style={'display': 'block'}),
            
            html.Button('Send', id='send-button', n_clicks=0, style={'display': 'block', 'margin-top': '10px'}),
        ], style={'margin-top': '20px', 'margin-bottom': '40px', 'border': '1px solid #CCCCCC', 'background-color': '#F5F5F5', 'padding': '60px'})
        ])
])
# Update the mailto link dynamically based on form input
@app.callback(
    Output('send-link', 'href'),
    [Input('name', 'value'),
     Input('email', 'value'),
     Input('message', 'value')]
)
def update_mailto_link(name, email, message):
    mailto = f'mailto:info@example.com?subject=Contact%20Us&body=Name:%20{name}%0D%0AEmail:%20{email}%0D%0AMessage:%20{message}'
    return mailto

dataset_layout = html.Div([
    html.H3('Dataset',style={'margin-left': '20px','margin-top': '20px'}),
    html.P(Dataset_DESCRIPTION,style={'margin-left': '20px','margin-top': '20px'}),
    

    dcc.Dropdown(
    
        id='time-interval-dropdown',
        options=[
            {'label': 'Hourly', 'value': 'hourly'},
            {'label': 'Weekly', 'value': 'weekly'},
            {'label': 'Monthly', 'value': 'monthly'},
        ],
        value='hourly',  # Default selection
        style={'width': '33%', 'height': '%50'}
    ),
    dcc.Dropdown(
        id='site-dropdown',
        options=[{'label': f'Site {i}', 'value': i} for i in range(0, 15)],
        value=1,  # Default selection
        style={'width': '33%'}
    ),

    dcc.Dropdown(
        id='primary-use-dropdown',
        options=[{'label': primary_use, 'value': primary_use} for primary_use in train_data['primary_use'].unique()],
        value='Education',  # Default selection
        style={'width': '33%', 'height': '%50'}
    ),

    

    dcc.Graph(id='model-performance-graph', style={'display': 'inline-block', 'width': '50%', 'height': '50%'}),
    dcc.Graph(id='site-distribution-plot', style={'display': 'inline-block', 'width': '50%', 'height': '50%'}),
    dcc.Graph(id='primary-use-distribution-plot', style={'width': '100%', 'height': '80vh'}),
    
],
    
style={
    'background-color': 'lavender'
    
})


# Callback to update the site distribution plot based on the selected site
@app.callback(
    Output('site-distribution-plot', 'figure'),
    [Input('site-dropdown', 'value')]
)
def update_site_distribution_plot(selected_site):
    # Filter the data based on the selected site
    filtered_data = train_data[train_data['site_id'] == selected_site]

    # Check if the filtered data is empty
    if filtered_data.empty:
        raise PreventUpdate(f"No data for site ID: {selected_site}")

    # Check if the required column is present in the data
    if 'log_meter_reading' not in filtered_data.columns:
        raise PreventUpdate("Column 'log_meter_reading' not found in the filtered data")

    # Create the KDE plot
    fig = go.Figure()

    # Plot the KDE using Plotly
    fig.add_trace(go.Histogram(
        x=filtered_data['log_meter_reading'],
        histnorm='probability',
        name=f'site_id={selected_site}',
        marker_color='blue',
        opacity=0.7
    ))

    fig.update_layout(
        xaxis_title='Log-transformed Meter Reading',
        yaxis_title='Density',
        title=f'Distribution Plot - Site ID: {selected_site}',
        paper_bgcolor='lavender'  # Set the color of the background
    )

    return fig
# Callback to update the primary use distribution plot based on the selected primary use
@app.callback(
    Output('primary-use-distribution-plot', 'figure'),
    [Input('primary-use-dropdown', 'value')]
)
def update_primary_use_distribution_plot(selected_primary_use):
    # Filter the data based on the selected primary use
    filtered_data = train_data[train_data['primary_use'] == selected_primary_use]

    # Check if the filtered data is empty
    if filtered_data.empty:
        raise PreventUpdate(f"No data for primary use: {selected_primary_use}")

    # Check if the required column is present in the data
    if 'log_meter_reading' not in filtered_data.columns:
        raise PreventUpdate("Column 'log_meter_reading' not found in the filtered data")

    # Create the histogram plot
    fig = go.Figure()

    # Plot the KDE using Plotly
    fig.add_trace(go.Histogram(
        x=filtered_data['log_meter_reading'],
        histnorm='probability',
        name=f'primary_use={selected_primary_use}',
        marker_color='orange',
        opacity=0.7
    ))

    fig.update_layout(
        xaxis_title='Log-transformed Meter Reading',
        yaxis_title='Density',
        title=f'Distribution Plot - Primary Use: {selected_primary_use}',
        paper_bgcolor='lavender'  # Set the color of the background
    )

    return fig

@app.callback(
    Output('model-performance-graph', 'figure'),
    [Input('time-interval-dropdown', 'value')]
)
def update_model_performance(selected_interval):
    # Your existing figure generation code for 'hour_mean'
    hour_mean = train_data.groupby('hour')['log_meter_reading'].mean()
    fig_hour_mean = make_subplots(rows=1, cols=1)
    trace_hour_mean = go.Scatter(x=hour_mean.index, y=hour_mean.values, mode='lines', line=dict(color='skyblue'))
    fig_hour_mean.add_trace(trace_hour_mean)
    fig_hour_mean.update_layout(
        xaxis=dict(title='Hour'),
        yaxis=dict(title='Mean of Log Meter readings'),
        paper_bgcolor='lavender',
        showlegend=False
    )

    # Your existing figure generation code for 'day_df'
    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
    day_df = train_data.groupby(['dayofweek']).log_meter_reading.mean().reset_index()
    fig8 = make_subplots(rows=1, cols=1)
    trace = go.Scatter(x=day_df['dayofweek'], y=day_df['log_meter_reading'], mode='lines', line=dict(color='purple'))
    fig8.add_trace(trace)
    fig8.update_layout(xaxis=dict(title='Days of the week'),
                       yaxis=dict(title='Log of Meter readings'),
                       paper_bgcolor='lavender',
                       showlegend=False)

    # Your existing figure generation code for 'month_mean_data'
    month_mean_data = train_data.groupby('month')['log_meter_reading'].mean()
    fig_month_mean = px.line(
        data_frame=pd.DataFrame({'Month': month_mean_data.index, 'Mean Meter Reading': month_mean_data.values}),
        x='Month',
        y='Mean Meter Reading',
        labels={'x': 'Month', 'y': 'Mean Meter Reading'},
        title='Monthly Mean Meter Reading',
        line_shape='linear',
        render_mode='svg'
    )
    fig_month_mean.update_layout(paper_bgcolor='lavender')

    # Choose the appropriate figure based on the selected interval
    if selected_interval == 'hourly':
        return fig_hour_mean
    elif selected_interval == 'weekly':
        return fig8
    elif selected_interval == 'monthly':
        return fig_month_mean


# Manually assign date values to each fold
fold_date_mapping = {
    0: '02.03.2016 - 02.05.2016',
    1: '02.05.2016 - 02.07.2016',
    2: '02.07.2016 - 01.09.2016',
    3: '01.09.2016 - 31.10.2016',
    4: '31.10.2016 - 31.12.2016'
}

# Add the layout to the app callback
xgboost_layout = html.Div([
    html.H3('Model Predictions',style={'margin-left': '20px','margin-top': '20px'}),
    html.P(XGBoost_MODEL_DESCRIPTION,style={'margin-left': '20px','margin-top': '20px'}),
    

    # Dropdown for selecting fold
    dcc.Dropdown(
        id='fold-dropdown',
        options=[{'label': f'Date: {fold_date_mapping[idx]}', 'value': idx} for idx in range(len(fold_date_mapping))],
        value=0  # Default selected fold
    ),

    
    # Graph for displaying selected fold
    dcc.Graph(id='fold-plot'),

    # Metric Selection Dropdown
    dcc.Dropdown(
            id='metric-dropdown',
            options=[
                {'label': 'MAE', 'value': 'mae'},
                {'label': 'R2', 'value': 'r2'},
                {'label': 'MSE', 'value': 'mse'},
            ],
            value='mse',
            multi=False,
            style={'width': '50%'}  # Corrected indentation
    ),
    # Display the metric definition
    html.Div(id='metric-definition',style={'margin-left': '65px','margin-top': '20px'}),

    # Create a single graph for both train and test data
    dcc.Graph(id='metric-plot'),    
],
    style={
        'background-color': 'lavender'
})

# Callback to update the displayed graph based on the selected fold
@app.callback(
    Output('fold-plot', 'figure'),
    [Input('fold-dropdown', 'value')]
)
def update_fold_plot(selected_fold):
    actual_path = results_info['actual_data_paths'][selected_fold]
    predicted_path = results_info['predicted_data_paths'][selected_fold]

    actual_data = pd.read_csv(actual_path)
    predicted_data = pd.read_csv(predicted_path)

    figure = {
        'data': [
            {
                'x': actual_data['timestamp'],
                'y': actual_data['actual'],
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Actual'
            },
            {
                'x': predicted_data['timestamp'],
                'y': predicted_data['predicted_test'],
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Predicted'
            }
        ],
        'layout': {
            'title': f'Fold {selected_fold + 1} - Aggregated Daily Predictions vs Actual Values',
            'xaxis': {'title': 'Date'},
            'yaxis': {'title': 'Target Variable'},
        }
    }

    return figure


tabs_content = {
    '/': index_page,
    '/team':team_layout,
    '/dataset': dataset_layout,
    '/xgboost': xgboost_layout,
}


# Callback to update the graph and metric definition based on the selected metric
@app.callback(
    [Output('metric-plot', 'figure'),
     Output('metric-definition', 'children')],  # Corrected syntax here
    [Input('metric-dropdown', 'value')]
)
def update_metric_plot_and_definition(selected_metric):
    # Your existing code for updating the metric plot goes here

    # Choose the appropriate graph based on the selected metric
    title = f'Training and Test {selected_metric.upper()}'

    train_column = f'train_{selected_metric}'
    test_column = f'test_{selected_metric}'

    if train_column not in XGBresults.columns or test_column not in XGBresults.columns:
        raise PreventUpdate

    y_train = XGBresults[train_column]
    y_test = XGBresults[test_column]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=XGBresults['fold'], y=y_train, mode='lines+markers', name='Train'))
    fig.add_trace(go.Scatter(x=XGBresults['fold'], y=y_test, mode='lines+markers', name='Test'))  # Corrected x values

    fig.update_layout(title=title, xaxis_title='Fold', yaxis_title=selected_metric.upper())


    

    # Define metric definitions
    metric_definitions = {
        'mae': 'Mean Absolute Error (MAE) is the average absolute difference between the actual and predicted values.',
        'r2': 'R-squared (R2) is a measure of how well the predicted values match the actual values.',
        'mse': 'Mean Squared Error (MSE) is the average of the squared differences between the actual and predicted values.'
        # Add more metrics and definitions as needed
    }

    # Get the selected metric definition
    metric_definition = metric_definitions.get(selected_metric, 'No definition available for this metric.')

    return fig, metric_definition




  

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=80)


