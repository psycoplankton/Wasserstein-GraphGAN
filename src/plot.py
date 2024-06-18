import os
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from utils import MetricDictionary
import GraphGAN.config as config

files = {'nonfuzzy': config.nonfuzzy,
         'fuzzy': config.fuzzy}

metrics = MetricDictionary(files)


x_values = list(range(1, 41))

# Create an empty figure
fig = go.Figure()

# Add traces for each line
fig.add_trace(go.Scatter(x=x_values, y=metrics['fuzzy']['gen'], mode='lines', name='Generator Fuzzy', line=dict(color='black')))
#fig.add_trace(go.Scatter(x=x_values, y=b_values, mode='lines', name='Discriminator', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=x_values, y=metrics['fuzzy']['dis'], mode='lines', name='Discriminator Fuzzy', line=dict(color='green')))
#fig.add_trace(go.Scatter(x=x_values, y=d_values, mode='lines', name='Discriminator Macro F1', line=dict(color='yellow')))

# Customize layout
fig.update_layout(
    plot_bgcolor='white',  # Set background color to white
    paper_bgcolor='white',  # Set paper background color to white
    xaxis=dict(showgrid=True, gridcolor='lightgrey'),  # Show x-axis gridlines
    yaxis=dict(showgrid=True, gridcolor='lightgrey'),   # Show y-axis gridlines
    title='Bio_dmela Node2Vec with W Loss and RMSProp And Fuzzy Embeddings ',
    annotations=[
        dict(
            text="Embeddings generated using Node2Vec(walk_length=80, num_walks=40, window = 50)" , #description text            
            xref="paper",  # Set x position relative to the entire plot area
            yref="paper",  # Set y position relative to the entire plot area
            x=0.5,  # Set x position to the middle of the plot
            y=1.05,  # Set y position just above the title
            showarrow=False,  # Do not show an arrow
            font=dict(size=12, color="black")  # Set font size and color
        ),
        dict(
            text="",  # Another line of description text
            xref="paper",  # Set x position relative to the entire plot area
            yref="paper",  # Set y position relative to the entire plot area
            x=0.5,  # Set x position to the middle of the plot
            y=1.10,  # Set y position just below the first annotation
            showarrow=False,  # Do not show an arrow
            font=dict(size=12, color="black")  # Set font size and color
        )
    ]    # Add a title to the graph
)


# Show the plot
fig.show()