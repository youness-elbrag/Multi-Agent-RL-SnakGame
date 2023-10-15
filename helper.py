import plotly.graph_objs as go
from plotly.subplots import make_subplots
import IPython.display as ipyd

# Create a Plotly figure
def create_figure(scores, mean_scores):
    fig = make_subplots(rows=1, cols=1)

    # Add the 'Scores' trace
    fig.add_trace(go.Scatter(x=list(range(1, len(scores) + 1)), y=scores, name='Scores'))

    # Add the 'Mean Scores' trace
    fig.add_trace(go.Scatter(x=list(range(1, len(mean_scores) + 1)), y=mean_scores, name='Mean Scores'))

    # Set the layout
    fig.update_layout(
        title='Training...',
        xaxis_title='Number of Games',
        yaxis_title='Score',
        yaxis_range=[0, max(scores + mean_scores)],
    )

    return fig

def plot(scores, mean_scores):
    # Create the Plotly figure
    fig = create_figure(scores, mean_scores)

    # Display the figure in Jupyter Notebook
    ipyd.display(ipyd.HTML('<script src="/static/components/requirejs/require.js"></script>'))
    ipyd.display(fig)
