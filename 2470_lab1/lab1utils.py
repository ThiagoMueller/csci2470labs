import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import itertools

# It is perfectly okay if you have no idea what's happening in this file.
# Most of this file is about creating nice looking 3D visualizations with plotly.
# Also, you are not allowed to use the scikit-learn package in this course anyway.

default_offset = 0.5
default_ground_level = -5.0

default_plasma_min = 10.0
default_plasma_max = 25.0

def visualize_data(X, y_true, 
                  cmin=default_plasma_min, cmax=default_plasma_max):
    fig_linreg = go.Figure()
    x1, x2, y = X[:, 0], X[:, 1], y_true.reshape(-1)
    go_linreg = go.Scatter3d(
        x=x1, y=x2, z=y,
        mode="markers",
        marker=dict(size=4, color=y, colorscale="Plasma", 
                    cmin=cmin, cmax=cmax, opacity=0.8)
    )
    
    fig_linreg.add_trace(go_linreg)
    fig_linreg.update_layout(
        width=600, height=400, showlegend=False,
        margin = dict(l=0, r=0, b=0, t=0), 
        scene = dict(
            xaxis_title="x1",
            yaxis_title="x2",
            zaxis_title="y", 
            aspectratio= {"x":1.2, "y":1.2, "z":0.9}),
        scene_camera = dict(
            up=dict(x=0, y=0, z=1),
            eye=dict(x=-1.2, y=1.8, z=0.))
    )

    return fig_linreg


def get_mean_go(mean, 
                cmin=default_plasma_min, cmax=default_plasma_max):
    """
    go = graphics object
    """
    x1_grid, x2_grid = np.linspace(-8, 8, 5), np.linspace(-8, 8, 5)
    x1_mesh, x2_mesh = np.meshgrid(x1_grid, x2_grid, indexing="xy")
    y_pred_mesh = mean*np.ones(x1_mesh.shape)
    
    go_mean = go.Surface(
        x=x1_mesh, y=x2_mesh, z=y_pred_mesh, 
        colorscale="plasma", cmin=cmin, cmax=cmax, showscale=False, opacity=0.4)
    
    return go_mean


def linreg_answer_key(X, y_true):
    sklearn_reg = LinearRegression().fit(X, y_true.reshape(-1))
    coef_optimized = sklearn_reg.coef_.reshape(-1, 1)
    bias_optimized = sklearn_reg.intercept_
    
    return (coef_optimized, bias_optimized)


def get_linreg_go(coef, bias, 
                  cmin=default_plasma_min, cmax=default_plasma_max):
    """
    go = graphics object
    """
    b1, b2 = coef[0], coef[1]
    x1_grid, x2_grid = np.linspace(-8, 8, 5), np.linspace(-8, 8, 5)
    x1_mesh, x2_mesh = np.meshgrid(x1_grid, x2_grid, indexing="xy")
    y_pred_mesh = bias + b1*x1_mesh + b2*x2_mesh
    
    go_answer_key = go.Surface(
        x=x1_mesh, y=x2_mesh, z=y_pred_mesh, 
        colorscale="Plasma", cmin=cmin, cmax=cmax, showscale=False, opacity=0.4)
    
    return go_answer_key


def get_loss_mesh(X, y_true, b0):
    x1, x2, y = X[:, 0], X[:, 1], y_true.reshape(-1)
    
    b1_grid, b2_grid = np.linspace(-0.5, 1.5, 41), np.linspace(-1, 1, 41)
    b1_mesh, b2_mesh = np.meshgrid(b1_grid, b2_grid, indexing="ij")

    mse_list = []
    for each_b1, each_b2 in itertools.product(b1_grid, b2_grid):
        each_mse = np.mean((y - each_b1*x1 - each_b2*x2 - b0)**2)
        mse_list += [each_mse]

    mse_mesh = np.array(mse_list).reshape(41, 41)
    
    return b1_mesh, b2_mesh, mse_mesh


def visualize_loss_function(b1_mesh, b2_mesh, mse_mesh, 
                            ground_level=default_ground_level):
    fig_loss = go.Figure()
    go_loss = go.Surface(
        x=b1_mesh, y=b2_mesh, z=mse_mesh,
        colorscale="deep", cmin=1.3, cmax=12, opacity = 1, 
        showscale=True, colorbar_orientation="v", colorbar_len=0.75, colorbar_thickness=15,
        contours_z=dict(
            show=True, usecolormap=False, project_z=False,
            start=1.5, end=12, size=0.75)
    )
    fig_loss.add_trace(go_loss)
    
    go_loss_2D = go.Surface(
        x=b1_mesh, y=b2_mesh, z=ground_level*np.ones(mse_mesh.shape),
        colorscale="deep", cmin=1.3, cmax=12, surfacecolor=mse_mesh, 
        showscale=False, opacity = 0.8, 
    )
    fig_loss.add_trace(go_loss_2D)

    fig_loss.update_layout(
        width=600, height=400, showlegend=False,
        margin = dict(l=0, r=0, b=0, t=0), 
        scene = dict(
            xaxis_title="b1",
            yaxis_title="b2",
            zaxis_title="loss"),
        scene_camera = dict(
            up=dict(x=0, y=0, z=1),
            eye=dict(x=-1.2, y=1.8, z=0.6))
    )
    return fig_loss


def get_gradients_go(coef_trajectory, loss_array,
                     offset=default_offset, 
                     ground_level=default_ground_level):
    b1_trajectory = coef_trajectory[:, 0]
    b2_trajectory = coef_trajectory[:, 1]

    go_gradients = go.Scatter3d(
        x=b1_trajectory, y=b2_trajectory, z=(offset+loss_array),
        marker_size=4, marker_color="red",
        line_color="red", line_width=4, 
    )

    go_gradients_2D = go.Scatter3d(
        x=b1_trajectory, y=b2_trajectory, z=ground_level*np.ones(loss_array.shape),
        marker_size=2, marker_color="red",
        line_color="red", line_width=2, 
    )
    
    return go_gradients, go_gradients_2D


def visualize_trajectory(X, y_true, coef_trajectory, bias_trajectory):
    trace_list_linreg = []
    n_epochs = len(bias_trajectory)

    for each_coef, each_bias in zip(coef_trajectory, bias_trajectory):
        each_coef = each_coef.reshape(-1, 1)
        trace_list_linreg += [get_linreg_go(each_coef, each_bias)]

    fig_process = visualize_data(X, y_true)
    fig_process.add_traces(trace_list_linreg)

    for i in range(1+n_epochs):
        fig_process.data[i].visible = False

    fig_process.data[0].visible = True
    fig_process.data[1].visible = True

    steps = []
    for each_epoch in range(1, 1+n_epochs):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig_process.data)}],  # layout attribute
            label = f"{each_epoch-1}"
        )
        step["args"][0]["visible"][each_epoch] = True  # Toggle i'th trace to "visible"
        step["args"][0]["visible"][0] = True  
        steps += [step]

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Epoch  = "},
        pad={"t": 0, "b": 20},
        steps=steps
    )]

    fig_process.update_layout(
        width=600, height=600,
        scene_zaxis = dict(nticks=11, range=[0.0, 30.0]),
        sliders=sliders
    )

    return fig_process

