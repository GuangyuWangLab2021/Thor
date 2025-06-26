import argparse
import json
import os
import sys
import scanpy as sc

try:
    from thor.utils import require_packages, get_library_id
except ModuleNotFoundError:
    from utils import require_packages, get_library_id
except:
    current = os.path.dirname(os.path.realpath(__file__))
    parent = os.path.dirname(current)
    sys.path.append(parent)
    from utils import require_packages, get_library_id
finally:
    pass


# https://dash.plotly.com/annotations
@require_packages("dash", "plotly")
def draw_roi(img, json_file="sample.json", title=None):
    import plotly.express as px
    from dash import Dash, Input, Output, callback, dcc, html, no_update
    fig = px.imshow(img)
    fig.update_layout(
        title_text=title,
        title_x=0.5,
        autosize=True,
        #width=2000,
        #height=800,
    )
    fig.update_layout(dragmode="drawclosedpath")
    config = {
        "modeBarButtonsToAdd":
            [
                #"drawline",
                #"drawopenpath",
                "drawclosedpath",
                #"drawcircle",
                "drawrect",
                "eraseshape",
            ],
        "fillFrame": False,
    }

    # Build the Dash app
    app = Dash(__name__)
    app.layout = html.Div(
        [
            dcc.Graph(
                id="fig-image",
                figure=fig,
                config=config,
                style={
                    'width': '100vw',
                    'height': '100vh'
                }
            ),
            dcc.Markdown(
                "Characteristics of shapes",
                style={
                    "background-color": "white",
                    "border": "solid 1px black",
                    "text-align": "center"
                }
            ),
            html.Pre(
                id="annotations-pre", style={'backgroundColor': 'white'}
            ),
        ]
    )

    # Define the callback function to handle the annotation data
    @callback(
        Output("annotations-pre", "children"),
        Input("fig-image", "relayoutData"),
        prevent_initial_call=True,
    )
    def on_new_annotation(relayout_data):
        if any(["shapes" in key for key in relayout_data]):
            json_object = json.dumps(relayout_data, indent=2)
            with open(json_file, "w") as outfile:
                outfile.write(json_object)
            return json_object
        return no_update

    return app


def draw_roi_in_browser(
    img,
    localhost="127.0.0.1",
    port=5000,
    json_file="sample.json",
    title=None,
):
    app = draw_roi(img, json_file=json_file, title=title)
    if __name__ == "__main__":
        # Open the browser and run the app
        if not os.environ.get("WERKZEUG_RUN_MAIN"):
            open_browser(localhost, port)
        # Otherwise, continue as normal
        app.run(debug=False, host=localhost, port=port)


def draw_roi_in_notebook(img, json_file="sample.json", title=None):
    app = draw_roi(img, json_file=json_file, title=title)
    app.run_server(mode='inline', debug=False)
    return app


@require_packages("webbrowser")
def open_browser(host, port):
    import webbrowser
    webbrowser.open_new(f"http://{host}:{port}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Draw shapes on an image and save the annotation data as JSON."
    )
    parser.add_argument(
        "adata_path", type=str, help="Path to the adata file that contains the lower-resolution image and corresponding scale factor"
    )
    parser.add_argument(
        "img_key", type=str, help="Resolution key of the image to be drawn on", default="hires", choices=["hires", "lowres"]
    )
    parser.add_argument(
        "--localhost", type=str, default="127.0.0.1", help="Local host address"
    )
    parser.add_argument("--port", type=int, default=5000, help="Port number")
    parser.add_argument(
        "--json_file", type=str, default="roi.json", help="JSON file name that saved the annotation data in the input resolution"
    )

    args = parser.parse_args()
    adata = sc.read_h5ad(args.adata_path)

    libray_id = get_library_id(adata)
    img = adata.uns["spatial"][libray_id]["images"][args.img_key]
    
    draw_roi_in_browser(
        img, args.localhost, args.port, args.json_file
    )
