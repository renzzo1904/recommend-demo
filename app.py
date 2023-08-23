import gradio as gr
from main.models import ModelClass
from helpers.helper_functions import *
import pandas as pd

css_path = "page/style.css"
intro_html = "page/intro.html"

with open(intro_html, "r") as file:
    html_content = file.read()

with gr.Blocks(css=css_path) as demo:
    with gr.Row():
        df_gr = gr.Dataframe(pd.read_csv("examples/mini_df.csv"))
        html_intro = gr.HTML(value=html_content)

    with gr.Row():
        img_kwargs = {
            "container": True,
            "show_download_button": False,
            "scale": 1,
            "show_label": False,
        }

        im1 = gr.Image("page/skate.jpeg", **img_kwargs)
        im2 = gr.Image("page/ball.jpeg", **img_kwargs)
        im3 = gr.Image("page/globe.jpeg", **img_kwargs)

    with gr.Row():
        txt_kwargs = {
            "max_lines": 10,
            "interactive": False,
            "show_label": False,
            "container": False,
        }

        txt1 = gr.Textbox(
            value="skateboard punk skate vans skating cool board", **txt_kwargs
        )
        txt2 = gr.Textbox(
            value="football soccer ball sports toy worldcup", **txt_kwargs
        )
        txt3 = gr.Textbox(
            value="globe world geogrpahy class country continent", **txt_kwargs
        )

    with gr.Row():
        bt1 = gr.Button("Recommend based on this", elem_id="button ")
        bt2 = gr.Button("Recommend based on this", elem_id="button")
        bt3 = gr.Button("Recommend based on this", elem_id="button")

    html = gr.HTML(value="Hello World")

    # model = ModelClass()

    def get_predictions(user_likes):
        pass

    bt1.click(get_predictions, inputs=txt1, outputs=html)
    bt2.click(get_predictions, inputs=txt2, outputs=html)
    bt3.click(get_predictions, inputs=txt3, outputs=html)


if __name__ == "__main__":
    demo.launch(server_port=8080)
