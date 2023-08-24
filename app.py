"Appplication file"

import gradio as gr
import pandas as pd
import numpy as np
import random

from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler

from main.models import ModelClass
from helpers.helper_functions import create_profiles, generate_inventory

# css_path = "page/style.css"
intro_html = "page/intro.html"

with open(intro_html, "r") as file:
    html_content = file.read()


class UI:
    def __init__(self) -> None:
        self.min_df = pd.read_csv("examples/mini_df.csv")
        self.users_df = pd.read_hdf("examples/users.h5", key="df_key")
        self.df = pd.read_hdf("examples/data.h5", key="df_key")
        self.model = ModelClass()

    def create_ui(self):
        "Constructing the UI"
        with gr.Blocks(theme="gradio/monochrome") as self.demo:
            with gr.Row():
                gr.HTML(value=html_content, scale=2)

                with gr.Column(scale=1):
                    gr.Markdown(
                        """### Dataset used to feed the Model Recommendations Pattern 
                        Something like this is going to be obtained through page analytics."""
                    )

                    df_kwargs = {
                        "scale": 1,
                        "headers": self.min_df.columns.tolist(),
                        "interactive": False,
                        "type": "pandas",
                        "row_count": (5, "fixed"),
                        "col_count": (9, "fixed"),
                    }

                    gr.Dataframe(self.min_df, **df_kwargs)

            with gr.Row():
                img_kwargs = {
                    "container": True,
                    "show_download_button": False,
                    "scale": 1,
                    "show_label": False,
                    "shape":(512,512)
                }

                im1 = gr.Image("page/flashlight.jpeg", **img_kwargs)
                im2 = gr.Image("page/ball.jpeg", **img_kwargs)
                im3 = gr.Image("page/teddy.jpeg", **img_kwargs)

            with gr.Row():
                txt_kwargs = {
                    "max_lines": 10,
                    "interactive": False,
                    "show_label": False,
                    "container": False,
                    "info": "Tags that model uses for recommendation",
                }

                txt1 = gr.Textbox(value="linterna luz casa", **txt_kwargs)
                txt2 = gr.Textbox(value="football soccer ball ", **txt_kwargs)
                txt3 = gr.Textbox(value="oso teddy peluche sleep  ", **txt_kwargs)

            with gr.Row():
                bt1 = gr.Button("Recommend based on this", elem_id="button ")
                bt2 = gr.Button("Recommend based on this", elem_id="button")
                bt3 = gr.Button("Recommend based on this", elem_id="button")

            with gr.Row():
                html1 = gr.HTML(value="")

                with gr.Column():
                    txt_box = gr.Textbox(
                        "", container=True, info="Try you own Likes list", label="♥️"
                    )
                    with gr.Row():
                        txt_button = gr.Button("Submit")
                        clear_button = gr.ClearButton()
                    html2 = gr.HTML(value="")

            clear_button.add(txt_box)
            txt_button.click(
                self.get_predictions, inputs=txt_box, outputs=html1, queue=False
            ).then(self.get_cluster, txt_box, html2)

            bt1.click(
                self.get_predictions, inputs=txt1, outputs=html1, queue=False
            ).then(self.get_cluster, txt1, html2)
            bt2.click(
                self.get_predictions, inputs=txt2, outputs=html1, queue=False
            ).then(self.get_cluster, txt2, html2)
            bt3.click(
                self.get_predictions, inputs=txt3, outputs=html1, queue=False
            ).then(self.get_cluster, txt3, html2)

            im1.select(self.get_predictions, inputs=txt1, outputs=html1)
            im2.select(self.get_predictions, inputs=txt2, outputs=html1)
            im3.select(self.get_predictions, inputs=txt3, outputs=html1)

    def get_cluster(self, user_likes) -> str:
        """Design the clusters assigned to the user"""

        df = self.users_df.copy()  # User associated with all their purchases

        # Initialize DBSCAN with your preferred parameters
        eps = 10  # The maximum distance between two samples to be considered as neighbors)
        min_samples = 5  # The number of samples (or total weight) in a neighborhood for a point to be considered as a core point
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)

        features = self.model.create_embeddings(user_likes)  # createthe embeddings
        new_point_df = pd.DataFrame({"Embeddings": [features]})
        df = pd.concat([df, new_point_df], axis=0, ignore_index=True)

        scaler = StandardScaler()

        df["Embeddings"] = df["Embeddings"].apply(lambda x: scaler.fit_transform(x.T).T)

        df["cluster"] = dbscan.fit_predict(np.concatenate(df.Embeddings))

        point_label = df.iloc[-1].cluster

        if point_label != -1:
            result = generate_inventory(
                df[df.cluster == point_label].sample(
                    n=5, random_state=random.randint(0, 40)
                ),
                "likes",
                ",",
            )

        else:
            # There is too much sparsity to assign a cluster. Therefore we do classic Kmeans clustering

            kmeans = KMeans(n_clusters=10, n_init="auto", random_state=23)

            df = df.iloc[:-1]  # Remove new_point from fitting

            kmeans.fit(np.concatenate(df.Embeddings))

            df["cluster"] = kmeans.labels_

            point_label = kmeans.predict(features)[0]

            result = generate_inventory(
                df[df.cluster == point_label].sample(
                    n=5, random_state=random.randint(0, 40)
                ),
                "likes",
                ",",
            )

        # Create the HTML text as a table
        html_text = "<h2>People Like You Bought This!</h2>"
        html_text += "<table>"
        html_text += "<tr><th>Product</th></tr>"
        for i, product in enumerate(result.keys()):
            if i >= 5:
                break
            html_text += f"<tr><td>{product}</td></tr>"
        html_text += "</table>"

        return html_text

    def get_predictions(self, user_likes, df=None) -> str:
        """Used to get the closest elements in the dataset to the user likes"""

        if df is None:
            df = self.df

        features = self.model.create_embeddings(user_likes)

        # Calculate Euclidean distances and add a new column "Distance" to the DataFrame
        embeddings_array = np.stack(df["Embeddings"].to_numpy())
        # reduced_embeddings_array = pca.transform(embeddings_array)  # Apply PCA to embeddings
        distances = np.linalg.norm(embeddings_array - features, axis=2)
        df["Distance"] = distances

        # Calculate similarity as percentage (inverse of distance)
        max_distance = np.max(distances)
        similarities = 100 - (distances / max_distance) * 100
        df["similarity"] = similarities

        # Sort the DataFrame based on the "Distance" column
        sorted_df = df.sort_values(by="Distance")

        # Drop duplicates based on the "Description" column
        sorted_df = sorted_df.drop_duplicates(subset="Description")

        # Retrieve the top 5 results with the "Description" and "Similarity (%)" columns
        top_10_results = sorted_df.head(10)[["Description", "similarity"]]

        # Create an HTML table to display the results
        html_table = "<h2>Purchases that Look More Alike</h2>"
        html_table += "<table>"
        html_table += "<tr><th>Rank</th><th>Product</th><th>Similarity (%)</th></tr>"
        for i, (_, row) in enumerate(top_10_results.iterrows(), start=1):
            html_table += f"<tr><td>{i}</td><td>{row.Description}</td><td>{row.similarity:.2f}%</td></tr>"
        html_table += "</table>"

        return html_table


if __name__ == "__main__":
    ui = UI()
    ui.create_ui()
    ui.demo.launch(debug=False, server_port=8080)
