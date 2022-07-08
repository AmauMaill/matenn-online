"""
Source: https://towardsdatascience.com/how-to-deploy-a-machine-learning-ui-on-heroku-in-5-steps-b8cd3c9208e6
"""

from fastai.vision.all import load_learner
import gradio as gr

def predict(inp):
    model = load_learner("./export.pkl")
    labels = model.dls.vocab
    prediction = model.predict(inp)
    confidences = {labels[i]: float(prediction[2][i]) for i in range(2)}
    return confidences

gr.Interface(
    fn=predict,
    inputs=gr.inputs.Image(type="filepath"),
    outputs=gr.outputs.Label(num_top_classes=2),
    examples=["./data/kanelbullar.jpg", "./data/semlor.jpg"]
).launch()