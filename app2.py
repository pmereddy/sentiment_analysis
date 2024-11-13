import gradio as gr
import requests
import whisper
import torch
import numpy as np
from transformers import pipeline
import matplotlib.pyplot as plt

transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en", device="cpu")

# Define the endpoint URL for the deployed sentiment analysis model
MODEL_ENDPOINT = "https://modelservice.ml-e094915e-672.ps-sandb.a465-9q4k.cloudera.site/model"

def transcribe_audio(audio):
    sr, y = audio
    # Convert to mono if stereo
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    return transcriber({"sampling_rate": sr, "raw": y})["text"]  


def analyze_sentiment(text):
    input_data = {
        "request": {
            "created_at": "2023-01-11T15:05:45.000Z",  # Adjust if necessary
            "id": "1613190434120949761",  # Ensure this is unique for each request
            "text": text
        }
    }

    response = requests.post(
        f"{MODEL_ENDPOINT}?accessKey=mjmd1ry4nsjxtxgveld0jgjchlsay2hv",
        json=input_data,
        headers={'Content-Type': 'application/json'}
    )

    if response.status_code == 200:
        json_response = response.json()
        if json_response.get("success"):
            temp = json_response["response"]["label"]
            label = f"Overall Sentiment: {temp.upper()} \n"
            label += f"Input Text (128 chars): {text[:128]}"
            return label , {
                'positive': json_response["response"].get("positive", 0.0),
                'negative': json_response["response"].get("negative", 0.0),
                'neutral': json_response["response"].get("neutral", 0.0)
            }
        else:
            return "Error: Model processing failed.", {}
    else:
        print(f"Request failed with status code: {response.status_code}")
        print(f"Response: {response.text}")
        return {"error": "Model request failed."}

def process_input(input_text, input_audio):
    text = input_text
    if input_audio is not None:
        text = transcribe_audio(input_audio)
    
    sentiment_result = analyze_sentiment(text)
    
    if isinstance(sentiment_result, dict) and "error" in sentiment_result:
        return "Error: " + sentiment_result["error"], {}
    
    label, scores = sentiment_result
    return label, scores, create_sentiment_bar_chart(scores)



def create_sentiment_bar_chart(data):
    sentiment_values = {k: round(v * 100) for k, v in data.items()}
    categories = list(sentiment_values.keys())
    values = list(sentiment_values.values())
    colors = {
        "positive": "green",
        "neutral": "grey",
        "negative": "red"
    }
    bar_colors = [colors[category] for category in categories]
    
    plt.figure(figsize=(6, 4))
    plt.bar(categories, values, color=bar_colors)
    plt.title("Sentiment Analysis Results")
    plt.xlabel("Sentiment")
    plt.ylabel("Percentage (%)")
    
    # Save the plot and return filepath
    chart_path = "/tmp/sentiment_bar_chart.png"
    plt.savefig(chart_path)
    plt.close()
    return chart_path

def clear_fields():
    return "", None, "", {}, None


with gr.Blocks(css="""
    .gradio-container {
        width: 50%;
        margin: 0 auto;
        padding: 20px;
    }
""") as app:
    gr.Markdown("## Real-Time Sentiment Analysis")
    
    text_input = gr.Textbox(label="Enter text message")
    audio_input = gr.Audio(label="Upload audio file", sources=["microphone", "upload"])
    
    
    with gr.Row():
        submit_button = gr.Button("Analyze", variant="secondary")
        clear_button = gr.Button("Clear", variant="stop")
    
    sentiment_output = gr.Textbox(label="Results", interactive=False)
    with gr.Row():
        scores_image = gr.Image(type="filepath", label="Sentiment Bar Chart")
        scores_output = gr.JSON(label="Sentiment Scores")

    clear_button.click(clear_fields, outputs=[text_input, audio_input, sentiment_output, scores_output, scores_image])
    submit_button.click(fn=process_input, inputs=[text_input, audio_input], outputs=[sentiment_output, scores_output, scores_image])
# Launch the Gradio app
app.launch(server_name="0.0.0.0", server_port=7860, share=True)
