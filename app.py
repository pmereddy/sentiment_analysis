import os
import time
import gradio as gr
import requests
import whisper
from openai import OpenAI
import json
import torch
import numpy as np
from transformers import pipeline
import matplotlib.pyplot as plt


# Define the endpoint URL for the deployed sentiment analysis model
MODEL_ENDPOINT = "https://modelservice.ml-e094915e-672.ps-sandb.a465-9q4k.cloudera.site/model"
ACCESS_KEY = os.getenv("MODEL_ACCESS_KEY")

def transcribe_audio(audio, audio_model):
    transcriber = pipeline("automatic-speech-recognition", model=audio_model, device="cpu")
    sr, y = audio

    # Convert to mono if stereo
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    return transcriber({"sampling_rate": sr, "raw": y})["text"]  


def analyze_openai_sentiment(text, senti_model):
    key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=key)
    messages = [
        {"role": "system", "content": "You are a helpful assistant that performs sentiment analysis."},
        {"role": "user", "content": f"Analyze the sentiment of the following text and provide scores as floating points for positive, negative, and neutral sentiment such that the total sum of these is equal to 1 in JSON format. Never give me empty response:\n\n\"{text}\""}
    ]
    response = client.chat.completions.create(
        model=senti_model,
        messages=messages,
        max_tokens=60,
        temperature=0.0
    )
    print(response)

    sentiment = response.choices[0].message.content
    try:
        sentiment_json = json.loads(sentiment)
    except json.JSONDecodeError:
        return "Error: Unable to parse sentiment response", {}
    
    max_key = max(sentiment_json, key=sentiment_json.get)
    label = f"Overall Sentiment: {max_key.upper()} \n"
    label += f"Input Text (1k chars): {text[:1024]}"
    return label, sentiment_json


def analyze_sentiment(text, senti_model):
    input_data = {
        "request": {
            "created_at": "2023-01-11T15:05:45.000Z",  # Adjust if necessary
            "id": "1613190434120949761",  # Ensure this is unique for each request
            "text": text
        }
    }

    response = requests.post(
        f"{MODEL_ENDPOINT}?accessKey={ACCESS_KEY}",
        json=input_data,
        headers={'Content-Type': 'application/json'}
    )

    if response.status_code == 200:
        json_response = response.json()
        if json_response.get("success"):
            temp = json_response["response"]["label"]
            label = f"Overall Sentiment: {temp.upper()} \n"
            label += f"Input Text (1k chars): {text[:1024]}"
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


def process_input(input_text, input_audio, senti_model, audio_model):
    start_time = time.time()
    text = input_text
    if input_audio is not None:
        text = transcribe_audio(input_audio, audio_model)
    
    if senti_model == "gpt-3.5-turbo" or senti_model == "gpt-4":
        sentiment_result = analyze_openai_sentiment(text, senti_model)
    else:
        sentiment_result = analyze_sentiment(text, senti_model)
    
    if isinstance(sentiment_result, dict) and "error" in sentiment_result:
        return "Error: " + sentiment_result["error"], {}
    
    end_time = time.time()
    elapsed_time = (end_time-start_time)
    label, scores = sentiment_result

    new_label = f"Time Taken: {elapsed_time:.2f} secs \n{label}"
    return new_label, scores, create_sentiment_bar_chart(scores)



def create_sentiment_bar_chart(data):
    sentiment_values = {k: round(v * 100) for k, v in data.items()}
    categories = list(sentiment_values.keys())
    values = list(sentiment_values.values())
    colors = {
        "positive": "green",
        "neutral": "grey",
        "negative": "red"
    }
    bar_colors = [colors[category.lower()] for category in categories]
    
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
    gr.Markdown("""
        Perform sentiment analysis on 
        <b><span style="color: orange;">Tweets</span></b>, 
        <b><span style="color: orange;">Customer interaction chat</span></b>, 
        <b><span style="color: orange;">product or service feedback</span></b>, 
        <b><span style="color: green;">audio files</span></b>, 
        <b><span style="color: green;">live audio</span></b> 
        with a choice of 
        <b><span style="color: orange;">sentiment analysis</span></b> model and 
        <b><span style="color: orange;">audio transcription</span></b> model.
    """)

    with gr.Row():
        model_dropdown = gr.Dropdown(
            label="Model", 
            choices=["cardiffnlp/twitter-roberta-base-sentiment-latest", "gpt-3.5-turbo", "gpt-4"], 
            value="cardiffnlp/twitter-roberta-base-sentiment-latest" 
        )
        audio_model_dropdown = gr.Dropdown(
            label="Audio Model", 
            choices=["openai/whisper-base.en", "openai/whisper-medium", "openai/whisper-small"], 
            value="openai/whisper-base.en"  # Default value
        )

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
    submit_button.click(fn=process_input, inputs=[text_input, audio_input, model_dropdown, audio_model_dropdown], outputs=[sentiment_output, scores_output, scores_image])
# Launch the Gradio app
app.launch(server_name="0.0.0.0", server_port=7861, share=True)
