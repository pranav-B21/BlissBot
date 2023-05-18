import streamlit as st
from streamlit_chat import message
import random
import json
import torch
from nltk_utils import bag_of_words, tokenize
from model import NeuralNet

# Chatbot API
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Aura"

# Chat history
history = []

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    else:
        return "I do not understand..."

# Web application
def show_home_page():
    # Add content for the home page
    st.image("logo.png")
    st.markdown("Welcome to Aura, an intelligent chatbot designed to assist you! Aura is a powerful tool that can provide helpful information, answer your questions, and engage in conversation. Whether you need guidance, support, or simply want to chat, Aura is here for you. By leveraging natural language processing and machine learning techniques, Aura can understand your messages and provide relevant and meaningful responses. Just type your message in the text area and click Send to start a conversation. Aura's goal is to make your life easier and more enjoyable by offering a personalized and interactive experience. Give it a try and let Aura assist you with its knowledge and conversational abilities.")

def show_chat_page():
    st.markdown("# Chat Page")

    # Display chatbot messages
    for sender, msg in history:
        if sender == "You":
            message(msg, is_user=True, label='User')
        else:
            message(msg)

    # User input at the bottom
    user_input = st.text_area("Your message", height=50)
    if st.button("Send"):
        on_enter_pressed(user_input)

def on_enter_pressed(msg):
    if not msg:
        return

    response = get_response(msg)
    history.append((bot_name, response))
    message(response)  # Add this line to display the bot's response


def show_assesment_page():
    st.markdown("# Self-Assessment and Screening")

    st.write("Please answer the following questions to assess your mental well-being.")

    # Questions and response options
    questions = [
        "How often have you been feeling down, depressed, or hopeless?",
        "Do you find little interest or pleasure in doing things?",
        "Have you been feeling nervous, anxious, or on edge?",
        "Have you been feeling irritable or easily angered?",
        "Do you have difficulty sleeping or experience changes in sleep patterns?",
        "Have you been experiencing persistent feelings of worry or fear?",
        "Have you been having difficulty concentrating or making decisions?",
        "Do you often feel tired or have a lack of energy?",
        "Have you been experiencing physical symptoms such as headaches or stomachaches?",
        "Have you had thoughts of self-harm or suicide?",
    ]

    response_options = [
        "Not at all",
        "Several days",
        "More than half the days",
        "Nearly every day"
    ]

    # Display questions and collect responses
    responses = []
    for i, question in enumerate(questions):
        st.markdown(f"**Q{i+1}:** {question}")
        response = st.selectbox(f"Select your response##{i}", response_options)
        responses.append(response)

    # Submit button to get results
    if st.button("Submit"):
        show_assessment_results(responses)

def show_assessment_results(responses):
    # Calculate and display assessment results based on responses
    # Add your logic here to interpret the responses and generate results

    total_score = 0
    for response in responses:
        if response == "Several days":
            total_score += 1
        elif response == "More than half the days":
            total_score += 2
        elif response == "Nearly every day":
            total_score += 3

    # Interpret the total score and categorize it on a scale
    if total_score <= 5:
        result_text = "Your mental well-being appears to be in a good range."
    elif total_score <= 10:
        result_text = "Your mental well-being suggests some areas of concern. Consider seeking support if needed."
    else:
        result_text = "Your mental well-being indicates a significant level of distress. It is advisable to seek professional help."

    st.markdown("# Assessment Results")
    st.write("Based on your responses, here are your assessment results:")
    st.write(f"Total Score: {total_score}")
    st.write(result_text)




# Set up the sidebar navigation
sidebar_options = {
    "Home": show_home_page,
    "Chat": show_chat_page,
    "Assesment": show_assesment_page
}

# Set up the page layout
def page_layout():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("", list(sidebar_options.keys()))
    sidebar_options[page]()

# Run the Streamlit app
if __name__ == '__main__':
    page_layout()
