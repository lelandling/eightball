
import time
from metaphor_python import Metaphor
from bs4 import BeautifulSoup
import requests
import os
import sys
from transformers import BarkModel
import torch
from IPython.display import Audio
from dotenv import load_dotenv
import keyboard
import openai
import string

device = "cuda" if torch.cuda.is_available() else "cpu"
model = BarkModel.from_pretrained("suno/bark-small", torch_dtype=torch.float16).to(device)
model.enable_cpu_offload()

load_dotenv(r"C:\Users\lling\Documents\metaphortechnical\keys.env")
openai_org_key = os.getenv("openai_org_key")
metaphor_key = os.getenv("metaphor_key")
openai_api_key = os.getenv("openaiapi_key")


print(str(type(openai_org_key)) + " " + str(type(openai_api_key)))

openai.organization = openai_org_key
openai.api_key = openai_api_key
openai.Model.list()
# metaphorkey = os.getenv("metaphorkey")

metaphor = Metaphor("f1ab1016-1795-45ae-9e04-e35caf0e5d48")

def scrapeURLs(question):
    # question = "What are the sentiments of climate change in America?"
    searchresults = metaphor.search(question, use_autoprompt=True)
    return searchresults

# Define the URLs of the websites you want to scrape\
def scraper(metaphorresults):
    scraped_content = []
    for i in range(10):
        # Send an HTTP GET request to the URL
        response = requests.get(metaphorresults.results[i].url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the HTML content of the page
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract the data you need from the HTML using BeautifulSoup methods
            contents = soup.find_all('body')
            for content in contents:
                scraped_content.append(content.text.replace("\t", "").replace("\r", "").replace("\n", ""))
            
        else:
            None
            #print(f"Failed to fetch data from {metaphorresults.results[i].url}")

    return scraped_content

# Print the summary
def condense(scraped_content):
    summaries = []
    for content in scraped_content:
        if len(content) < 10000:
            summaries.append(meeting_minutes(content[:10000]))
            time.sleep(30) #need line so that openai max token/min isnt crossed
    return summaries

def answerq(question, summaries):
    summary = summarize(question + "," +str(summaries))

    return summary


def meeting_minutes(transcription):
    abstract_summary = abstract_summary_extraction(transcription)
    key_points = key_points_extraction(transcription)
    sentiment = sentiment_analysis(transcription)
    return {
        'abstract_summary': abstract_summary,
        'key_points': key_points,
        'sentiment': sentiment
    }

def abstract_summary_extraction(transcription):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are a highly skilled AI trained in language comprehension and summarization. I would like you to read the following text and summarize it into a concise abstract paragraph. Aim to retain the most important points, providing a coherent and readable summary that could help a person understand the main points of the discussion without needing to read the entire text. Please avoid unnecessary details or tangential points."
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )
    return response['choices'][0]['message']['content']

def key_points_extraction(transcription):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are a proficient AI with a specialty in distilling information into key points. Based on the following text, identify and list the main points that were discussed or brought up. These should be the most important ideas, findings, or topics that are crucial to the essence of the discussion. Your goal is to provide a list that someone could read to quickly understand what was talked about."
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )
    return response['choices'][0]['message']['content']

def sentiment_analysis(transcription):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "As an AI with expertise in language and emotion analysis, your task is to analyze the sentiment of the following text. Please consider the overall tone of the discussion, the emotion conveyed by the language used, and the context in which words and phrases are used. Indicate whether the sentiment is generally positive, negative, or neutral, and provide brief explanations for your analysis where possible."
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )
    return response['choices'][0]['message']['content']

def summarize(transcription):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "As an AI with expertise in language and emotion analysis, answer the question using the list given."
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )
    return response['choices'][0]['message']['content']

def followup(q1, summary):
    check = input("\n Any further questions or follow up questions? ")

    while ('yes' not in check.lower()) and ('no' not in check.lower()):
        print("Uncertain")
        check = input("\n Any further questions or follow up questions? ")

    
    if 'yes' in check.lower():
        q2 = input("\n What question do you have?")
    elif 'no' in check.lower():
        return

    conversation = [
        {"role": "system", "content": "You are a helpful AI assitant that summarizes and explains content."},
        {"role": "user", "content": q1},
        {"role": "assistant", "content": "summary"},
        {"role": "user", "content": q2}  # This is the follow-up question.
    ]

    # Make API Call
    response = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0,
        messages=conversation,
    )

    return response['choices'][0]['message']['content']

def getanswer(question):
    searchresults = scrapeURLs(question)
    scraped_content = scraper(searchresults)
    summaries = condense(scraped_content)
    answer = answerq(question, summaries)
    return answer

def main():
    print("Press 'Escape' key to exit...")
    while True:
        question = input("Oh, what can the all powerful eightball answer for you? ")

        if keyboard.is_pressed('esc'):
            print("Escape key is pressed. Exiting...")
            break

        answer = getanswer(question)
        print(answer)
        followupq = followup(question, answer)

        if followupq is not None:
            print(followupq)


if __name__ == "__main__":
    # print(":D " + openai_org_key + " " + openai_api_key)
    main()