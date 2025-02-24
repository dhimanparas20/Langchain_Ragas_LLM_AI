# uses ragas to evaluate the dataset

import os
from dotenv import load_dotenv
from ragas import evaluate
from datasets import Dataset
import pandas as pd
import json
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
os.system("clear")

# Load environment variables from .env file
load_dotenv()

# Sample dataset
sample_dataset = {
    "question": [
      "What is the current capital of Himachal Pradesh?",
      "What was the summer capital of British India?",
      "Which temple was Shimla most popular for during the 19th century?",
      "Who constructed the first British summer home in Shimla?",
      "In what year was the Kalka-Shimla railway line constructed?"
    ],
    
    "contexts": [
      [
        "The former summer capital of British India, and the present capital of Himachal Pradesh, Shimla has been blessed with all the natural bounties which one can think of."
      ],
      [
        "The former summer capital of British India, and the present capital of Himachal Pradesh, Shimla has been blessed with all the natural bounties which one can think of."
      ],
      [
        "The Shimla back to the 19th century when it was founded by the British in the year 1819 after the Gorkha war. During that period, it was most popular for the temple of Hindu Goddess Shyamala Devi."
      ],
      [
        "In 1822, the first British summer home was constructed by Scottish civil servant Charles Pratt Kennedy."
      ],
      [
        "A remarkable event took place in the history of Shimla when the Kalka-Shimla railway line was constructed in the year 1906 that significantly added to its quick accessibility and it gained immense popularity."
      ]
    ],
    "answer": [
        "The current capital of Himachal Pradesh is Shimla.", 
        "The summer capital of British India was Shimla.", 
        "Shimla was most popular for the temple of Hindu Goddess Shyamala Devi during the 19th century.", 
        "The first British summer home in Shimla was constructed by Scottish civil servant Charles Pratt Kennedy in 1822.", 
        "The Kalka-Shimla railway line was constructed in the year 1906."
    ],
    "ground_truth": [
      "Shimla",
      "Shimla",
      "Hindu Goddess Shyamala Devi",
      "Charles Pratt Kennedy",
      "1906"
    ]
}


# Load the JSON test set file
with open('output.json', 'r') as file:
    test_set = json.load(file)

# Convert sample dataset to Dataset object
dataset = Dataset.from_dict(test_set)

# Evaluate the dataset using Ragas metrics
result = evaluate(
    dataset,
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
    ]
)

print(result)

# Optionally convert the results into a pandas DataFrame for further analysis
df = result.to_pandas()
df.head()


# # Save results to a CSV file if needed
# df.to_csv("evaluation_results.csv", index=False)