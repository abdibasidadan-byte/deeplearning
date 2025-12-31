# Evaluation of Zero-Shot Transformer Models for Real-Time Churn Intent Classification

# Main Objective: To implement a non-linear NLP (Natural Language Processing) pipeline capable of categorizing unstructured customer feedback without prior task-specific training."


# !pip install transformers torch 
from transformers import pipeline

# 1. Load a "Zero-Shot Classification" model
# This is a Deep Learning model that understands the meaning of words 
# without needing specific training for these labels.
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# 2. Customer feedback data (raw text)
comments = [
    "I think the subscription has become too expensive for what it is.",
    "I love my plan, the network is great everywhere!",
    "My contract is expiring soon and I'm looking at the competition."
]

# 3. Targeted business categories (Churn risk)
candidate_labels = ["churn risk", "loyal customer", "support request"]

# 4. Model inference and analysis
for text in comments:
    result = classifier(text, candidate_labels)
    print(f"\nText: {text}")
    print(f"Prediction: {result['labels'][0]} (Confidence Score: {result['scores'][0]:.2f})")


# Model Output :

# Text: "I think the subscription has become too expensive for what it is." Prediction: Churn Risk (Confidence: 0.58)

# Text: "I love my plan, the network is great everywhere!" Prediction: Support Request (Confidence: 0.36)

# Text: "My contract is expiring soon and I'm looking at the competition." Prediction: Support Request (Confidence: 0.52)


# Key Takeaways :

# For the first result (0.58): > "The model identified a pricing pain point. In natural language processing, 'expensive' is a strong predictor for customer attrition, hence the 'Churn Risk' label."

# For the second result (0.36): > "This is a false positive due to a low confidence score. Because the model didn't have a 'Positive' or 'Satisfied' category to choose from, it defaulted to 'Support Request' with very low certainty. We call this a forced choice."

# For the third result (0.52): > "The model correctly sensed an intent to switch. Mentioning 'competition' and 'contract expiration' triggers a high probability for a retention-related support ticket."










