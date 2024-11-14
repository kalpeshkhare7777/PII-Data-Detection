import streamlit as st
from presidio_analyzer import AnalyzerEngine
import spacy
import stanza
import pandas as pd
from transformers import pipeline

# Initialize the Presidio Analyzer
analyzer = AnalyzerEngine()

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load Hugging Face transformer NER model
hf_ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", tokenizer="dbmdz/bert-large-cased-finetuned-conll03-english")

# Initialize Stanza (Stanford NLP)
stanza.download('en')  # Download the English model if not already done
nlp_stanza = stanza.Pipeline('en', processors='tokenize,ner')

# Function to detect PII from text using Presidio
def detect_pii_presidio(text):
    results = analyzer.analyze(text=text, language="en")
    pii_entities = {}
    for result in results:
        entity_type = result.entity_type
        if entity_type not in pii_entities:
            pii_entities[entity_type] = []
        pii_entities[entity_type].append(text[result.start:result.end])
    return pii_entities

# Function to detect PII from text using spaCy
def detect_pii_spacy(text):
    doc = nlp(text)
    pii_entities = {
        "PERSON": [],
        "ORG": [],
        "GPE": [],  # Geopolitical entities (countries, cities, etc.)
        "DATE": [],
        "MONEY": [],
        "PHONE": [],
        "EMAIL": [],
    }
    
    # Check for named entities using spaCy
    for ent in doc.ents:
        if ent.label_ in pii_entities:
            pii_entities[ent.label_].append(ent.text)
    
    return pii_entities

# Function to detect PII using Hugging Face transformer model
def detect_pii_huggingface(text):
    entities = hf_ner(text)
    pii_entities = {
        "PERSON": [],
        "ORG": [],
        "GPE": [],
        "DATE": [],
    }

    for entity in entities:
        entity_type = entity['entity']
        if entity_type == "B-PER" or entity_type == "I-PER":
            pii_entities["PERSON"].append(entity['word'])
        elif entity_type == "B-ORG" or entity_type == "I-ORG":
            pii_entities["ORG"].append(entity['word'])
        elif entity_type == "B-LOC" or entity_type == "I-LOC":
            pii_entities["GPE"].append(entity['word'])
        elif entity_type == "B-DATE" or entity_type == "I-DATE":
            pii_entities["DATE"].append(entity['word'])

    return pii_entities

# Function to detect PII using Stanza (Stanford NLP)
def detect_pii_stanza(text):
    doc = nlp_stanza(text)
    pii_entities = {
        "PERSON": [],
        "ORG": [],
        "GPE": [],
        "DATE": [],
    }

    for ent in doc.ents:
        if ent.type in pii_entities:
            pii_entities[ent.type].append(ent.text)
    
    return pii_entities

# Function to compare the results from all models and create a table
def compare_results(presidio_results, spacy_results, hf_results, stanza_results):
    # Prepare the data for a table
    all_categories = set(presidio_results.keys()).union(set(spacy_results.keys())).union(set(hf_results.keys())).union(set(stanza_results.keys()))
    comparison_data = []

    for category in all_categories:
        presidio_entities = presidio_results.get(category, [])
        spacy_entities = spacy_results.get(category, [])
        hf_entities = hf_results.get(category, [])
        stanza_entities = stanza_results.get(category, [])
        
        # Create a row for each category comparison
        comparison_data.append({
            "Category": category,
            "Presidio": ', '.join(presidio_entities) if presidio_entities else 'None',
            "spaCy": ', '.join(spacy_entities) if spacy_entities else 'None',
            "HuggingFace": ', '.join(hf_entities) if hf_entities else 'None',
            "Stanza": ', '.join(stanza_entities) if stanza_entities else 'None',
        })

    # Convert the data to a DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    return comparison_df

# Streamlit UI
def main():
    st.title("PII Detection Comparison App \n Presidio, spaCy, Hugging Face, and Stanza")

    st.write("Enter a text below to detect and compare Personally Identifiable Information (PII) between Presidio, spaCy, Hugging Face, and Stanza:")

    # Text input for the user
    text = st.text_area("Text", height=200)
    
    if st.button("Detect and Compare PII"):
        if text:
            # Detect PII using all four models
            presidio_entities = detect_pii_presidio(text)
            spacy_entities = detect_pii_spacy(text)
            hf_entities = detect_pii_huggingface(text)
            stanza_entities = detect_pii_stanza(text)
            
            st.subheader("Detected PII Entities Comparison")

            # Compare the results and create a DataFrame for display
            comparison_df = compare_results(presidio_entities, spacy_entities, hf_entities, stanza_entities)
            
            if not comparison_df.empty:
                st.table(comparison_df)  # Display the comparison as a table
            else:
                st.write("No PII detected in the input text.")
        else:
            st.warning("Please enter some text for detection.")

# Run the app
if __name__ == "__main__":
    main()
