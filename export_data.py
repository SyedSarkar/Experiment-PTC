"""
Export all Firestore responses to CSV for analysis.
Run this after your experiment to get all data in one file.
"""
import json
import codecs
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
import re

# Initialize Firebase (use same credentials as your app)
def init_firebase():
    if not firebase_admin._apps:
        # Read TOML file and parse the JSON credential
        with open('.streamlit/secrets.toml', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract the JSON string from FIREBASE_CREDENTIALS_JSON = "..."
        # Handle both single-line and multi-line formats
        match = re.search(r'FIREBASE_CREDENTIALS_JSON\s*=\s*["\'](.+)["\']', content, re.DOTALL)
        if not match:
            raise ValueError("Could not find FIREBASE_CREDENTIALS_JSON in secrets.toml")
        
        creds_str = match.group(1)
        # Decode escape sequences (\n -> newline, etc.)
        creds_str = codecs.decode(creds_str, 'unicode_escape')
        creds_json = json.loads(creds_str)
 
        cred = credentials.Certificate(creds_json)
        firebase_admin.initialize_app(cred)
    return firestore.client()
 
def export_to_csv(output_file="all_responses.csv"):
    """Export all responses from Firestore to a single CSV file."""
    db = init_firebase()
 
    # Get all documents from 'responses' collection
    docs = db.collection("responses").stream()
 
    data = []
    for doc in docs:
        row = doc.to_dict()
        row['firestore_id'] = doc.id  # Keep track of document ID
        data.append(row)
 
    if not data:
        print("No data found in Firestore!")
        return
 
    # Convert to DataFrame
    df = pd.DataFrame(data)
 
    # Reorder columns for better readability
    columns_order = [
        'timestamp', 'user', 'specific_id', 'phase', 'cue', 'sentence',
        'response', 'sentiment', 'confidence', 'score', 
        'response_time_sec', 'accepted', 'firestore_id'
    ]
 
    # Only include columns that exist
    available_cols = [c for c in columns_order if c in df.columns]
    df = df[available_cols]
 
    # Sort by user and timestamp
    df = df.sort_values(['user', 'timestamp'])
 
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"✅ Exported {len(df)} responses to {output_file}")
 
    # Print summary
    print("\n📊 Summary:")
    print(f"   Total participants: {df['user'].nunique()}")
    print(f"   Phase 1 responses: {len(df[df['phase'] == 1])}")
    print(f"   Phase 2 responses: {len(df[df['phase'] == 2])}")
    print(f"   Accepted responses: {len(df[df['accepted'] == True])}")
 
    return df
 
if __name__ == "__main__":
    df = export_to_csv()