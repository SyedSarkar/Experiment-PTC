# Positive Phrase Intervention

A Streamlit application for experimental research that guides participants through a two-phase task encouraging positive emotional expression and sentiment reflection. Features AI-powered response validation, Firebase Firestore integration for data collection, and real-time analytics.

## Features

### Two-Phase Structure
- **Phase 1**: Respond to cue words (Goat, Post, Real, Money, Space) with semantically relevant, positive phrases
- **Phase 2**: Complete sentence prompts ("I am", "He is", "Ali never had") with appropriate responses

### AI-Powered Validation
The app uses multiple AI models for intelligent response validation:

1. **Sentiment Analysis** (`tabularisai/robust-sentiment-analysis`)
   - Detects positive, negative, and neutral sentiment
   - Trained on diverse real-world text for better accuracy

2. **Gibberish Detection** (`madhurjindal/autonlp-Gibberish-Detector-492513457`)
   - Filters out nonsense words and random keystrokes
   - Threshold: 50% confidence for rejection

3. **Semantic Relevance** (`sentence-transformers/all-MiniLM-L6-v2`)
   - Checks if response relates to cue word or sentence context
   - Falls back to keyword matching if sentence_transformers unavailable

### Validation Logic
**Phase 1** (Cue Words):
1. Not already used
2. Not gibberish
3. **Semantically relevant to cue word**
4. Not negative sentiment
→ Accept if all pass

**Phase 2** (Sentences):
1. Not already used
2. Not gibberish
3. Not negative sentiment
4. **Positive sentiment** → Accept (skip relevance check)
5. **Neutral sentiment** → Must be semantically relevant to sentence

### Data Storage
- **Primary**: Firebase Firestore (real-time, non-blocking writes)
- **Export**: `export_data.py` script exports all responses to CSV
- **Analysis**: Includes response time, sentiment confidence, acceptance status

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

Required packages:
- streamlit
- pandas, matplotlib
- transformers, torch, torchvision
- sentence-transformers (optional, for AI semantic relevance)
- nltk
- firebase-admin

### 2. Configure Firebase
Create `.streamlit/secrets.toml`:
```toml
FIREBASE_CREDENTIALS_JSON = '{"type":"service_account",...}'
```

Get your Firebase service account key from Firebase Console → Project Settings → Service Accounts → Generate New Private Key.

### 3. Run the App
```bash
streamlit run app.py
```

### 4. Export Data
After data collection, export to CSV:
```bash
python export_data.py
```

Outputs `all_responses.csv` with columns:
- timestamp, user, specific_id
- phase, cue, sentence, response
- sentiment, confidence, score, response_time_sec, accepted

## Data Analysis

The app includes an analytics dashboard (Phase 3) with:
- AI Confidence Over Time chart
- Cumulative Score tracking
- Individual participant data table
- CSV download for each participant

## Performance Optimizations

- **Async Firestore writes** - No UI blocking during data persistence
- **Cached model loading** - Models load once and stay in memory
- **Session state management** - Minimal reruns, instant feedback
- **CSS fade animations** - Feedback visible for 3 seconds without slowing process

## File Structure

```
.
├── app.py                    # Main Streamlit application
├── export_data.py            # Export Firestore data to CSV
├── requirements.txt          # Python dependencies
├── data/
│   ├── cue_words.txt         # Phase 1 cue words (one per line)
│   └── sentences.txt         # Phase 2 sentence prompts
├── .streamlit/
│   └── secrets.toml          # Firebase credentials (not in git)
└── results/                  # Local CSV backups (optional)
```

## Customization

### Change Cue Words
Edit `data/cue_words.txt` - one word per line

### Change Sentences
Edit `data/sentences.txt` - one sentence per line

### Adjust Validation Thresholds
In `app.py`:
- Gibberish threshold: `gibberish_result['score'] > 0.5`
- Semantic relevance (Phase 1): `threshold=0.30`
- Semantic relevance (Phase 2): `threshold=0.25`

## Troubleshooting

### sentence_transformers not found
App works without it, but uses keyword matching instead of AI semantic similarity.
To install: `pip install sentence-transformers`

### Torchvision warnings
These are harmless warnings from transformers library for optional image models.
Install torchvision to eliminate: `pip install torchvision`

### ModuleNotFoundError: huggingface-hub
Update huggingface-hub: `pip install huggingface-hub==1.5.0`

## License

Research use only.
