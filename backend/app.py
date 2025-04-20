from flask import Flask, request, render_template
import os
import joblib
from utils import preprocess_text

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "../frontend/templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "../frontend/static")
)

# Load the model
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
try:
    model = joblib.load(model_path)
except Exception as e:
    print("❌ Failed to load model.pkl:", e)
    model = None

# Emotion mapping
emotion_keywords = {
    "positive": ["joy", "happy","happiness","excited", "delight", "elation", "satisfaction", "cheerfulness", "optimism", "contentment", "glee", "radiance", "bliss", "gratitude", "enthusiasm", "thrill", "excitement", "serenity", "hopefulness", "euphoria", "zest", "pleasure", "triumph", "peace", "fulfillment", "confidence", "love"],
    "negative": ["anger","hate", "suffer","frustration", "hatred", "annoyance", "irritation", "hostility", "rage", "fury", "disgust", "disappointment", "bitterness", "jealousy", "resentment", "guilt", "shame", "envy", "grief", "agony", "dread", "despair", "fear", "panic", "worry", "loathing", "hurt"],
    "neutral": ["calm", "see","call","indifference", "objectivity", "plainness", "normalcy", "regularity", "simplicity", "routine", "equilibrium", "balance", "mildness", "temperance", "fairness", "reservation", "restraint", "peacefulness", "modesty", "clarity", "detachment", "stillness", "evenness", "dispassion", "uniformity", "typicality", "mutedness"]
}

@app.route("/", methods=["GET", "POST"])
def home():
    sentiment = ""
    if request.method == "POST":
        text = request.form["text"]
        if not text.strip():
            sentiment = "Please enter text."
        else:
            processed = preprocess_text(text)
            ml_pred = model.predict([processed])[0]
            final_pred = ml_pred
            emotion_list = emotion_keywords.get(final_pred.lower(), [])
            processed_words = preprocess_text(text).split()
            emotion_match = next((word.capitalize() for word in emotion_list if word in processed_words), None)
            emotion = emotion_match if emotion_match else ""
            sentiment = f"Sentiment: {final_pred.capitalize()}" + (f" ({emotion})" if emotion else "")
    return render_template("index.html", sentiment=sentiment)

if __name__ == "__main__":
    app.run()
