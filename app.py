import pickle
import regex as re
import gradio as gr

with open('model.pkl','rb') as f:
    package = pickle.load(f)
    
model = package['model']
vectorizer = package['vectorizer']
slang_dict = package['slang_dict']

# Preprocessing Functions

def clean_text(text):
    if not text or str(text).strip() == "":
        return ""

    text = str(text).lower()

    # Collapse 3+ repeated chars → 2 (e.g. "bangetttt" → "bangett")
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # Hardcoded emoji Unicode ranges — covers all common emoji categories
    emoji_ranges = (
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA70-\U0001FAFF"
        "\U00002600-\U000026FF"
        "\U00002700-\U000027BF"
    )

    # Remove anything that's not a word char, whitespace, punctuation, or emoji
    text = re.sub(r"[^" + emoji_ranges + r"\w\s.!?]", " ", text)

    return text.strip()

def normalize_slang(text):
    if not text:
        return ""
    return " ".join([slang_dict.get(w, w) for w in text.split()])
    
# Inference Function

def predict_sentiment(text):
    if not text or text.strip() == "":
        return "Please enter some text to analyze."
    
    cleaned = clean_text(text)
    normalized = normalize_slang(cleaned)
    
    features = vectorizer.transform([normalized])
    prediction = model.predict_proba(features)[0]
    
    label_map={
        0: "negatif",
        1: "positif"
    }
    
    return {label_map[cls]: float(prob) for cls, prob in zip(model.classes_, prediction)}

# Gradio Interface
    
iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        lines=3,
        placeholder="Masukkan komentar TikTok Anda di sini...",
        label="Komentar TikTok",),
    outputs=gr.Label(num_top_classes=2, label="Sentimen Komentar"),
    title="Analisis Sentimen Pertalite",
    description=(
        "Masukkan komentar TikTok berbahasa Indonesia seputar kebijakan Pertalite "
        "atau Menteri Bahlil Lahadalia. Model akan mengklasifikasikan sentimen "
        "sebagai **positif** atau **negatif**."
    ),
    examples=[
        ["Kebijakan Pertalite sangat membantu masyarakat!"],
        ["Menteri Bahlil tidak kompeten, kebijakan Pertalite malah merugikan!"],
        ["gak ngerti knp harga bbm naik terus padahal katanya subsidi"],
    ],
    allow_flagging="manual",
    flagging_dir="flagged_data",
)
if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0",server_port=7860)