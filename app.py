import os
import re
import sqlite3
from functools import wraps
from pathlib import Path

from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

import torch
from PIL import Image
from torchvision import models, transforms
import google.generativeai as genai
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "resnet50_medicinal.pth"
DB_PATH = BASE_DIR / "users.db"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

load_dotenv(BASE_DIR / ".env")

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "change-me-please")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def login_required(view_func):
    @wraps(view_func)
    def wrapped_view(**kwargs):
        if "user" not in session:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))
        return view_func(**kwargs)

    return wrapped_view


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, len(idx_to_class))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    transforms_pipeline = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return model, idx_to_class, transforms_pipeline


model, idx_to_class, transforms_pipeline = load_model()
gemini_model = None


def get_gemini_model():
    global gemini_model
    if gemini_model or not GEMINI_API_KEY:
        return gemini_model
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel(GEMINI_MODEL)
    return gemini_model


@app.route("/")
def landing():
    return render_template("landing.html", user=session.get("user"), show_nav=False)


@app.route("/register", methods=["GET", "POST"])
def register():
    init_db()
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if not username or not password:
            flash("Username and password are required.", "danger")
            return redirect(url_for("register"))

        conn = get_db()
        try:
            conn.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username, generate_password_hash(password)),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            flash("Username already exists.", "warning")
            return redirect(url_for("register"))
        finally:
            conn.close()

        flash("Registration successful. Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    init_db()
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        conn = get_db()
        user = conn.execute(
            "SELECT id, username, password_hash FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        conn.close()

        if user and check_password_hash(user["password_hash"], password):
            session["user"] = {"id": user["id"], "username": user["username"]}
            flash(f"Welcome, {user['username']}!", "success")
            return redirect(url_for("predict"))

        flash("Invalid username or password.", "danger")
        return redirect(url_for("login"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


def run_inference(image_file):
    image = Image.open(image_file).convert("RGB")
    tensor = transforms_pipeline(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        top_prob, top_idx = torch.max(probs, dim=0)
    return idx_to_class[int(top_idx)], float(top_prob)


def parse_gemini_response(text: str) -> dict:
    """Parse the Gemini response into structured sections. Falls back to raw HTML/text."""
    def _format(val: str) -> str:
        """Remove stray asterisks and convert **bold** to <strong> tags."""
        val = val.strip()
        # convert bold markers
        val = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", val)
        # remove any leftover single/double stars
        val = val.replace("*", "").strip()
        return val

    sections = {
        "overview": "",
        "medicinal_properties": [],
        "health_benefits": [],
        "therapeutic_applications": [],
        "precautions": [],
        "raw_html": text,
        "plain_text": text,
        "has_structured": False,
    }

    current_section = None

    # Strip simple Markdown/HTML code fences if present
    code_fences = ["```html", "```", "<html"]
    if any(text.strip().lower().startswith(cf) for cf in code_fences):
        text = text.strip("` \n")

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Handle headers like "OVERVIEW:", "Health Benefits:" (case-insensitive)
        if line.endswith(":"):
            section_name = line[:-1].strip().lower().replace(" ", "_")
            if section_name in sections:
                current_section = section_name
            continue

        # Handle bullet points (leading "-", "•", or "*")
        if line.startswith(("-", "•", "*")):
            line = line.lstrip("-•* ").strip()
            line = _format(line)
            if current_section and current_section in sections and isinstance(
                sections[current_section], list
            ):
                sections[current_section].append(line)
            continue

        # Regular text belongs to current section if it exists
        if current_section and current_section in sections:
            if isinstance(sections[current_section], list):
                sections[current_section].append(_format(line))
            elif isinstance(sections[current_section], str):
                sections[current_section] = (
                    sections[current_section] + " " + _format(line)
                ).strip()

    # Clean empty values
    parsed_any = any(
        [
            sections["overview"],
            sections["medicinal_properties"],
            sections["health_benefits"],
            sections["therapeutic_applications"],
            sections["precautions"],
        ]
    )
    sections["has_structured"] = parsed_any
    if not parsed_any:
        # Return only raw_html to render as-is
        return {
            "raw_html": text,
            "plain_text": text,
            "has_structured": False,
        }

    return sections

def describe_plant(plant_name: str) -> dict:
    """Generate a detailed description of a medicinal plant using Gemini AI."""
    model_instance = get_gemini_model()
    if not model_instance:
        return {'error': 'Gemini not configured. Please set GEMINI_API_KEY in your environment variables.'}
    
    prompt = """You are a professional botanist with expertise in medicinal plants. 
    Provide a comprehensive description of the plant '{plant_name}' with the following sections:
    
    OVERVIEW:
    A brief introduction to the plant (2-3 sentences).
    
    MEDICINAL PROPERTIES:
    - List 3-5 key medicinal properties with brief explanations.
    
    HEALTH BENEFITS:
    - 4-6 specific health conditions it can help with.
    - Include traditional uses if applicable.
    
    THERAPEUTIC APPLICATIONS:
    - How it's typically used (tea, extract, poultice, etc.)
    - Recommended preparations and dosages if known.
    
    PRECAUTIONS:
    - Any known side effects or contraindications.
    - Special considerations (pregnancy, interactions with medications, etc.)
    
    Format the response with clear section headers in ALL CAPS followed by a colon.
    Use bullet points for lists. Keep the total response between 200-300 words.
    """.format(plant_name=plant_name)
    
    try:
        response = model_instance.generate_content(prompt)

        def _extract_text(resp):
            # Prefer direct text
            if getattr(resp, "text", None):
                return resp.text
            # Try candidates -> parts -> text (Gemini commonly uses this structure)
            candidates = getattr(resp, "candidates", []) or []
            for cand in candidates:
                parts = getattr(getattr(cand, "content", None), "parts", []) or []
                for part in parts:
                    if getattr(part, "text", None):
                        return part.text
                # Fallback: str(content)
                content_obj = getattr(cand, "content", None)
                if content_obj:
                    return str(content_obj)
            # Last resort: string conversion
            return str(resp)

        response_text = _extract_text(response) or ""

        sections = parse_gemini_response(response_text)
        # Always keep raw/flat text for fallback rendering
        sections["raw_html"] = response_text
        sections["plain_text"] = response_text
        sections["error"] = None
        return sections
    except Exception as e:
        error_msg = f"Error generating plant description: {str(e)}"
        print(error_msg)
        return {
            "error": error_msg,
            "raw_html": "",
            "plain_text": "",
            "has_structured": False,
        }


def generate_health_chat(message: str, plant_name: str | None = None) -> str:
    """Generate a concise health Q&A reply with optional plant context."""
    model_instance = get_gemini_model()
    if not model_instance:
        return "Gemini not configured. Please set GEMINI_API_KEY."

    plant_context = (
        f" Relevant plant context: {plant_name}. Use it if helpful."
        if plant_name else ""
    )
    prompt = (
        "You are a certified health assistant. Provide clear, safe guidance, "
        "and recommend consulting a professional for serious issues. "
        f"{plant_context} Keep replies under 120 words, concise and actionable."
        f"\nUser question: {message}"
    )
    try:
        response = model_instance.generate_content(prompt)
        return getattr(response, "text", "") or ""
    except Exception:
        return "Sorry, I couldn't answer right now. Please try again."


def generate_remedies(symptoms: str, plant_name: str | None = None) -> str:
    """Provide simple home/Ayurvedic remedies based on symptoms and optional plant."""
    model_instance = get_gemini_model()
    if not model_instance:
        return "Gemini not configured. Please set GEMINI_API_KEY."

    plant_text = (
        f"Prioritize remedies that safely use the plant '{plant_name}' when appropriate."
        if plant_name else
        "Use common safe household or Ayurvedic remedies."
    )
    prompt = (
        "You are an herbal/Ayurvedic assistant. Suggest 2-4 safe, practical home remedies. "
        "Include preparation steps, dosage ranges if relevant, and caution notes. "
        "Keep total under 150 words. "
        f"{plant_text} Symptoms: {symptoms}"
    )
    try:
        response = model_instance.generate_content(prompt)
        return getattr(response, "text", "") or ""
    except Exception:
        return "Remedy suggestions are unavailable right now."


def format_basic_markdown(text: str) -> str:
    """Lightweight formatter for inline bold and newlines."""
    if not text:
        return ""
    html = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", text)
    html = html.replace("\n\n", "<br><br>").replace("\n", "<br>")
    return html


@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    gemini_enabled = bool(GEMINI_API_KEY)

    if request.method == "GET":
        return render_template(
            "predict.html",
            user=session.get("user"),
            classes=sorted(idx_to_class.values()),
            prediction=None,
            confidence=None,
            gemini_enabled=gemini_enabled,
        )

    # Handle POST request
    if "image" not in request.files:
        flash("No file part in the request.", "danger")
        return redirect(request.url)

    file = request.files["image"]
    if file.filename == "":
        flash("No file selected.", "warning")
        return redirect(request.url)

    if not allowed_file(file.filename):
        flash("Allowed file types: png, jpg, jpeg.", "danger")
        return redirect(request.url)

    try:
        # Run prediction
        prediction, confidence = run_inference(file)

        if not prediction:
            flash("Could not identify the plant. Please try another image.", "warning")
            return redirect(request.url)

        # Stay on predict page, show result + button to Gemini details
        return render_template(
            "predict.html",
            user=session.get("user"),
            classes=sorted(idx_to_class.values()),
            prediction=prediction,
            confidence=confidence,
            gemini_enabled=gemini_enabled,
        )

    except Exception as e:
        flash(f"An error occurred: {str(e)}", "danger")
        return redirect(request.url)


@app.route("/api/chat", methods=["POST"])
@login_required
def chat_api():
    if not GEMINI_API_KEY:
        return jsonify({"error": "Gemini not configured"}), 400

    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    plant_name = (data.get("plant") or "").strip() or None

    if not message:
        return jsonify({"error": "Message is required"}), 400

    reply = generate_health_chat(message, plant_name)
    return jsonify({"reply": reply})


@app.route("/api/remedies", methods=["POST"])
@login_required
def remedies_api():
    if not GEMINI_API_KEY:
        return jsonify({"error": "Gemini not configured"}), 400

    data = request.get_json(silent=True) or {}
    symptoms = (data.get("symptoms") or "").strip()
    plant_name = (data.get("plant") or "").strip() or None

    if not symptoms:
        return jsonify({"error": "Symptoms are required"}), 400

    reply = generate_remedies(symptoms, plant_name)
    formatted = format_basic_markdown(reply)
    return jsonify({
        "reply": formatted,
        "plain": reply,
        "used_plant": plant_name or ""
    })


@app.route("/gemini/<plant_name>")
@login_required
def gemini_details(plant_name: str):
    """Show the Gemini details for a predicted plant."""
    gemini_enabled = bool(GEMINI_API_KEY)

    # Default container to keep template keys stable
    gemini_response = {
        "overview": "",
        "medicinal_properties": [],
        "health_benefits": [],
        "therapeutic_applications": [],
        "precautions": [],
        "raw_html": "",
        "plain_text": "",
        "has_structured": False,
        "error": None,
    }

    if gemini_enabled:
        gemini_response = describe_plant(plant_name)
        if gemini_response.get("error"):
            print(f"Gemini Error: {gemini_response['error']}")
    else:
        gemini_response["error"] = "Gemini not configured. Please set GEMINI_API_KEY in your .env."

    return render_template(
        "gemini_response.html",
        plant_name=plant_name,
        confidence=None,
        sections=gemini_response,
        error=gemini_response.get("error"),
        user=session.get("user"),
    )


@app.route("/performance")
@login_required
def performance():
    """Show model performance and training info."""
    model_info = {
        "name": "ResNet50 (ImageNet1K_V2 backbone + custom head)",
        "dataset": "Medicinal plant leaves (10 classes)",
        "framework": "PyTorch",
        "params": "≈25.6M",
        "input_size": "224x224 RGB",
        "normalization": "ImageNet mean/std",
        "train_images": 1168,
        "val_images": 292,
    }
    metrics = {
        "top1_acc": "92.8%",
        "val_loss": "0.399",
        "last_epoch": 5,
        "train_acc": "96.4%",
        "train_loss": "0.408",
        "best_val_acc": "92.8%",
    }
    return render_template(
        "performance.html",
        model_info=model_info,
        metrics=metrics,
        user=session.get("user"),
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
