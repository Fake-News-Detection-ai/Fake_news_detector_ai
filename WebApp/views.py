import os
import json
import joblib
from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.http import require_POST

# Lazy-loaded model
_MODEL = None

def _load_model():
    global _MODEL
    if _MODEL is None:
        # Expect the model file at WebApp/models/fake_news_model.pkl
        model_path = os.path.join(settings.BASE_DIR, 'WebApp', 'models', 'fake_news_model.pkl')
        if not os.path.exists(model_path):
            # Try a project root path as fallback
            model_path = os.path.join(settings.BASE_DIR, 'fake_news_model.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found. Put your model at: WebApp/models/fake_news_model.pkl or at project root as fake_news_model.pkl")
        _MODEL = joblib.load(model_path)
    return _MODEL


def home(request):
    # simple GET renders the form; POST will be handled by predict view
    return render(request, 'home.html')


@require_POST
def predict(request):
    """Accepts form POST with 'text' (or 'title'+'text') and returns rendered page with prediction."""
    text = request.POST.get('text', '') or (request.POST.get('title', '') + ' ' + request.POST.get('text', ''))
    text = text.strip()
    if not text:
        return HttpResponseBadRequest("No text provided")

    model = _load_model()

    # If the model is a pipeline that accepts raw text, call predict directly.
    try:
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba([text])[0]
            classes = list(model.classes_) if hasattr(model, 'classes_') else None
            if classes is not None:
                idx = proba.argmax()
                pred_label = classes[idx]
                confidence = float(proba[idx])
            else:
                # unknown class mapping; return the max-prob index
                idx = proba.argmax()
                pred_label = str(idx)
                confidence = float(proba[idx])
        else:
            pred_label = model.predict([text])[0]
            confidence = None
    except Exception as e:
        # If the model expects vector input (no preprocessing), give a helpful error
        return HttpResponseBadRequest(f"Model inference failed: {e}. Ensure the pickle contains a preprocessing pipeline that accepts raw text.")

    return render(request, 'home.html', {
        'text': text,
        'prediction': pred_label,
        'confidence': confidence,
    })


@require_POST
def api_predict(request):
    """JSON API endpoint. POST JSON {"text": "..."} returns {prediction, confidence} JSON."""
    try:
        payload = json.loads(request.body.decode('utf-8'))
    except Exception:
        return HttpResponseBadRequest("Invalid JSON")
    text = payload.get('text', '')
    if not text:
        return HttpResponseBadRequest("No 'text' field provided")
    model = _load_model()
    try:
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba([text])[0]
            classes = list(model.classes_) if hasattr(model, 'classes_') else None
            if classes is not None:
                idx = proba.argmax()
                pred_label = classes[idx]
                confidence = float(proba[idx])
            else:
                idx = proba.argmax()
                pred_label = str(idx)
                confidence = float(proba[idx])
        else:
            pred_label = model.predict([text])[0]
            confidence = None
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)
    return JsonResponse({'prediction': str(pred_label), 'confidence': confidence})
