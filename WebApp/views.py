import os
import json
import joblib
from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.http import require_POST

# Lazy-loaded model
_MODEL = None
_VECT = None
_LABEL_MAP = None  # maps numeric class -> readable label (e.g., 0 -> 'real')
_MODEL_AMBIGUOUS = False



def _compute_label_map_from_dataset(model, vectorizer):
    """If the pickle doesn't include label mapping, infer it by running predictions
    on the project's dataset (`Documents/fake_news_dataset.csv`) and taking the
    majority true label per numeric class.
    Returns dict {numeric_label: string_label}.
    """
    possible_paths = [
        os.path.join(settings.BASE_DIR, 'Documents', 'fake_news_dataset.csv'),
        os.path.join(settings.BASE_DIR, 'PreparingDATA', 'precleaned_data.csv'),
        os.path.join(settings.BASE_DIR, 'PreparingDATA', 'precleaned_data.csv'),
    ]
    dataset_path = None
    for p in possible_paths:
        if os.path.exists(p):
            dataset_path = p
            break
    if dataset_path is None:
        # cannot infer mapping without labels; return numeric->string as str
        return {c: str(c) for c in getattr(model, 'classes_', [])}

    import pandas as pd
    df = pd.read_csv(dataset_path)
    if 'label' not in df.columns:
        return {c: str(c) for c in getattr(model, 'classes_', [])}
    texts = (df.get('title', '').fillna('') + ' ' + df.get('text', '').fillna('')) if 'title' in df.columns else df.iloc[:, 0].astype(str)
    X = vectorizer.transform(texts.tolist())
    preds = model.predict(X)
    # For each numeric class, find majority true label among rows predicted as that class
    from collections import Counter, defaultdict
    mapping = {}
    groups = defaultdict(list)
    for pred, true in zip(preds, df['label'].astype(str).str.lower()):
        groups[pred].append(true)
    for num_class, trues in groups.items():
        most_common = Counter(trues).most_common(1)
        mapping[num_class] = most_common[0][0] if most_common else str(num_class)
    # Ensure all classes in model.classes_ are present
    for c in getattr(model, 'classes_', []):
        if c not in mapping:
            mapping[c] = str(c)
    return mapping


def _check_model_ambiguity(model, vectorizer):
    """Return (is_ambiguous, message). Ambiguous if model predicts the same numeric
    class for both real and fake samples in the dataset (i.e., cannot distinguish).
    """
    possible = [
        os.path.join(settings.BASE_DIR, 'Documents', 'fake_news_dataset.csv'),
    ]
    dataset_path = next((p for p in possible if os.path.exists(p)), None)
    if dataset_path is None:
        return False, None
    import pandas as pd
    df = pd.read_csv(dataset_path)
    if 'label' not in df.columns:
        return False, None
    texts = (df.get('title', '').fillna('') + ' ' + df.get('text', '').fillna('')) if 'title' in df.columns else df.iloc[:, 0].astype(str)
    X = vectorizer.transform(texts.tolist())
    preds = model.predict(X)
    # Map true labels to the set of predicted numeric classes
    from collections import defaultdict, Counter
    groups = defaultdict(list)
    for pred, true in zip(preds, df['label'].astype(str).str.lower()):
        groups[true].append(pred)
    # compute mode for each true label
    modes = {lbl: Counter(vals).most_common(1)[0][0] if vals else None for lbl, vals in groups.items()}
    # if modes for 'real' and 'fake' exist and are equal, ambiguous
    if 'real' in modes and 'fake' in modes and modes['real'] == modes['fake']:
        msg = f"Model appears ambiguous: both 'real' and 'fake' examples are predicted as numeric class {modes['real']}. The model may be untrained or biased."
        return True, msg
    return False, None


def _load_model():
    global _MODEL, _VECT, _LABEL_MAP
    if _MODEL is None:
        # Expect the model file at WebApp/models/fake_news_model.pkl
        model_path = os.path.join(settings.BASE_DIR, 'WebApp', 'models', 'fake_news_model.pkl')
        if not os.path.exists(model_path):
            # Try a project root path as fallback
            model_path = os.path.join(settings.BASE_DIR, 'fake_news_model.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found. Put your model at: WebApp/models/fake_news_model.pkl or at project root as fake_news_model.pkl")
        obj = joblib.load(model_path)
        # Support two shapes: a) a pipeline/estimator object, b) a dict with keys 'model' and 'vectorizer'
        if isinstance(obj, dict):
            if 'model' in obj and 'vectorizer' in obj:
                _MODEL = obj['model']
                _VECT = obj['vectorizer']
                # If a label map is included, use it
                if 'label_map' in obj:
                    _LABEL_MAP = obj['label_map']
                else:
                    # attempt to infer mapping using dataset
                    try:
                        _LABEL_MAP = _compute_label_map_from_dataset(_MODEL, _VECT)
                    except Exception:
                        _LABEL_MAP = {c: str(c) for c in getattr(_MODEL, 'classes_', [])}
                    # check ambiguity
                    try:
                        ambiguous, msg = _check_model_ambiguity(_MODEL, _VECT)
                        global _MODEL_AMBIGUOUS
                        _MODEL_AMBIGUOUS = ambiguous
                    except Exception:
                        _MODEL_AMBIGUOUS = False
            else:
                # unexpected dict structure
                raise ValueError('Pickle dict must contain keys "model" and "vectorizer"')
        else:
            # assume obj is a scikit-learn estimator or pipeline that accepts raw text
            _MODEL = obj
            _VECT = None
            _LABEL_MAP = None
    return _MODEL, _VECT, _LABEL_MAP


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

    model, vect, label_map = _load_model()
    try:
        if vect is not None:
            X = vect.transform([text])
        else:
            # if no vectorizer, assume model accepts raw text
            X = [text]

        # Inspect the transformed feature vector to ensure transform() is working
        try:
            # If X is a sparse matrix (from sklearn vectorizers)
            from scipy import sparse
            if sparse.issparse(X):
                nnz = X.nnz
                shape = X.shape
                sample_nonzero = None
                # get first row non-zero indices and values
                row = X.getrow(0)
                nz_idx = row.indices.tolist()
                nz_val = row.data.tolist()
                sample_nonzero = list(zip(nz_idx, nz_val))[:10]
            else:
                # dense array
                nnz = (X != 0).sum()
                shape = getattr(X, 'shape', (len(X),))
                sample_nonzero = X[0][:10].tolist() if hasattr(X[0], 'tolist') else list(X[0])[:10]
        except Exception as _e:
            nnz = None
            shape = getattr(X, 'shape', None)
            sample_nonzero = None

        # If the transformed feature vector is empty (no known tokens), return a special response
        no_tokens = False
        try:
            from scipy import sparse
            if sparse.issparse(X) and X.nnz == 0:
                no_tokens = True
        except Exception:
            # If scipy not available or X not sparse, attempt generic check
            try:
                if getattr(X, 'size', None) == 0 or (hasattr(X, '__len__') and len(X) and all(v == 0 for v in X[0])):
                    no_tokens = True
            except Exception:
                no_tokens = False

        if no_tokens:
            # Build a helpful message: show which tokens (if any) are recognized
            recognized = []
            try:
                analyzer = _VECT.build_analyzer()
                toks = analyzer(text)
                vocab = getattr(_VECT, 'vocabulary_', {}) if _VECT is not None else {}
                recognized = [t for t in toks if t in vocab]
            except Exception:
                recognized = []

            pred_label = 'Unrecognized input'
            confidence = None
            print("Debug - Text (truncated):", text[:120])
            print("Debug - Warning: transformed vector has zero non-zero features. No recognized tokens.")
            print("Debug - Analyzer tokens:", toks if 'toks' in locals() else None)
            print("Debug - Recognized tokens in vocab:", recognized)
        else:
            # Get probabilities and raw prediction
            proba = model.predict_proba(X)[0]  # Shape (2,) for binary classification
            raw_pred = model.predict(X)[0]      # 0 or 1

            # Get confidence as probability of predicted class
            # raw_pred is numeric class (0/1); safe to index proba
            confidence = float(proba[int(raw_pred)])

            # Convert numeric prediction to label using mapping
            pred_label = label_map.get(raw_pred, str(raw_pred))

            # Log prediction and vectorization details for debugging
            print("Debug - Text (truncated):", text[:120])
            print("Debug - Feature vector shape:", shape, "nonzeros:", nnz)
            print("Debug - Feature vector sample nonzeros (idx, val):", sample_nonzero)
            print("Debug - Raw prediction:", raw_pred)
            print("Debug - Probabilities:", proba)
            print("Debug - Mapped label:", pred_label)
            print("Debug - Confidence:", confidence)
    except Exception as e:
        return HttpResponseBadRequest(f"Model inference failed: {e}")

    return render(request, 'home.html', {
        'text': text,
        'prediction': pred_label,
        'confidence': confidence,
        'model_ambiguous': _MODEL_AMBIGUOUS,
        'no_tokens': no_tokens,
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
    model, vect, label_map = _load_model()
    try:
        if vect is not None:
            X = vect.transform([text])
        else:
            X = [text]

        # Inspect the transformed feature vector
        try:
            from scipy import sparse
            if sparse.issparse(X):
                nnz = X.nnz
                shape = X.shape
                row = X.getrow(0)
                sample_nonzero = list(zip(row.indices.tolist(), row.data.tolist()))[:10]
            else:
                nnz = (X != 0).sum()
                shape = getattr(X, 'shape', (len(X),))
                sample_nonzero = X[0][:10].tolist() if hasattr(X[0], 'tolist') else list(X[0])[:10]
        except Exception:
            nnz = None
            shape = getattr(X, 'shape', None)
            sample_nonzero = None

        # Get probabilities and raw prediction
        proba = model.predict_proba(X)[0]
        raw_pred = model.predict(X)[0]

        confidence = float(proba[int(raw_pred)])
        pred_label = label_map.get(raw_pred, str(raw_pred))

        # If no non-zero features, return an explanatory JSON response
        no_tokens_api = False
        try:
            from scipy import sparse
            if sparse.issparse(X) and X.nnz == 0:
                no_tokens_api = True
        except Exception:
            no_tokens_api = False

        print("Debug - API - Text (truncated):", text[:120])
        print("Debug - API - Feature vector shape:", shape, "nonzeros:", nnz)
        print("Debug - API - Feature vector sample nonzeros (idx, val):", sample_nonzero)
        print("Debug - API - Raw prediction:", raw_pred)
        print("Debug - API - Probabilities:", proba)
        print("Debug - API - Mapped label:", pred_label)
        print("Debug - API - Confidence:", confidence)

        if no_tokens_api:
            return JsonResponse({'prediction': 'Unrecognized input', 'confidence': None, 'recognized_tokens': [], 'model_ambiguous': _MODEL_AMBIGUOUS})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)
    return JsonResponse({'prediction': str(pred_label), 'confidence': confidence, 'model_ambiguous': _MODEL_AMBIGUOUS})
