# enhanced_fake_news_detector_v2.py
# FULLY FIXED VERSION - Better entity extraction, query construction, and relevance filtering
#
# NEW FIXES:
# 1. Improved entity extraction (handles brands, products, quoted terms)
# 2. Better query construction with fallback strategies
# 3. Strict relevance filtering before sentiment analysis
# 4. Minimum relevance threshold to ignore unrelated articles

import os
import re
import pickle
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from urllib.parse import urlparse, quote as urlquote
from xml.etree import ElementTree
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
from difflib import SequenceMatcher

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

UTC = timezone.utc

class EnhancedFakeNewsDetector:
    def __init__(self, use_web_verification=True, resolve_redirects=False, debug=False):
        self.vectorizer = TfidfVectorizer(
            max_features=50000, stop_words='english', ngram_range=(1, 2),
            analyzer='word', min_df=2
        )
        base = LogisticRegression(max_iter=2000, class_weight='balanced', C=4.0, n_jobs=None)
        self.model = CalibratedClassifierCV(base, method='isotonic', cv=5)

        self.use_web = use_web_verification
        self.resolve_redirects = resolve_redirects
        self.debug = debug

        # Trusted domains
        self.trusted_domains = {
            'reuters.com','apnews.com','bbc.com','bbc.co.uk','nytimes.com',
            'washingtonpost.com','theguardian.com','npr.org','pbs.org',
            'cnbc.com','bloomberg.com','cnn.com','abcnews.go.com','nbcnews.com',
            'usatoday.com','wsj.com','time.com','newsweek.com','politico.com',
            'snopes.com','factcheck.org','politifact.com'
        }
        self.trusted_name_aliases = {
            'reuters':'reuters.com','associated press':'apnews.com','ap news':'apnews.com',
            'bbc':'bbc.com','bbc news':'bbc.com','new york times':'nytimes.com',
            'the new york times':'nytimes.com','washington post':'washingtonpost.com',
            'the guardian':'theguardian.com','npr':'npr.org','pbs':'pbs.org','cnbc':'cnbc.com',
            'bloomberg':'bloomberg.com','cnn':'cnn.com','abc news':'abcnews.go.com',
            'nbc news':'nbcnews.com','usa today':'usatoday.com','wall street journal':'wsj.com',
            'wsj':'wsj.com','time':'time.com','newsweek':'newsweek.com','politico':'politico.com',
            'snopes':'snopes.com'
        }

        # Guards
        self.death_patterns = re.compile(r'\b(dies|dead|found\s+dead|passed\s+away|death|killed)\b', re.I)
        self.extraordinary_cues = [
            r'invisible', r'quantum\s+cloaking', r'cloaking\s+polymer',
            r'perpetual\s+motion', r'anti-?gravity', r'time\s+travel', 
            r'miracle\s+cure', r'sleep.*72.*hour', r'sleep.*three.*day'
        ]
        
        # Sentiment keywords
        self.debunk_keywords = [
            'false', 'fake', 'hoax', 'debunk', 'myth', 'misinformation', 
            'disinformation', 'untrue', 'not true', 'incorrect', 'wrong',
            'fact check', 'fact-check', 'misleading', 'unverified', 'no evidence',
            'skeptic', 'skeptical', 'doubt', 'questionable', 'unproven'
        ]
        self.support_keywords = [
            'confirmed', 'verify', 'true', 'accurate', 'reports', 
            'official', 'statement', 'announces', 'confirms', 'proven',
            'study shows', 'research shows', 'evidence', 'validated'
        ]

    # ------------- utils -------------
    @staticmethod
    def _hostname(domain_or_url: str) -> str:
        try:
            netloc = urlparse(domain_or_url).netloc.lower()
            if not netloc:
                netloc = (domain_or_url or '').lower()
            if netloc.startswith('www.'):
                netloc = netloc[4:]
            return netloc
        except Exception:
            return (domain_or_url or '').lower()

    def _alias_to_domain(self, name: str):
        return self.trusted_name_aliases.get((name or '').strip().lower())

    @staticmethod
    def _clean_for_ml(text):
        t = str(text).lower()
        t = re.sub(r'[^a-zA-Z\s]', ' ', t)
        return ' '.join(t.split())

    def _extract_key_terms(self, text: str):
        """
        Extract key terms for search query including:
        - Quoted terms
        - Capitalized brands/products/names
        - Important nouns
        """
        terms = set()
        
        # Extract quoted terms (brand names, product names)
        quoted = re.findall(r'"([^"]+)"', text or '')
        for q in quoted:
            if len(q.split()) <= 3:  # Short quoted terms are likely brand/product names
                terms.add(q)
        
        # Extract capitalized multi-word terms (e.g., "SomnaLabs", "TriRest")
        # Look for patterns like "SomnaLabs" or "CamelCase" words
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', text or '')
        terms.update(capitalized)
        
        # Extract regular proper nouns (capitalized words)
        proper = re.findall(r'\b[A-Z][a-z]{2,}\b', text or '')
        terms.update(proper[:5])  # Limit to avoid too generic
        
        # Extract important keywords (nouns that might be central to the claim)
        important_patterns = [
            r'\b(pill|drug|medication|vaccine|treatment|cure|device|technology|invention|product)\b',
            r'\b(study|research|trial|experiment|test|investigation)\b',
            r'\b(company|startup|corporation|firm|lab|laboratory)\b',
            r'\b(scientist|doctor|professor|researcher|expert)\b'
        ]
        for pattern in important_patterns:
            matches = re.findall(pattern, text or '', re.I)
            terms.update(m.lower() for m in matches[:2])
        
        # Filter out very common words
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                    'said', 'says', 'new', 'old', 'first', 'last', 'other'}
        terms = {t for t in terms if t.lower() not in stopwords and len(t) >= 3}
        
        return terms

    def _construct_search_query(self, text: str, max_terms=5):
        """
        Build an intelligent search query from the text.
        """
        terms = self._extract_key_terms(text)
        
        if not terms:
            # Fallback: extract most distinctive words
            words = text.split()[:20]
            # Get words that are longer and more likely to be meaningful
            meaningful = [w for w in words if len(w) >= 4 and w[0].isupper()]
            if meaningful:
                return ' '.join(meaningful[:max_terms])
            else:
                # Last resort: just use first few words
                return ' '.join(text.split()[:max_terms])
        
        # Prioritize quoted terms and capitalized terms
        query_terms = list(terms)[:max_terms]
        query = ' '.join(query_terms)
        
        if len(query) < 10:
            # Query too short, add context
            words = text.split()[:10]
            additional = [w for w in words if len(w) >= 4 and w not in query]
            query = query + ' ' + ' '.join(additional[:3])
        
        return query.strip()

    def _calculate_relevance(self, article_title: str, claim_text: str, query_terms: set) -> float:
        """
        Calculate how relevant an article is to the claim (0.0 to 1.0).
        """
        title_lower = article_title.lower()
        claim_lower = claim_text.lower()
        
        # Count matching terms
        matches = sum(1 for term in query_terms if term.lower() in title_lower)
        
        if len(query_terms) == 0:
            return 0.0
        
        term_match_ratio = matches / len(query_terms)
        
        # Sequence similarity
        similarity = SequenceMatcher(None, title_lower, claim_lower[:100]).ratio()
        
        # Combined relevance score
        relevance = 0.7 * term_match_ratio + 0.3 * similarity
        
        return min(relevance, 1.0)

    def _is_probably_news(self, text: str) -> bool:
        t = (text or '').strip()
        if len(t) < 20:
            return False
        if re.search(r'\b([A-Z][a-z]{2,}|202[0-9]|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\b', t):
            return True
        return False

    def _is_death_claim(self, text: str) -> bool:
        return bool(self.death_patterns.search(text or ''))

    def _looks_extraordinary(self, text: str) -> bool:
        tl = (text or '').lower()
        return any(re.search(c, tl) for c in self.extraordinary_cues)
    
    def _analyze_article_sentiment(self, article_title: str, claim_text: str) -> str:
        """
        Determine if an article is SUPPORTING, DEBUNKING, or NEUTRAL.
        """
        title_lower = article_title.lower()
        
        debunk_score = sum(1 for kw in self.debunk_keywords if kw in title_lower)
        support_score = sum(1 for kw in self.support_keywords if kw in title_lower)
        
        if debunk_score > support_score and debunk_score > 0:
            return 'debunk'
        elif support_score > debunk_score and support_score > 0:
            return 'support'
        else:
            return 'neutral'

    # ------------- data / training -------------
    def load_and_prepare_data(self, fake_path, true_path):
        print("Loading datasets...")
        fake_df = pd.read_csv(fake_path)
        true_df = pd.read_csv(true_path)
        
        fake_df['label'] = 0
        true_df['label'] = 1
        
        print(f"Fake dataset shape: {fake_df.shape}, assigned label: 0 (FAKE)")
        print(f"True dataset shape: {true_df.shape}, assigned label: 1 (TRUE)")
        
        df = pd.concat([fake_df, true_df], ignore_index=True)
        print(f"Total samples: {len(df)}")
        print(f"Label distribution: FAKE (0): {(df['label']==0).sum()}, TRUE (1): {(df['label']==1).sum()}")

        cols = {c.lower(): c for c in df.columns}
        title_col, text_col = cols.get('title'), cols.get('text')
        if title_col and text_col:
            df['content'] = df[title_col].fillna('') + ' ' + df[text_col].fillna('')
        elif text_col:
            df['content'] = df[text_col].fillna('')
        elif title_col:
            df['content'] = df[title_col].fillna('')
        else:
            raise ValueError("Dataset must contain 'title' or 'text' column")

        print("Cleaning text data...")
        df['content_ml'] = df['content'].apply(self._clean_for_ml)
        df = df[df['content_ml'].str.len() > 0]
        return df['content_ml'], df['label']

    def train(self, fake_path, true_path):
        X, y = self.load_and_prepare_data(fake_path, true_path)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        print(f"\nTraining samples: {len(Xtr)}")
        print(f"  - FAKE (0): {(ytr==0).sum()}")
        print(f"  - TRUE (1): {(ytr==1).sum()}")
        print(f"Testing samples:  {len(Xte)}")
        print(f"  - FAKE (0): {(yte==0).sum()}")
        print(f"  - TRUE (1): {(yte==1).sum()}")

        print("\nVectorizing text...")
        VXtr = self.vectorizer.fit_transform(Xtr)
        VXte = self.vectorizer.transform(Xte)

        print("Training model (calibrated LR)...")
        self.model.fit(VXtr, ytr)

        print("\nEvaluating model...")
        yhat = self.model.predict(VXte)
        acc = accuracy_score(yte, yhat)
        print(f"\nAccuracy: {acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(yte, yhat, target_names=['FAKE (0)', 'TRUE (1)']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(yte, yhat)
        print(f"              Predicted FAKE  Predicted TRUE")
        print(f"Actual FAKE   {cm[0,0]:15d}  {cm[0,1]:14d}")
        print(f"Actual TRUE   {cm[1,0]:15d}  {cm[1,1]:14d}")
        
        return acc

    # ------------- web search -------------
    def search_google_news(self, query):
        try:
            url = f"https://news.google.com/rss/search?q={urlquote(query)}&hl=en-US&gl=US&ceid=US:en"
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code != 200:
                return []
            root = ElementTree.fromstring(r.content)
            out = []
            for item in root.findall('.//item')[:25]:
                title_el = item.find('title')
                link_el = item.find('link')
                src_el = item.find('source')
                date_el = item.find('pubDate')
                title = title_el.text if title_el is not None else ''
                link = link_el.text if link_el is not None else ''
                src_name = src_el.text if (src_el is not None and src_el.text) else ''
                pub_str = date_el.text if date_el is not None else None
                pub_dt = parsedate_to_datetime(pub_str) if pub_str else None
                
                if src_el is not None and hasattr(src_el, 'attrib'):
                    src_url = src_el.attrib.get('url', '')
                else:
                    src_url = ''
                
                out.append({
                    'title': title,
                    'link': link,
                    'source_name': src_name,
                    'source_url': src_url,
                    'pub_date': pub_dt
                })
            return out
        except Exception as e:
            if self.debug:
                print(f"  [DEBUG] Google News search failed: {e}")
            return []

    def _filter_and_score_results(self, results, claim_text, query_terms):
        now = datetime.now(UTC)
        cutoff_days = 30
        
        filtered = []
        trusted_domains_found = []
        relevant_articles = []
        
        support_count = 0
        debunk_count = 0
        neutral_count = 0
        
        # MINIMUM RELEVANCE THRESHOLD
        MIN_RELEVANCE = 0.15  # Articles must be at least 15% relevant

        for r in results:
            pub = r.get('pub_date')
            if pub:
                age_days = (now - pub).days
                if age_days > cutoff_days:
                    continue
            
            title = r.get('title', '')
            
            # Calculate relevance
            relevance = self._calculate_relevance(title, claim_text, query_terms)
            
            if self.debug and relevance > 0:
                print(f"  [DEBUG] Article relevance: {relevance:.2f} - {title[:60]}")
            
            # SKIP IRRELEVANT ARTICLES
            if relevance < MIN_RELEVANCE:
                if self.debug:
                    print(f"  [DEBUG] Skipping low-relevance article: {title[:60]}")
                continue
            
            filtered.append(r)
            
            src_url = r.get('source_url', '')
            src_name = r.get('source_name', '')
            dom = self._hostname(src_url) if src_url else ''
            
            if not dom:
                d2 = self._alias_to_domain(src_name)
                if d2:
                    dom = d2
            
            is_trusted = (dom in self.trusted_domains) if dom else False
            
            if is_trusted:
                trusted_domains_found.append(dom)
                
                sentiment = self._analyze_article_sentiment(title, claim_text)
                if sentiment == 'support':
                    support_count += 1
                elif sentiment == 'debunk':
                    debunk_count += 1
                else:
                    neutral_count += 1
                
                relevant_articles.append({
                    'title': title,
                    'source': r.get('source_name') or dom or 'Unknown',
                    'sentiment': sentiment,
                    'relevance': relevance
                })

        trusted_count = len([d for d in trusted_domains_found if d])
        total_results = len(filtered)

        # Calculate web_score based on sentiment
        if trusted_count >= 2:
            if debunk_count > support_count:
                web_score = 0.15 + (support_count * 0.10)
                verdict = f"⚠️ Found {debunk_count} article(s) debunking this claim"
            elif support_count > debunk_count:
                web_score = 0.75 + (support_count * 0.05)
                web_score = min(web_score, 0.95)
                verdict = f"✓ Found {support_count} article(s) supporting this claim"
            else:
                web_score = 0.50
                verdict = f"⚠ Mixed coverage: {support_count} supporting, {debunk_count} debunking"
        elif trusted_count == 1:
            if debunk_count > 0:
                web_score = 0.25
                verdict = "⚠️ Single source indicates this may be false"
            elif support_count > 0:
                web_score = 0.55
                verdict = "✓ Single trusted source found"
            else:
                web_score = 0.40
                verdict = "⚠ Single source with neutral coverage"
        else:
            # NO TRUSTED SOURCES - very low score
            web_score = 0.20 if total_results >= 3 else 0.10
            verdict = f"✗ No trusted sources found ({total_results} untrusted/irrelevant results)"

        return filtered, trusted_domains_found, {
            'web_score': web_score,
            'trusted_sources': trusted_count,
            'total_results': total_results,
            'verdict': verdict,
            'sources': sorted([d for d in trusted_domains_found if d])[:5],
            'found_articles': relevant_articles[:5],
            'support_count': support_count,
            'debunk_count': debunk_count,
            'neutral_count': neutral_count
        }

    def verify_with_web(self, news_text):
        print("\n🔍 Searching the web for verification...")
        
        # Construct intelligent query
        query = self._construct_search_query(news_text)
        query_terms = self._extract_key_terms(news_text)
        
        print(f'  Query: "{query}"')
        if self.debug:
            print(f'  [DEBUG] Query terms: {query_terms}')

        results = self.search_google_news(query)

        if not results:
            print("  ⚠ No results found")
            return {
                'web_score': 0.0,
                'trusted_sources': 0,
                'total_results': 0,
                'verdict': "No verification data found",
                'sources': [],
                'found_articles': [],
                'support_count': 0,
                'debunk_count': 0,
                'neutral_count': 0
            }

        print(f"  Found {len(results)} articles initially")
        _, _, scored = self._filter_and_score_results(results, news_text, query_terms)
        
        print(f"  → {scored['total_results']} relevant articles after filtering")
        
        if self.debug:
            print(f"  [DEBUG] Web score: {scored['web_score']:.2f}")
            print(f"  [DEBUG] Support: {scored['support_count']}, Debunk: {scored['debunk_count']}, Neutral: {scored['neutral_count']}")
        
        return scored

    # ------------- inference -------------
    def predict(self, news_text):
        if not self._is_probably_news(news_text):
            return -1, 1.0, np.array([0.5, 0.5]), {
                'web_score': 0.0, 'trusted_sources': 0, 'total_results': 0,
                'verdict': 'Out-of-distribution (not a news claim)', 'sources': [], 
                'found_articles': [], 'support_count': 0, 'debunk_count': 0, 'neutral_count': 0
            }

        vec = self.vectorizer.transform([self._clean_for_ml(news_text)])
        ml_pred = int(self.model.predict(vec)[0])
        ml_prob = self.model.predict_proba(vec)[0]
        
        ml_fake_prob = float(ml_prob[0])
        ml_true_prob = float(ml_prob[1])
        
        if self.debug:
            print(f"\n[DEBUG] ML Prediction:")
            print(f"  Raw pred: {ml_pred} (0=FAKE, 1=TRUE)")
            print(f"  P(FAKE): {ml_fake_prob:.4f}, P(TRUE): {ml_true_prob:.4f}")

        web = self.verify_with_web(news_text) if self.use_web else None

        # Extraordinary-claim hard guard
        if self._looks_extraordinary(news_text):
            if (not web) or (web['total_results'] == 0) or (web['trusted_sources'] == 0):
                if self.debug:
                    print("[DEBUG] Extraordinary claim guard triggered → FAKE")
                return 0, 0.92, ml_prob, web

        # Death-claim hard rule
        if self._is_death_claim(news_text):
            if not web or web['trusted_sources'] < 2 or web['total_results'] == 0:
                if self.debug:
                    print("[DEBUG] Death claim guard triggered → FAKE")
                return 0, 0.92, ml_prob, web

        # Fusion with improved logic
        if web and (web['total_results'] > 0):
            if web['trusted_sources'] >= 2:
                w_web = 0.65
            elif web['trusted_sources'] == 1:
                w_web = 0.50
            else:
                # No trusted sources found - trust ML much more
                w_web = 0.25
            
            w_ml = 1.0 - w_web
            
            true_score = w_ml * ml_true_prob + w_web * web['web_score']
            fake_score = w_ml * ml_fake_prob + w_web * (1.0 - web['web_score'])
            
            final_pred = 1 if true_score >= fake_score else 0
            final_conf = max(true_score, fake_score)
            
            if self.debug:
                print(f"[DEBUG] Fusion:")
                print(f"  w_ml={w_ml:.2f}, w_web={w_web:.2f}")
                print(f"  true_score = {w_ml:.2f}*{ml_true_prob:.4f} + {w_web:.2f}*{web['web_score']:.4f} = {true_score:.4f}")
                print(f"  fake_score = {w_ml:.2f}*{ml_fake_prob:.4f} + {w_web:.2f}*{1.0-web['web_score']:.4f} = {fake_score:.4f}")
                print(f"  Final: {final_pred} (0=FAKE, 1=TRUE), conf={final_conf:.4f}")
        else:
            # No web evidence
            final_pred = ml_pred
            final_conf = min(0.85, ml_true_prob if ml_pred == 1 else ml_fake_prob)
            
            if self.debug:
                print(f"[DEBUG] No relevant web evidence, using ML: pred={final_pred}, conf={final_conf:.4f}")

        return final_pred, final_conf, ml_prob, web

    # ------------- persistence -------------
    def save_model(self, filename='enhanced_fake_news_model_v2.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump({'model': self.model, 'vectorizer': self.vectorizer}, f)
        print(f"\nModel saved to {filename}")

    def load_model(self, filename='enhanced_fake_news_model_v2.pkl'):
        with open(filename, 'rb') as f:
            d = pickle.load(f)
            self.model = d['model']
            self.vectorizer = d['vectorizer']
        print(f"Model loaded from {filename}")


def main():
    desktop = Path.home() / "Desktop"
    dataset_folder = desktop / "NEW DATASET"
    fake_path = dataset_folder / "Fake.csv"
    true_path = dataset_folder / "True.csv"

    if not fake_path.exists() or not true_path.exists():
        print(f"Error: Could not find datasets at {dataset_folder}")
        print("Expected files:")
        print(f"  - {fake_path}")
        print(f"  - {true_path}")
        return

    print("=" * 60)
    print("FAKE NEWS DETECTOR V2 - FULLY FIXED")
    print("=" * 60)

    det = EnhancedFakeNewsDetector(use_web_verification=True, resolve_redirects=False, debug=True)
    model_file = 'enhanced_fake_news_model_v2.pkl'

    if os.path.exists(model_file):
        print("\nFound existing model. Do you want to:")
        print("1. Use existing model")
        print("2. Train new model")
        choice = input("Enter choice (1 or 2): ").strip()
        if choice == '1':
            det.load_model(model_file)
        else:
            print("\n" + "=" * 60)
            det.train(fake_path, true_path)
            det.save_model(model_file)
    else:
        print("\n" + "=" * 60)
        det.train(fake_path, true_path)
        det.save_model(model_file)

    print("\n" + "=" * 60)
    print("FAKE NEWS DETECTOR - READY")
    print("=" * 60)
    print("This detector uses:")
    print("  • Improved entity extraction (brands, products, quoted terms)")
    print("  • Intelligent query construction with fallback strategies")
    print("  • Strict relevance filtering (ignores unrelated articles)")
    print("  • Sentiment-aware web verification")
    print("\nEnter news text to check. Type 'quit' to exit.\n")

    while True:
        try:
            text = input("\n📰 Enter news text: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting... Goodbye!")
            break
        if text.lower() in ['quit', 'exit', 'q']:
            print("Exiting... Goodbye!")
            break
        if not text:
            print("Please enter some text.")
            continue

        print("\n" + "=" * 60)
        print("ANALYZING...")
        print("=" * 60)
        pred, conf, ml_probs, web = det.predict(text)

        print("\n📊 ML MODEL ANALYSIS:")
        if pred == -1:
            print("  Skipped (OOD: not a news claim)")
        else:
            ml_label = 'TRUE' if ml_probs[1] >= ml_probs[0] else 'FAKE'
            print(f"  ML Prediction: {ml_label}")
            print(f"  P(FAKE): {ml_probs[0]*100:.2f}%")
            print(f"  P(TRUE): {ml_probs[1]*100:.2f}%")

        if web:
            print("\n🌐 WEB VERIFICATION:")
            print(f"  {web['verdict']}")
            print(f"  Credibility Score: {web['web_score']*100:.1f}%")
            if web.get('support_count', 0) > 0 or web.get('debunk_count', 0) > 0:
                print(f"  Articles: {web.get('support_count', 0)} supporting, {web.get('debunk_count', 0)} debunking, {web.get('neutral_count', 0)} neutral")
            if web.get('sources'):
                print(f"  Trusted sources: {', '.join(web['sources'][:5])}")
            if web.get('found_articles'):
                print("\n  📄 Relevant articles found:")
                for a in web['found_articles']:
                    sentiment_icon = "✅" if a.get('sentiment') == 'support' else ("❌" if a.get('sentiment') == 'debunk' else "⚪")
                    rel = a.get('relevance', 0)
                    print(f"    {sentiment_icon} [{rel:.0%} relevant] {(a.get('title') or '')[:60]} ({a.get('source') or 'Unknown'})")

        print("\n" + "=" * 60)
        print("⚖️  FINAL VERDICT:")
        print("=" * 60)
        if pred == -1:
            print("⚪ NOT A NEWS CLAIM (skipped)")
        elif pred == 1:
            print("✅ LIKELY TRUE NEWS")
        else:
            print("❌ LIKELY FAKE NEWS")
        print(f"Overall Confidence: {conf*100:.2f}%")
        print("=" * 60)


if __name__ == "__main__":
    main()
