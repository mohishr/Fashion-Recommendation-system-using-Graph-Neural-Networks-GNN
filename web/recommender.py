import json
import math
import os
from collections import Counter, defaultdict


class FashionRecommender:
    """Lightweight recommendation service backed by repository sample data.

    It mirrors the major recommendation tasks in the NGNN codebase:
    - Outfit compatibility scoring
    - Fill-in-the-blank (item replacement)
    - Multimodal ranking (category + text intent)
    """

    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.train_path = os.path.join(data_dir, "train_no_dup_new_100.json")
        self.test_path = os.path.join(data_dir, "test_no_dup_new_100.json")
        self.category_summary_path = os.path.join(data_dir, "category_summarize_100.json")
        self.category_names_path = os.path.join(data_dir, "category_id.txt")
        self.image_root = os.path.join(data_dir, "polyvore-images_smallsample")

        self.train = self._load_json(self.train_path)
        self.test = self._load_json(self.test_path)
        self.category_summary = self._load_json(self.category_summary_path)
        self.category_names = self._load_category_names(self.category_names_path)

        self.category_freq = self._build_category_frequency(self.train)
        self.cooccur = self._build_cooccurrence_graph(self.train)

        self.outfits_with_images = [
            o for o in (self.train + self.test)
            if os.path.isdir(os.path.join(self.image_root, str(o["set_id"])))
        ]
        if not self.outfits_with_images:
            self.outfits_with_images = self._build_fallback_outfits()

    @staticmethod
    def _load_json(path):
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def _load_category_names(path):
        names = {}
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cid, cname = line.split(" ", 1)
                names[int(cid)] = cname
        return names

    @staticmethod
    def _build_category_frequency(outfits):
        freq = Counter()
        for outfit in outfits:
            for cid in outfit["items_category"]:
                freq[int(cid)] += 1
        return freq

    @staticmethod
    def _build_cooccurrence_graph(outfits):
        graph = defaultdict(Counter)
        for outfit in outfits:
            cats = [int(c) for c in outfit["items_category"]]
            for i, c1 in enumerate(cats):
                for j, c2 in enumerate(cats):
                    if i == j:
                        continue
                    graph[c1][c2] += 1
        return graph

    def _category_name(self, cid):
        return self.category_names.get(int(cid), f"Category {cid}")


    def _build_fallback_outfits(self):
        fallback = []
        common_cats = [cid for cid, _ in self.category_freq.most_common(12)]
        if not common_cats:
            common_cats = [s["id"] for s in self.category_summary[:12]]

        for set_id in sorted(os.listdir(self.image_root)):
            folder = os.path.join(self.image_root, set_id)
            if not os.path.isdir(folder):
                continue
            item_images = [f for f in os.listdir(folder) if f.endswith(".jpg") and f != "0.jpg"]
            item_count = max(1, len(item_images))
            categories = [int(common_cats[i % len(common_cats)]) for i in range(item_count)]
            indices = list(range(1, item_count + 1))
            fallback.append({
                "set_id": str(set_id),
                "items_category": categories,
                "items_index": indices,
            })
        return fallback
    def sample_outfit_ids(self):
        return sorted({str(o["set_id"]) for o in self.outfits_with_images})

    def get_outfit(self, set_id):
        for outfit in self.outfits_with_images:
            if str(outfit["set_id"]) == str(set_id):
                return outfit
        return None

    def outfit_images(self, set_id):
        base = os.path.join(self.image_root, str(set_id))
        if not os.path.isdir(base):
            return []
        files = [f for f in os.listdir(base) if f.endswith(".jpg") and f != "0.jpg"]
        files = sorted(files, key=lambda x: int(x.split(".")[0]))
        return [f"/{self.image_root}/{set_id}/{f}" for f in files]

    def compatibility_score(self, outfit):
        cats = [int(c) for c in outfit["items_category"]]
        if not cats:
            return 0.0, []

        pair_scores = []
        for i, c1 in enumerate(cats):
            for j, c2 in enumerate(cats):
                if i >= j:
                    continue
                co = self.cooccur[c1][c2]
                denom = max(1, self.category_freq[c1] + self.category_freq[c2])
                pair_scores.append(float(co) / float(denom))

        raw = sum(pair_scores) / max(1, len(pair_scores))
        normalized = 100.0 * (1 - math.exp(-20 * raw))
        details = [self._category_name(c) for c in cats]
        return round(normalized, 2), details

    def fill_in_blank(self, outfit, top_k=5):
        cats = [int(c) for c in outfit["items_category"]]
        if len(cats) < 2:
            return []

        removed = cats[-1]
        context = cats[:-1]
        candidate_scores = []
        for summary in self.category_summary:
            candidate = int(summary["id"])
            if candidate in context:
                continue
            score = 0
            for c in context:
                score += self.cooccur[c][candidate]
            score += self.category_freq[candidate] * 0.1
            candidate_scores.append((candidate, score))

        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        top = candidate_scores[:top_k]
        return {
            "removed_category": self._category_name(removed),
            "suggestions": [
                {"category_id": c, "category_name": self._category_name(c), "score": round(float(s), 2)}
                for c, s in top
            ],
        }

    def multimodal_rank(self, query_text, preferred_categories, top_k=5):
        words = {w.strip().lower() for w in query_text.split() if w.strip()}
        prefs = {int(c) for c in preferred_categories if str(c).strip()}
        results = []

        for outfit in self.outfits_with_images:
            cats = [int(c) for c in outfit["items_category"]]
            cat_match = len([c for c in cats if c in prefs])
            text_match = 0
            for c in cats:
                cname = self._category_name(c).lower()
                if any(w in cname for w in words):
                    text_match += 1
            comp, _ = self.compatibility_score(outfit)
            score = (0.5 * comp) + (20.0 * cat_match) + (15.0 * text_match)
            results.append(
                {
                    "set_id": str(outfit["set_id"]),
                    "score": round(score, 2),
                    "categories": [self._category_name(c) for c in cats],
                    "image_count": len(self.outfit_images(outfit["set_id"])),
                }
            )

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
