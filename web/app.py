from flask import Flask, render_template, request, send_from_directory

from recommender import FashionRecommender


app = Flask(__name__, static_folder="static", template_folder="templates")
service = FashionRecommender(data_dir="data")


@app.route("/")
def index():
    outfit_ids = service.sample_outfit_ids()
    category_choices = sorted(service.category_names.items(), key=lambda x: x[1])
    return render_template(
        "index.html",
        outfit_ids=outfit_ids,
        category_choices=category_choices,
    )


@app.route("/data/polyvore-images_smallsample/<set_id>/<filename>")
def sample_images(set_id, filename):
    return send_from_directory(f"../data/polyvore-images_smallsample/{set_id}", filename)


@app.route("/compatibility", methods=["POST"])
def compatibility():
    set_id = request.form.get("set_id")
    outfit = service.get_outfit(set_id)
    if not outfit:
        return render_template("result.html", title="Compatibility", error="Outfit not found.")

    score, categories = service.compatibility_score(outfit)
    images = service.outfit_images(set_id)
    return render_template(
        "result.html",
        title="Compatibility Recommendation",
        data={
            "set_id": set_id,
            "score": score,
            "categories": categories,
            "images": images,
        },
    )


@app.route("/fill-in-blank", methods=["POST"])
def fill_in_blank():
    set_id = request.form.get("set_id")
    outfit = service.get_outfit(set_id)
    if not outfit:
        return render_template("result.html", title="Fill in Blank", error="Outfit not found.")

    recs = service.fill_in_blank(outfit, top_k=5)
    images = service.outfit_images(set_id)
    return render_template(
        "result.html",
        title="Fill-in-the-Blank Recommendation",
        data={"set_id": set_id, "recs": recs, "images": images},
    )


@app.route("/multimodal", methods=["POST"])
def multimodal():
    text_query = request.form.get("text_query", "")
    preferred_categories = request.form.getlist("preferred_categories")

    results = service.multimodal_rank(text_query, preferred_categories, top_k=5)
    return render_template(
        "result.html",
        title="Multimodal Recommendation",
        data={"text_query": text_query, "results": results},
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
