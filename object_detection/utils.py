def format_results(boxes, scores, image_id, cat_id):
    results = []
    for box, score in zip(boxes, scores):
        r = {
            "image_id": image_id,
            "category_id": cat_id,
            "bbox": [float(i) for i in box],
            "score": float(score),
        }
        results.append(r)
    return results
