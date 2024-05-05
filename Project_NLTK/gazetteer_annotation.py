def annotate_gazetteer(tokens, gazetteers):
    gazetteer_annotation = []
    for token in tokens:
        annotated = token
        lowercase_token = token.lower()
        for category, gazetteer in gazetteers.items():
            lowercase_gazetteer = {item.lower() for item in gazetteer}
            if lowercase_token in lowercase_gazetteer:
                annotated = f"({token} {category})"
                break
        gazetteer_annotation.append(annotated)
    return gazetteer_annotation
