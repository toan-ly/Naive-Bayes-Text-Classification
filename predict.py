import numpy as np
import preprocessing as pp

def predict(text, model, dictionary, le):
    processed_text = pp.preprocess_text(text)
    features = pp.create_features(processed_text, dictionary)
    features = np.array(features).reshape(1, -1)
    pred = model.predict(features)
    pred_cls = le.inverse_transform(pred)[0]
    return pred_cls

    