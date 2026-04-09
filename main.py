import os
import io
import json
import base64
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import cv2
import uvicorn

app = FastAPI(title="Green DZ - Score-CAM Plant API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@tf.keras.utils.register_keras_serializable(package="Custom")
class LKAM(tf.keras.layers.Layer):
    def __init__(self, channels=None, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
    def build(self, input_shape):
        if self.channels is None: self.channels = input_shape[-1]
        self.dw_conv = tf.keras.layers.DepthwiseConv2D(kernel_size=7, padding='same')
        self.pw_conv = tf.keras.layers.Conv2D(self.channels, kernel_size=1)
        super().build(input_shape)
    def call(self, x): return x * self.pw_conv(self.dw_conv(x))
    def get_config(self):
        config = super().get_config()
        config.update({"channels": self.channels})
        return config


def get_score_cam(img_input, model):
    try:
        base_model = model.get_layer('efficientnetb0')
        target_layer = base_model.get_layer('top_activation')
        features_model = tf.keras.models.Model(inputs=[base_model.input], outputs=[target_layer.output])
        features = features_model.predict(img_input, verbose=0)
        
        features = features[0]
        heatmap = np.zeros(features.shape[:2], dtype=np.float32)
        
    
        for i in range(min(features.shape[-1], 16)): 
            feature_map = features[:, :, i]
            upsampled = cv2.resize(feature_map, (224, 224))
            upsampled = (upsampled - upsampled.min()) / (upsampled.max() - upsampled.min() + 1e-10)
            
            masked_img = img_input[0] * upsampled[..., np.newaxis]
            dot_product = model.predict(np.expand_dims(masked_img, 0), verbose=0)[0]
            score = dot_product[np.argmax(dot_product)]
            
            heatmap += score * feature_map
            
        heatmap = np.maximum(heatmap, 0)
        heatmap /= (np.max(heatmap) + 1e-10)
        return heatmap
    except Exception as e:
        print(f"Score-CAM Error: {e}")
        return None

def process_heatmap(img_pil, heatmap):
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    img_cv = cv2.resize(img_cv, (224, 224))
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    result = cv2.addWeighted(img_cv, 0.6, heatmap_color, 0.4, 0)
    _, buff = cv2.imencode('.jpg', result)
    return base64.b64encode(buff).decode('utf-8')


MODEL_PATH = "PlantDisease_EfficientNetB0_LKAM.keras"
JSON_PATH = "class_map.json"

model = None
classes = []

try:
   
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'LKAM': LKAM}, compile=False)
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        class_map = json.load(f)
        classes = [class_map[str(i)] for i in range(len(class_map))]
    print("✅ System Ready with Score-CAM")
except Exception as e:
    print(f"❌ Load Error: {e}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
        img_res = image.resize((224, 224))
        img_input = np.expand_dims(np.array(img_res).astype('float32') / 255.0, axis=0)

        preds = model.predict(img_input, verbose=0)
        idx = np.argmax(preds[0])
        
        heatmap = get_score_cam(img_input, model)
        explanation_b64 = process_heatmap(img_res, heatmap) if heatmap is not None else None

        return {
            "status": "success",
            "class": classes[idx],
            "confidence": float(np.max(preds[0])),
            "explanation_image": explanation_b64
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
   
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)