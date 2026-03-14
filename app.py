from flask import Flask, render_template, request, jsonify, send_file
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import json
import base64
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# HAM10000 Classes
CLASS_NAMES = {
    'akiec': 'Actinic keratoses',
    'bcc': 'Basal cell carcinoma', 
    'bkl': 'Benign keratosis-like lesions',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic nevi',
    'vasc': 'Vascular lesions'
}

class_to_idx = {cls: idx for idx, cls in enumerate(CLASS_NAMES.keys())}
idx_to_class = {v: k for k, v in class_to_idx.items()}

# Load model
def load_model():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 7)
    
    # Load trained weights if available
    if os.path.exists('best_model.pth'):
        model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
        model.eval()
        print("Model loaded successfully!")
    else:
        print("No trained model found. Using pretrained weights only.")
    
    return model

model = load_model()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def __call__(self, x, class_idx):
        output = self.model(x)
        
        self.model.zero_grad()
        
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        
        output.backward(gradient=one_hot, retain_graph=True)
        
        gradients = self.gradients[0].cpu().numpy()
        activations = self.activations[0].cpu().numpy()
        
        weights = np.mean(gradients, axis=(1, 2))
        
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        cam = np.maximum(cam, 0)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        
        return cam

def generate_grad_cam(image, pred_class):
    grad_cam = GradCAM(model, model.features[-1])
    
    # Convert PIL to tensor
    img_tensor = transform(image).unsqueeze(0)
    
    # Generate Grad-CAM
    cam = grad_cam(img_tensor, pred_class)
    
    # Resize CAM to match image
    cam_resized = torch.nn.functional.interpolate(
        torch.tensor(cam).unsqueeze(0).unsqueeze(0), 
        size=(256, 256),
        mode='bilinear', 
        align_corners=False
    ).squeeze()
    
    # Create heatmap
    heatmap = plt.cm.jet(cam_resized)[:, :, :3]
    
    # Convert original image to numpy
    img_np = np.array(image.resize((256, 256)))
    if len(img_np.shape) == 3:
        img_np = img_np / 255.0
    
    # Overlay
    overlay = 0.6 * img_np + 0.4 * heatmap
    overlay = np.clip(overlay, 0, 1)
    
    # Convert to base64
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_np)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(cam_resized, cmap='jet')
    plt.title('Grad-CAM Heatmap')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title('Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save to buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_base64

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image from request
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Read and process image
        image = Image.open(file.stream).convert('RGB')
        
        # Preprocess
        img_tensor = transform(image).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probabilities, 1)
            pred_class = pred_idx.item()
            confidence_score = confidence.item()
        
        # Get class probabilities
        class_probs = probabilities[0].cpu().numpy()
        results = []
        
        for i, (class_code, class_name) in enumerate(CLASS_NAMES.items()):
            results.append({
                'class': class_name,
                'code': class_code,
                'probability': float(class_probs[i]),
                'percentage': f"{class_probs[i] * 100:.2f}%"
            })
        
        # Sort by probability
        results.sort(key=lambda x: x['probability'], reverse=True)
        
        # Generate Grad-CAM
        try:
            grad_cam_base64 = generate_grad_cam(image, pred_class)
        except Exception as e:
            print(f"Grad-CAM generation failed: {e}")
            grad_cam_base64 = None
        
        # Get original image base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        response = {
            'success': True,
            'prediction': {
                'class': CLASS_NAMES[idx_to_class[pred_class]],
                'code': idx_to_class[pred_class],
                'confidence': f"{confidence_score * 100:.2f}%"
            },
            'all_probabilities': results,
            'original_image': f"data:image/jpeg;base64,{img_base64}",
            'grad_cam': f"data:image/png;base64,{grad_cam_base64}" if grad_cam_base64 else None
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': os.path.exists('best_model.pth')})

if __name__ == '__main__':
    if not os.path.exists('templates'):
        os.makedirs('templates')
    if not os.path.exists('static'):
        os.makedirs('static')
        os.makedirs('static/css')
        os.makedirs('static/js')
    
    app.run(debug=True, host='0.0.0.0', port=5000)
