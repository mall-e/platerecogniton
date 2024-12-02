import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pytesseract
from PIL import Image
import argparse

# Model sınıfı aynı
class PlateDetectionModel(nn.Module):
    def __init__(self):
        super(PlateDetectionModel, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3), nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 12 * 12, 128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def preprocess_plate(plate_img):
    """Plaka görüntüsünü OCR için hazırla"""
    # Griye çevir
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    # Gürültü azaltma
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Adaptif eşikleme
    binary = cv2.adaptiveThreshold(blur, 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 2)

    # Morfolojik işlemler
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return cleaned

def read_plate(plate_img):
    """Plaka metnini oku"""
    # Görüntüyü hazırla
    processed_img = preprocess_plate(plate_img)

    # OCR yapılandırması
    custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    # OCR uygula
    text = pytesseract.image_to_string(processed_img, config=custom_config)

    # Temizle
    text = ''.join(c for c in text if c.isalnum())

    return text

def detect_and_read_plate(image_path, model_path, device):
    # Model yükleme
    model = PlateDetectionModel().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Görüntü yükleme ve ön işleme
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Görüntü yüklenemedi: {image_path}")

    original_img = img.copy()
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = torch.FloatTensor(img).permute(2, 0, 1).unsqueeze(0).to(device)

    # Plaka konumunu tahmin et
    with torch.no_grad():
        coords = model(img).cpu().numpy()[0]

    # Koordinatları orijinal görüntü boyutuna dönüştür
    h, w = original_img.shape[:2]
    x1, y1, x2, y2 = coords
    x1 = max(0, int(x1 * w))
    x2 = min(w, int(x2 * w))
    y1 = max(0, int(y1 * h))
    y2 = min(h, int(y2 * h))

    # Plaka bölgesini kes
    plate_img = original_img[y1:y2, x1:x2]

    # Plaka metnini oku
    plate_text = read_plate(plate_img)

    # Sonuçları görselleştir
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
    plt.gca().add_patch(rect)
    plt.title(f"Tespit Edilen Plaka: {plate_text}")
    plt.axis('off')
    plt.show()

    return (x1, y1, x2, y2), plate_text

def main():
    # Argüman parser'ı oluştur
    parser = argparse.ArgumentParser(description='Plaka tespiti ve okuma')
    parser.add_argument('--image', type=str, required=True,
                      help='Test edilecek görüntünün dosya yolu')
    parser.add_argument('--model', type=str,
                      default='best_model_20241202_175448.pth',
                      help='Kullanılacak model dosyasının yolu')
    args = parser.parse_args()

    # GPU kullanılabilirse kullan, yoksa CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Kullanılan cihaz: {device}")

    try:
        # Plaka tespiti ve okuma yap
        coords, plate_text = detect_and_read_plate(args.image, args.model, device)
        print(f"Test edilen görüntü: {args.image}")
        print(f"Tespit edilen plaka koordinatları: {coords}")
        print(f"Okunan plaka: {plate_text}")
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")

if __name__ == "__main__":
    main()
