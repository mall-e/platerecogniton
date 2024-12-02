import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np

class PlakaTespitModeli(nn.Module):
    def __init__(self):
        super(PlakaTespitModeli, self).__init__()
        # CNN mimarisi
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 4)  # 4 koordinat için (x1, y1, x2, y2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class KarakterTanimaModeli(nn.Module):
    def __init__(self, num_chars=36):  # 26 harf + 10 rakam
        super(KarakterTanimaModeli, self).__init__()
        # CNN + LSTM mimarisi
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.lstm = nn.LSTM(64 * 8, 128, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(256, num_chars)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        batch_size, channels, height, width = x.size()
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch_size, width, channels * height)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

def plaka_tanima_canli():
    # Modelleri yükle
    plaka_model = PlakaTespitModeli()
    karakter_model = KarakterTanimaModeli()

    # Eğer önceden eğitilmiş modeller varsa
    # plaka_model.load_state_dict(torch.load('plaka_model.pth'))
    # karakter_model.load_state_dict(torch.load('karakter_model.pth'))

    # GPU kullanılabilirse
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    plaka_model.to(device)
    karakter_model.to(device)

    # Görüntü önişleme
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Görüntüyü PyTorch tensor'a çevir
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Plaka tespiti
        with torch.no_grad():
            plaka_koordinat = plaka_model(img_tensor)

        # Koordinatları orijinal görüntü boyutuna çevir
        x1, y1, x2, y2 = plaka_koordinat[0].cpu().numpy()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Plaka bölgesini kes
        plaka_bolge = frame[y1:y2, x1:x2]

        if plaka_bolge.size > 0:
            # Karakter tanıma için önişleme
            plaka_gray = cv2.cvtColor(plaka_bolge, cv2.COLOR_BGR2GRAY)
            plaka_tensor = transforms.ToTensor()(plaka_gray).unsqueeze(0).to(device)

            # Karakterleri tanı
            with torch.no_grad():
                karakter_tahmin = karakter_model(plaka_tensor)

            # Tahminleri metne çevir
            karakterler = karakter_tahmini_cozumle(karakter_tahmin)

            # Görüntüye plaka ve metni çiz
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, karakterler, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Derin Ogrenme ile Plaka Tanima', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def karakter_tahmini_cozumle(tahminler):
    # CTC çözümleme veya en yüksek olasılıklı karakterleri seç
    return "34ABC123"  # Örnek çıktı

if __name__ == "__main__":
    plaka_tanima_canli()
