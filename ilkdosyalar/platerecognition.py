import cv2
import numpy as np
import pytesseract
from PIL import Image
import time
import os

def plaka_tani_canli():
    # Info.plist dosyasının varlığını kontrol et
    if not os.path.exists('Info.plist'):
        print("Info.plist dosyası oluşturuluyor...")
        with open('Info.plist', 'w') as f:
            f.write('''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>NSCameraUsageDescription</key>
    <string>Bu uygulama plaka tanıma için kameraya erişim gerektirir</string>
    <key>NSCameraUseContinuityCameraDeviceType</key>
    <true/>
</dict>
</plist>''')

    print("Kamera başlatılıyor...")

    # Mevcut kameraları listele
    kamera_indexleri = []
    index = 0
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            kamera_indexleri.append(index)
        cap.release()
        index += 1

    if not kamera_indexleri:
        print("Hata: Hiç kamera bulunamadı!")
        return

    print(f"Bulunan kameralar: {kamera_indexleri}")

    # İlk kullanılabilir kamerayı seç
    cap = cv2.VideoCapture(kamera_indexleri[0])

    if not cap.isOpened():
        print("Hata: Kamera açılamadı!")
        return

    # Kamera ayarlarını optimize et
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    son_plaka = ""
    son_okuma_zamani = time.time()

    print("Plaka tarama başladı. Çıkmak için 'q' tuşuna basın.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kamera görüntüsü alınamıyor! Yeniden deneniyor...")
            cap.release()
            cap = cv2.VideoCapture(kamera_indexleri[0])
            continue

        # Görüntüyü işle
        gri = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gri = cv2.bilateralFilter(gri, 13, 15, 15)
        kenarlar = cv2.Canny(gri, 30, 200)

        # Konturları bul
        konturlar = cv2.findContours(kenarlar.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        konturlar = konturlar[0] if len(konturlar) == 2 else konturlar[1]
        konturlar = sorted(konturlar, key=cv2.contourArea, reverse=True)[:10]

        plaka_konturu = None
        plaka_yeri = None

        for kontur in konturlar:
            approx = cv2.approxPolyDP(kontur, 10, True)
            if len(approx) == 4:
                # Dikdörtgenin en-boy oranını kontrol et
                x, y, w, h = cv2.boundingRect(kontur)
                aspect_ratio = float(w) / h

                # Türk plakalarının yaklaşık en-boy oranı: 520mm/120mm ≈ 4.33
                # Biraz tolerans ile 2.5-5.5 arasındaki oranları kabul et
                if 2.5 <= aspect_ratio <= 5.5:
                    plaka_konturu = approx
                    plaka_yeri = frame[y:y + h, x:x + w]

                    # Kontur çerçevesini çiz
                    cv2.drawContours(frame, [plaka_konturu], -1, (0, 255, 0), 3)

                    # Her 2 saniyede bir OCR yap
                    if time.time() - son_okuma_zamani > 2:
                        try:
                            # Plaka bölgesini işle
                            gri_plaka = cv2.cvtColor(plaka_yeri, cv2.COLOR_BGR2GRAY)

                            # Adaptif eşikleme uygula
                            ikili = cv2.adaptiveThreshold(
                                gri_plaka,
                                255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY,
                                11,
                                2
                            )

                            # Gürültü temizleme
                            kernel = np.ones((1, 1), np.uint8)
                            ikili = cv2.morphologyEx(ikili, cv2.MORPH_OPEN, kernel)
                            ikili = cv2.morphologyEx(ikili, cv2.MORPH_CLOSE, kernel)

                            # OCR işlemi
                            metin = pytesseract.image_to_string(ikili, config='--psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
                            temiz_metin = ''.join(e for e in metin if e.isalnum())

                            # Yeni bir plaka okunduğunda göster
                            if temiz_metin != son_plaka and len(temiz_metin) > 5:
                                print("\n" + "="*50)
                                print(f"Tespit edilen plaka: {temiz_metin}")
                                print("="*50)
                                son_plaka = temiz_metin

                            son_okuma_zamani = time.time()
                        except Exception as e:
                            print(f"OCR hatası: {str(e)}")
                    break

        # Görüntüleri göster
        cv2.imshow("Canli Plaka Tanima", frame)
        if plaka_yeri is not None:
            cv2.imshow("Plaka Bolgesi", plaka_yeri)

        # 'q' tuşuna basılırsa çık
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Temizle
    cap.release()
    cv2.destroyAllWindows()

def main():
    # Tesseract yolunu ayarla (MacOS için)
    # Eğer brew ile kurduysanız bu satırı uncomment edin:
    # pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'

    try:
        plaka_tani_canli()
    except KeyboardInterrupt:
        print("\nProgram kullanıcı tarafından sonlandırıldı.")
    except Exception as e:
        print(f"\nHata oluştu: {str(e)}")
    finally:
        print("\nProgram sonlandırılıyor...")

if __name__ == "__main__":
    main()
