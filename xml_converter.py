import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import shutil

def xml_to_yolo(xml_path, image_path):
    """
    XML dosyasından plaka koordinatlarını okur ve normalize eder
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Resim boyutlarını al
    img = cv2.imread(image_path)
    if img is None:
        print(f"Hata: {image_path} okunamadı!")
        return None

    img_height, img_width = img.shape[:2]

    # Plaka koordinatlarını bul
    for obj in root.findall('.//object'):
        if obj.find('name').text.lower() in ['plate', 'licence', 'license']:
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)

            # Koordinatları normalize et
            x1 = xmin / img_width
            y1 = ymin / img_height
            x2 = xmax / img_width
            y2 = ymax / img_height

            return [x1, y1, x2, y2]

    return None

def prepare_dataset(image_dir, annotation_dir, output_dir):
    """
    Veri setini hazırlar ve train-val olarak böler
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(os.path.join(output_dir, 'train'))
        os.makedirs(os.path.join(output_dir, 'val'))

    data_list = []
    skipped_files = []

    print("Dosyalar işleniyor...")

    # Tüm görüntüleri işle
    for img_name in os.listdir(image_dir):
        if img_name.endswith(('.png', '.jpg', '.jpeg')):
            base_name = os.path.splitext(img_name)[0]
            xml_path = os.path.join(annotation_dir, base_name + '.xml')
            img_path = os.path.join(image_dir, img_name)

            if os.path.exists(xml_path):
                coords = xml_to_yolo(xml_path, img_path)
                if coords:
                    data_list.append((img_path, coords))
                else:
                    skipped_files.append(img_name)
            else:
                skipped_files.append(img_name)

            # İlerleme göster
            if len(data_list) % 100 == 0 and len(data_list) > 0:
                print(f"{len(data_list)} dosya işlendi...")

    if not data_list:
        print("Hata: Hiç geçerli veri bulunamadı!")
        return 0, 0

    # Veriyi train ve validation olarak böl
    train_data, val_data = train_test_split(data_list, test_size=0.2, random_state=42)

    print("\nDosyalar kopyalanıyor...")

    # Train verilerini kaydet
    for img_path, coords in train_data:
        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]

        # Resmi kopyala
        shutil.copy2(img_path, os.path.join(output_dir, 'train', img_name))

        # Koordinatları kaydet
        with open(os.path.join(output_dir, 'train', base_name + '.txt'), 'w') as f:
            f.write(','.join(map(str, coords)))

    # Validation verilerini kaydet
    for img_path, coords in val_data:
        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]

        # Resmi kopyala
        shutil.copy2(img_path, os.path.join(output_dir, 'val', img_name))

        # Koordinatları kaydet
        with open(os.path.join(output_dir, 'val', base_name + '.txt'), 'w') as f:
            f.write(','.join(map(str, coords)))

    if skipped_files:
        print("\nAtlanan dosyalar:")
        for file in skipped_files[:10]:  # İlk 10 dosyayı göster
            print(f"- {file}")
        if len(skipped_files) > 10:
            print(f"... ve {len(skipped_files) - 10} dosya daha")

    return len(train_data), len(val_data)

def main():
    # Klasör yollarını belirle
    image_dir = "veri_seti/images"  # Görüntülerin bulunduğu klasör
    annotation_dir = "veri_seti/annotations"  # XML dosyalarının bulunduğu klasör
    output_dir = "prepared_dataset"  # İşlenmiş veri setinin kaydedileceği klasör

    print("Veri seti hazırlanıyor...")
    print(f"Görüntü klasörü: {image_dir}")
    print(f"Annotation klasörü: {annotation_dir}")

    train_count, val_count = prepare_dataset(image_dir, annotation_dir, output_dir)

    if train_count > 0 or val_count > 0:
        print("\nVeri seti hazırlama tamamlandı!")
        print(f"Train görüntü sayısı: {train_count}")
        print(f"Validation görüntü sayısı: {val_count}")
        print(f"\nVeriler '{output_dir}' klasörüne kaydedildi.")
    else:
        print("\nHata: Veri seti oluşturulamadı!")

if __name__ == "__main__":
    main()
