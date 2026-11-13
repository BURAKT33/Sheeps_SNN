# Sheeps_SNN

Koyun görüntülerinde birden fazla hayvanı tespit eden spiking neural network (SNN) dedektörü. Etiketler YOLOv8 biçimindedir; model kareleri spike dizisine çevirip kutu ve sınıf tahmini üretir.

Veri kümesi Roboflow üzerinden dışa aktarılmıştır (CC BY 4.0, 525 görüntü, 576×576). Kaynak sınıflar `drinking`, `eating`, `lying` ve `stand`dır. Eğitimde bunlar üç model sınıfına indirilir:

| Kaynak etiket | Model sınıfı |
| --- | --- |
| `lying` | `sit` |
| `stand` | `stand` |
| `eating`, `drinking` | `eat` |

## Kurulum

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch, SpikingJelly ve OpenCV gerekir. GPU varsa eğitim ve çıkarımı `--device cuda` ile çalıştırın; varsayılan aygıt CPU'dur.

## Eğitim

Görüntüler `data/train` ve `data/valid` altındadır. En iyi ağırlıklar `outputs/checkpoints/best.pt` dosyasına yazılır.

```bash
python3 train.py --epochs 30 --batch-size 8 --device cpu
```

Diğer bayraklar: `--lr`, `--image-size`, `--time-steps`, `--num-workers`.

## Çıkarım

Tek görüntü:

```bash
python3 predict_image.py --image data/test/images/ornek.jpg --checkpoint outputs/checkpoints/best.pt
```

RTSP, web kamerası veya video dosyası:

```bash
python3 inference_rtsp.py --source video.mp4 --checkpoint outputs/checkpoints/best.pt
```

`--source` verilmezse `config.py` içindeki RTSP adresi kullanılır. Önizlemeyi kapatmak için `--no-show` ekleyin. Pencere açıkken `q` çıkarımı durdurur.

## Dosyalar

| Dosya | Görevi |
| --- | --- |
| `train.py` | Eğitim döngüsü ve `best.pt` kaydı |
| `predict_image.py` | Tek görüntüde kutu çizimi |
| `inference_rtsp.py` | Canlı veya video çıkarımı |
| `encoder.py` | Kareden spike dizisi |
| `models/snn_detector.py` | Dedektör ağı |
| `dataset.py` | YOLO etiketlerini yükleme |
| `config.py` | Hiperparametreler ve sınıf eşlemesi |
| `camera.py` | Son kareyi tutan akış |
