# Kullanım Rehberi — Phase 1 doğrulama

Bu rehber `volley-analytics` Phase 1 pipeline'ını **kendi maç videon
üzerinde uçtan uca çalıştırmak için** adım adım yönergelerdir. Kod tarafı
tamam, eksik olan tek şey gerçek bir video + modellerle ilk geçişi
yapmak ve çıktıları doğrulamak.

> Tahmini süre: kurulum **45 dk**, ilk geçiş **30–60 dk** (video uzunluğu + GPU'ya göre).

---

## 0. Önkoşullar

| Şey | Neden | Notlar |
|---|---|---|
| Python 3.10+ | Pipeline | 3.11 önerilir |
| `ffmpeg` PATH'te | Ralli klipleri kesme | macOS: `brew install ffmpeg`, Ubuntu: `sudo apt install ffmpeg`, Windows: https://ffmpeg.org/download.html → PATH'e ekle |
| `git` | Repoyu indirmek | — |
| GPU (opsiyonel) | Top takibi 10–20× hızlanır | CUDA destekli NVIDIA + uyumlu `onnxruntime-gpu` varsa otomatik kullanılır |
| Disk ~5 GB | Modeller + ara dosyalar | Video boyutuna göre artar |

> **Önemli**: `01_calibrate` adımı bir OpenCV penceresi açar (4 köşeye
> tıklayacaksın). Bunu **masaüstünde** çalıştır — uzak sunucu / Colab
> üzerinde çalışmaz.

---

## 1. Repoyu hazırla

```bash
git clone https://github.com/utkugunce/volley-analytics.git
cd volley-analytics

# Sanal ortam (önemli — sistem Python'unu kirletme)
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Çekirdek + takip ekstraları
pip install -e ".[tracking]"
```

Doğrula:

```bash
pytest -q                 # 27 test, hepsi geçmeli
python -c "import volley_analytics; print(volley_analytics.__version__)"
```

---

## 2. Model dosyalarını yerleştir

İki model gerekecek: bir top için (ONNX), bir oyuncu için (YOLO).

### 2a. Top modeli (manuel indirme)

Hedef dosya: `models/ball.onnx`

```bash
mkdir -p models
```

Şu sayfayı tarayıcıda aç:
**https://github.com/asigatchov/fast-volleyball-tracking-inference/releases**

İndir → `VballNetV1_seq9_grayscale_330_h288_w512.onnx` (veya en yakın
seq9-grayscale varyantı). `models/ball.onnx` olarak kaydet (veya
yeniden adlandır):

```bash
mv ~/Downloads/VballNetV1_seq9_grayscale_330_h288_w512.onnx models/ball.onnx
```

> Eğer release sayfasında dosya yoksa, repo'nun README'sindeki güncel
> linke bak (Hugging Face barındırma olabilir).

### 2b. Oyuncu modeli (otomatik iniş)

Hiçbir şey yapmana gerek yok. `ultralytics` ilk çalıştırmada
`yolov8n.pt`'yi (~6 MB) otomatik indirir ve mevcut çalışma dizinine
koyar. İnternet bağlantısı yeterli.

İlerideki maçlar için daha doğru tespit istersen `yolov8s.pt` veya
`yolov8m.pt` ile değiştirebilirsin:

```bash
python scripts/02_track.py --player-model yolov8m.pt ...
```

---

## 3. Test videosu hazırla

`data/raw/match.mp4` yoluna **kısa bir test videosu** koy. İlk denemede
**1–3 dakikalık** bir kesit kullan; tüm maç değil.

```bash
mkdir -p data/raw
cp ~/path/to/your_video.mp4 data/raw/match.mp4

# Eğer uzun maç videosundan 2 dakikalık parça çıkarmak istersen:
ffmpeg -ss 00:05:00 -i full_match.mp4 -t 00:02:00 -c copy data/raw/match.mp4
```

**Kaliteli sonuç için video kontrol listesi:**

- [ ] 1080p veya üzeri çözünürlük
- [ ] 30 FPS veya daha yüksek
- [ ] Sabit kamera (tripod) — sallanan çekim tespiti bozar
- [ ] Yüksek noktadan çekilmiş (tribün, balkon) — yere yakın el kamerası **kötü**
- [ ] Tüm saha kadrajda — net hattı + iki endline görünüyor
- [ ] İyi ışık — flou / loş salonlarda top tespiti zayıflar

Kötü kaynak → kötü sonuç. Bu, daha fazla kodla telafi edilemez.

---

## 4. Adım 1 — Sahayı kalibre et

```bash
python scripts/01_calibrate.py \
    --video data/raw/match.mp4 \
    --out   data/interim/calibration.json
```

Bir OpenCV penceresi açılır, ilk kareyi gösterir. **Tam sırayla** 4 köşeye tıkla:

```
        net (uzak taraf)
 ┌──────────────────────┐
 │ TL(2)            TR(3)│   ← uzak endline
 │                       │
 │        SAHA           │
 │                       │
 │ BL(1)            BR(4)│   ← yakın endline (kameraya en yakın)
 └──────────────────────┘
        kamera
```

Sıra **TL → TR → BR → BL** (saat yönünde, sol üstten). Yanlış sıra
homografiyi alt-üst eder; ısı haritası bozuk çıkar.

> İpucu: zoom için tıklamadan önce pencere köşesini sürükle, daha
> büyük göster. Endline (uç çizgi) ile sidline (yan çizgi)
> kesişimlerine tam tıkla — birkaç piksel sapma kabul edilebilir, 20+
> piksel sapma kabul edilemez.

Kaydedince çıkış:
```
Saved calibration → data/interim/calibration.json
```

Kontrol: dosya 4 piksel köşesi + 4 saha köşesi (metre) + 3×3 homografi
matrisini içermeli.

---

## 5. Adım 2 — Top ve oyuncu takibi

```bash
python scripts/02_track.py \
    --video data/raw/match.mp4 \
    --ball-model models/ball.onnx
```

İki CSV çıkar:
- `data/interim/ball.csv` — her tespit edilen kare için top piksel koordinatları
- `data/interim/players.csv` — her oyuncu kutusu (kalıcı `track_id` ile)

**Süre tahmini** (2 dakikalık 1080p30 video için):
- CPU ile: 5–15 dakika
- GPU ile: 30–90 saniye

İlk birkaç saniye `yolov8n.pt`'yi indirmek için harcanır (sadece bir kez).

**Bekleyebileceğin satır sayıları:**
- `ball.csv`: video kare sayısının %50–90'ı (top her karede görünmez — bloklarda, oyun-dışı anlarda yok)
- `players.csv`: kare sayısı × ortalama oyuncu sayısı (genelde 10–12 → kareyenoranı 10–12x)

Hızlı kontrol:
```bash
wc -l data/interim/ball.csv data/interim/players.csv
head -5 data/interim/ball.csv
```

> **Sadece top istiyorsan**: `--skip-players` ekle (3× hızlandırır).
> **Sadece oyuncu istiyorsan**: `--skip-ball` (ONNX modelinin olmasını gerektirmez).

---

## 6. Adım 3 — Top ısı haritası

```bash
python scripts/03_heatmap.py \
    --ball        data/interim/ball.csv \
    --calibration data/interim/calibration.json \
    --out         data/out/ball_heatmap.png \
    --title       "Top yörüngesi — Maç 1"
```

`data/out/ball_heatmap.png` üretilir. Aç ve şuna bak:

**İyi sonuç şöyle görünür:**
- Aktivite **net hattı çevresinde (y ≈ 9 m)** yoğunlaşır — smaç/blok pozisyonu
- **3 m hücum çizgilerinde** ikinci yoğunluk — pas/set yüksekliği
- Endline'lara doğru seyrelir — servis konumu
- Tüm dağılım saha içinde, dışarı taşmaz

**Kötü sonuç** belirtileri ve nedenleri:
| Belirti | Olası neden | Çözüm |
|---|---|---|
| Isı dağılımı saha dışına taşıyor | Kalibrasyon kötü tıklanmış | `01_calibrate`'i tekrar koş |
| Tek bir noktada parlak leke | Top modeli sürekli aynı yere takılmış (yanlış pozitif) | `--min-conf 0.5` deneyin |
| Neredeyse tümden boş | Top hiç tespit edilmemiş | Video kalitesi düşük olabilir, ya da model çözünürlüğü uymuyor |
| Saha kenarlarına sıkışmış | Köşe sırası TL/TR/BR/BL değildi | Kalibrasyonu yenile |

Düşük güvenli tespitleri ele:
```bash
python scripts/03_heatmap.py --ball ... --calibration ... --out ... --min-conf 0.5
```

---

## 7. Adım 4 — Ralli klipleri

```bash
python scripts/04_rally_clips.py \
    --video   data/raw/match.mp4 \
    --ball    data/interim/ball.csv \
    --out-dir data/out/rallies
```

`data/out/rallies/rally_001.mp4`, `rally_002.mp4`, … şeklinde dosyalar
çıkar. Her ralli ayrı bir mp4.

Çıktı şuna benzer:
```
  rally 001:    3.21s →   18.47s  (rally_001.mp4)
  rally 002:   24.13s →   41.85s  (rally_002.mp4)
  ...
Saved 12 rally clips → data/out/rallies
```

`ffmpeg -c copy` kullanılır — birkaç saniyede biter, kalite kaybı yok.

**Ayarlama**:
- Çok kısa "ralli"ler (1-2 saniye) çıkıyorsa: `--min-rally-frames 90` (3 saniyeden kısa olanları at)
- İki ayrı ralliyi tek olarak birleştiriyorsa: `--gap-frames 30` (daha agresif ayır)
- Klipler çok dar görünüyorsa: `--pad 2.0` (önce/sonra 2'şer saniye ekstra)

Default değerler 30 FPS için makul:
- `--gap-frames 45` (~1.5 sn topsuz boşluk = yeni ralli)
- `--min-rally-frames 60` (~2 sn'den kısa rallileri at)
- `--pad 1.0` (1 sn pre/post padding)

60 FPS videoda bu değerleri 2× yap.

---

## 8. Doğrulama listesi

Adım adım tamamladıktan sonra şu kontrolleri yap:

- [ ] `data/interim/calibration.json` mevcut ve 3 alan (`H`, `image_corners`, `court_corners`) içeriyor
- [ ] `data/interim/ball.csv` 100+ satır (3-dk videodan beklenir)
- [ ] `data/interim/players.csv` 1000+ satır
- [ ] `data/out/ball_heatmap.png` görsel olarak makul (yukarıdaki tabloya bak)
- [ ] `data/out/rallies/` içinde 5+ mp4 dosyası var
- [ ] Rastgele 1 ralli klibini aç — gerçek bir rallinin başından sonuna kadar olmalı

Her şey ✅ ise Phase 1 başarılı.

---

## 9. Sık karşılaşılan sorunlar

### `ModuleNotFoundError: No module named 'onnxruntime'`
```bash
pip install -e ".[tracking]"
```

### `ffmpeg: command not found`
Kurulum: `brew install ffmpeg` / `sudo apt install ffmpeg` / Windows için
https://www.gyan.dev/ffmpeg/builds/ — PATH'e ekle, terminali yeniden aç.

### `cv2.error: ... findHomography ...`
Kalibrasyon penceresinde **4 köşeye tıklamadan ESC bastın**. Tekrar koş,
4 tıklama yap.

### Top hiç tespit edilmiyor (`ball.csv` boş veya çok kısa)
Olası nedenler:
1. Model dosyası bozuk — yeniden indir
2. Video çözünürlüğü çok düşük (480p altı) — daha yüksek kaynak gerekli
3. Top renk/zemin kontrastı çok düşük — bu modelin sınırı

### Oyuncu takibi çok yavaş (CPU)
- `yolov8n.pt` zaten en hafif. Daha hızlısı yok.
- GPU kullanmak için: `pip install onnxruntime-gpu` (top için) +
  PyTorch CUDA sürümü (oyuncu için). Detaylar:
  https://docs.ultralytics.com/integrations/

### Aynı oyuncu iki track_id alıyor (ByteTrack ID flip)
- Üst üste binme veya hızlı hareket sonrası beklenir. Şimdilik tolere et.
- Phase 2'de rotasyon takibine girdiğinde, ID kararlılığını jersey
  numarası OCR veya SAM 3 takım promptu ile güçlendireceğiz.

### Isı haritası saha dışına taşıyor
Kalibrasyon köşe sırası yanlış. `01_calibrate`'i yeniden koş, sırayı
**TL → TR → BR → BL** (saat yönünde) gözet.

---

## 10. Sonra ne?

İlk geçiş başarılı olduktan sonra geri dön ve sonuçları paylaş.
Aşağıdakilerden bir veya birkaçı sıradaki adım olur:

1. **Oyuncu ısı haritası** — `heatmap.py`'yi `players.csv` ile beslemek
   (foot point ile saha izdüşümü). Takım dizilim/tercih analizine kapı açar.
2. **SAM 3 takım ayrımı** — `players.csv`'ye `team_id` kolonu;
   "beyazlı oyuncu" / "kırmızılı oyuncu" promptlarıyla. Jersey rengi
   karışıyorsa veya naive renk kümeleme yetmiyorsa.
3. **Smoothing / outlier filtreleme** — `ball.csv`'de tek-kare sıçramaları
   median filtre ile temizle, yörünge görselleştirmesi daha temiz olur.
4. **Phase 2 başlat** — başka bir repoda `players.csv` + kalibrasyondan
   formasyon / saha tercihi / dönüş analizi.

Hangisi öncelikli, gerçek sonuçlara bakınca netleşir.

---

## Hızlı referans — tüm pipeline tek seferde

İlk geçişi tamamladıktan sonra hatırlatma için:

```bash
source .venv/bin/activate

python scripts/01_calibrate.py --video data/raw/match.mp4 --out data/interim/calibration.json

python scripts/02_track.py --video data/raw/match.mp4 --ball-model models/ball.onnx

python scripts/03_heatmap.py \
    --ball data/interim/ball.csv \
    --calibration data/interim/calibration.json \
    --out data/out/ball_heatmap.png

python scripts/04_rally_clips.py \
    --video data/raw/match.mp4 \
    --ball data/interim/ball.csv \
    --out-dir data/out/rallies
```
