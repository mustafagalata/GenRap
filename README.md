# 🤖 GenRap: Stil Bazlı Türkçe Rap Sözü Üretici
Bu, [PyTorch](https://pytorch.org/) ve [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) kütüphaneleri kullanılarak geliştirilmiş bir Doğal Dil İşleme (NLP) projesidir. Projenin amacı, `ytu-ce-cosmos/turkish-gpt2` modelini, 240+ şarkıdan oluşan özel etiketli bir Türkçe rap veri seti üzerinde "ince ayar" (fine-tuning) yaparak, verilen stil etiketlerine ([AGRESİF], [FLEX], [FELSEFİK] vb.) göre orijinal rap şarkı sözleri üretebilen bir model oluşturmaktır.

---

## 🛠️ Kullanılan Teknolojiler ve Kütüphaneler
###  **Python 3.10+**

###  Derin Öğrenme & NLP:

- `PyTorch`: Modelin eğitimi ve çıkarımı (inference) için temel çerçeve.

- `Hugging Face Transformers`: Ön-eğitimli modeli (`AutoModelForCausalLM`) ve sözlüğü (`AutoTokenizer`) yüklemek için.

- `Hugging Face Datasets`: 'train.txt' dosyasını verimli bir şekilde işlemek için.

### Veri Toplama:

- `Genius API`: Şarkı sözü verisinin kaynağı.

- `lyricsgenius`: Genius API için kullanılan Python sarmalayıcısı (wrapper).

### Eğitim Ortamı:

- `Google Colab` (T4 GPU): Yetersiz lokal VRAM kısıtlamasını aşmak için bulutta GPU ile eğitim.

### Araçlar:

- `Git & GitHub`: Sürüm kontrolü ve portföy sunumu.

- `python-dotenv`: API anahtarlarını güvenli bir şekilde saklamak için.

- `venv`: Proje bağımlılıklarını izole etmek için.

---

## 🔄 Proje İş Akışı (Pipeline)

Bu proje, bir makine öğrenimi projesinin 4 ana adımını takip eder:

### 1. Veri Toplama
* `lyricsgenius` kütüphanesi kullanılarak, belirlenen Türk rap sanatçılarının şarkı sözleri Genius API üzerinden çekildi.
* Her şarkı, `raw_data/` klasörü altına `Sanatçı-ŞarkıAdı.txt` formatında ayrı bir metin dosyası olarak kaydedildi.

### 2. Veri Ön İşleme
* Projenin en kritik adımı. 241 şarkının tamamı, `[AGRESİF]`, `[MELANKOLİK]`, `[FLEX]`, `[BATTLE]` gibi etiketler kullanılarak **manuel olarak etiketlendi.**
* Tüm şarkı sözlerini ve etiketlerini `[ETİKET] [ETİKET] <sözler...>` formatında birleştirdi.
* Modelin eğitimde "bias" (önyargı) geliştirmemesi için `random.shuffle()` kullanılarak veri setinin tamamı **karıştırıldı**.

### 3. Model Eğitimi
* Lokal donanımın yetersizliği nedeniyle eğitim süreci **Google Colab** üzerinde, T4 GPU kullanılarak gerçekleştirildi.
* `ytu-ce-cosmos/turkish-gpt2` modeli, `transformers` kütüphanesinin `Trainer` API'si kullanılarak `train.txt` üzerinde "ince ayar" (fine-tuning) yapıldı.
* En iyi "kayıp" (loss) değerini bulmak için hiperparametre optimizasyonu (deney takibi) yapıldı.
* Eğitim tamamlandıktan sonra, eğitilmiş model dosyaları (`genrap-model` klasörü) lokale indirildi.

### 4. Söz Üretimi
* Eğitilmiş olan `genrap-model` klasörü lokalden yüklendi.
* `model.generate()` fonksiyonu kullanılarak, kullanıcı tarafından sağlanan prompt'a (örn: `[FELSEFİK]`) göre yeni sözler üretildi.

---

## 📈 Eğitim ve Deney Raporu

En iyi modeli bulmak için `num_train_epochs` (tur sayısı) gibi hiperparametreler üzerinde deneyler yapılmıştır.

| Deney Kodu     | `num_train_epochs` (Tur) | `per_device_train_batch_size` (Paket) | Eğitim Süresi (Yaklaşık) | Son `Training Loss` Değeri                                                       |
|:---------------|:-------------------------|:--------------------------------------|:-------------------------|:---------------------------------------------------------------------------------|
| **v1 (Temel)** | 3                        | 2                                     | ~1 dakika  36 saniye     | **4.59** -> **4.24** -> **3.89** -> **3.60**                                     | 
| **v2**         | 8                        | 4                                     | ~2 dakika  56 saniye     | **4.43** -> **3.81** -> **3.41** -> **3.14** -> **2.85** -> **2.64**             |
| **v3 (Final)** | 10                       | 4                                     | ~3 dakika  44 saniye     | **4.43** -> **3.81** -> **3.39** -> **3.10** -> **2.75** -> **2.50** -> **2.36** |


---
## 🚀 Kurulum ve Çalıştırma
**Not:** Bu proje, `.gitignore` ile korunduğu için eğitilmiş model dosyalarını (`genrap-model`), etiketleri veya ham veriyi içermez. Projeyi çalıştırmak için kendi modelinizi bu script'leri kullanarak eğitmeniz gerekmektedir.

**1. Kurulum:**

```bash
# Projeyi klonlayın
git clone https://github.com/mustafagalata/GenRap.git
cd GenRap

# Sanal ortamı kurun ve aktive edin
python -m venv venv
venv\Scripts\activate

# Bağımlılıkları yükleyin
pip install -r requirements.txt

# API anahtarınızı ayarlayın
# Proje ana dizininde .env dosyası oluşturun ve içine şunu ekleyin:
# GENIUS_ACCESS_TOKEN="CLIENT_ACCESS_TOKENINIZI_BURAYA_YAZIN"
```

**2. Proje Akışını Çalıştırma:**
```bash
# 1. Ham veriyi toplayın
python data_collection.py

# 2. Veri işleme şablonunu (data_processing_template.py)
#    doldurun ve 'data_processing.py' olarak çalıştırın.

python data_processing.py

# 3. Modelinizi Google Colab'de 'model_training.py' script'i ile eğitin
#    ve eğitilmiş 'genrap-model' klasörünü bu dizine indirin.

# 4. Eğitilmiş modelinizle söz üretin!
python generate_lyrics.py
```
---

## ⚠️ Sınırlamalar ve Veri Uyarısı (Limitations and Data Warning)

Bu model, çok çeşitli sanatçılardan toplanan gerçek dünya rap şarkı sözleri üzerinde eğitilmiştir. Rap müzik türü, doğası gereği toplumsal eleştiri, protesto, argo dil (`profanity`) ve zaman zaman saldırgan veya tartışmalı temalar içerebilmektedir.

**"Veri Neyse, Model Odur" (Data In, Model Out) ilkesi gereği:**

* Modelin ürettiği şarkı sözleri, orijinal veri setindeki bu argo ve tartışmalı ifadeleri **taklit edebilir** ve yansıtabilir.
* Üretilen metinler, geliştiricinin kişisel görüşlerini **yansıtmaz** ve toplumsal normlara uygun olmayabilir.

Bu modelin çıktıları, bu riskler göz önünde bulundurularak ve sorumlu bir şekilde kullanılmalıdır.