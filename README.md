# GenRap: Stil Bazlı Türkçe Rap Sözü Üretici

Bu projede, NLP kullanarak kullanıcının istediği bir türe (agresif, melankolik, flex vb.) 
uygun, orijinal rap türünde şarkı sözleri üretmeyi amaçlanmaktadır.

## 🎯Projenin Hedefleri
Proje 3 ana modülden oluşmaktadır:

1.  **Veri Toplama:** `lyricsgenius` kütüphanesi kullanılarak Genius API üzerinden çeşitli Türk rap sanatçılarının şarkı sözlerinin toplanması.
2.  **Veri İşleme ve Etiketleme:** Toplanan ham `.txt` verilerinin, her şarkı için manuel olarak `[STİL]` etiketleriyle (`[AGRESİF]`, `[BATTLE]` vb.) etiketlenip `train.txt` adında tek bir eğitim dosyası haline getirilmesi.
3.  **Model Eğitimi (Fine-Tuning):** Ön-eğitimli bir Türkçe dil modelinin, hazırlanan `train.txt` dosyası ile ince ayar (fine-tuning) yapılarak stil bazlı metin üretmesinin sağlanması.

## ⚠️ Sınırlamalar ve Veri Uyarısı (Limitations and Data Warning)

Bu model, çok çeşitli sanatçılardan toplanan gerçek dünya rap şarkı sözleri üzerinde eğitilmiştir. Rap müzik türü, doğası gereği toplumsal eleştiri, protesto, argo dil (`profanity`) ve zaman zaman saldırgan veya tartışmalı temalar içerebilmektedir.

**"Veri Neyse, Model Odur" (Data In, Model Out) ilkesi gereği:**

* Modelin ürettiği şarkı sözleri, orijinal veri setindeki bu argo ve tartışmalı ifadeleri **taklit edebilir** ve yansıtabilir.
* Üretilen metinler, geliştiricinin kişisel görüşlerini **yansıtmaz** ve toplumsal normlara uygun olmayabilir.

Bu modelin çıktıları, bu riskler göz önünde bulundurularak ve sorumlu bir şekilde kullanılmalıdır.

### Deney Sonuçları Tablosu

| Deney Kodu     | `num_train_epochs` (Tur) | `per_device_train_batch_size` (Paket) | Eğitim Süresi (Yaklaşık) | Son `Training Loss` Değeri                                                       |
|:---------------|:-------------------------|:--------------------------------------|:-------------------------|:---------------------------------------------------------------------------------|
| **v1 (Temel)** | 3                        | 2                                     | ~1 dakika  36 saniye     | **4.59** -> **4.24** -> **3.89** -> **3.60**                                     | 
| **v2**         | 8                        | 4                                     | ~2 dakika  56 saniye     | **4.43** -> **3.81** -> **3.41** -> **3.14** -> **2.85** -> **2.64**             |
| **v3**         | 8                        | 4                                     | ~3 dakika  44 saniye     | **4.43** -> **3.81** -> **3.39** -> **3.10** -> **2.75** -> **2.50** -> **2.36** |
