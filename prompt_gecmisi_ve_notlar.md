# Proje Geçmişi, Önemli Promptlar ve Gelişim Notları

## 📊 Proje Özeti
Bu proje, Borsa İstanbul (BİST) şirketlerinin finansal bilançolarını ve Türkiye'nin makroekonomik verilerini kullanarak bir **Erken Uyarı Sistemi (İflas Tahmin Modeli)** geliştirmeyi amaçlamaktadır. Proje süresince model mimarisi 5 temel aşamadan geçerek "dürüst ve bilimsel" bir seviyeye taşınmıştır.

## 🚀 Gelişim Aşamaları (Sürümler)

### 1. Sürüm v1-v2: Temel Kurulum
- BİST şirketleri için Excel tabanlı veri yükleme sistemi kuruldu.
- Temel finansal oranlar (ROA, Borç Oranı vb.) ve makro veriler (Enflasyon, GSYİH) birleştirildi.

### 2. Sürüm v3: Etkileşim Terimleri (Interaction Features)
- **Kritik Prompt:** "featureları tekrardan gözden geçirelim etkilişim olanların birbirini tekrar eden ve modellerde katkısını gözden geçirelim"
- Mikro (bilanço) ve Makro verilerin birbirini nasıl etkilediği (Örn: Borçluluk × Faiz) 18 etkileşim terimi ile modellendi.
- SHAP analizi ile en etkili terimler belirlendi.

### 3. Sürüm v4: Literatür Bazlı Rasyo Değişimleri
- **Kritik Prompt:** "bu rasyolardan hangileri bizde mevcut... şunları denesek bizdekilerle değiştirip"
- Akademik literatüre dayanarak daha güçlü tahmin gücü olan rasyolar seçildi:
    - ROA yerine **Operating Income / Total Assets**
    - Toplam Borç yerine **Short Term Liabilities / Total Assets**
    - Yeni eklenen: **Gross Profit / Long Term Liabilities**
- Sonuç: AUC 0.89'a yükseldi.

### 4. Sürüm v5: İleri Düzey Metodoloji (V5)
- **Kritik Tartışma:** İflastan çok önceki yılların (Örn: 2018) `1` (iflas) olarak etiketlenmesinin yarattığı "gürültü" (noise) sorunu paylaşıldı.
- **Yenilikler:**
    - **Time-to-Default (T-3):** Sadece iflastan önceki son 3 yıl `1` olarak etiketlendi, eski yıllar temizlendi.
    - **Trend (Momentum) Features:** Rasyoların 1 yıllık değişim hızları eklendi. (İvme takibi)
    - **Lagged Macro:** Makroekonomik verilerin 1 yıl önceki gecikmeli etkisi dahil edildi.

### 5. Sürüm v5.1: Stabilizasyon ve Overfitting Engelleme
- **Kritik Prompt:** "1. seçeneği deneyelim modeli daha doğru ratiolar ile eğitelim ve böylece model içerisindeki stabilitesini arttıralım."
- Veri azlığı (104 iflas satırı) nedeniyle 47 özellikten sadece **SHAP İlk 10**'a giren en güçlü özellikler seçildi.

## 📝 Önemli Kararlar ve Notlar
- **Labeling Stratejisi:** Modelin "teşhis" değil "erken uyarı" yapması için T-3 sınırlaması getirildi.
- **Algoritma Seçimi:** LightGBM, CatBoost, Random Forest ve XGBoost modelleri 10-Fold CV ile karşılaştırıldı.
- **Trend Analizi:** Şirketin o anki durumundan ziyade "borç ivmesi" ve "kârlılık erimesi"nin iflası tetikleyen ana faktörler olduğu kanıtlandı.

---
*Bu dosya, projenin sadece kod değil, bir "karar verme süreci" olduğunu belgelemek amacıyla oluşturulmuştur.*
