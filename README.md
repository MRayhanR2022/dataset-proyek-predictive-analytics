# Laporan Proyek Machine Learning - Muhammad Rayhan Rasyad (MC006D5Y1862)

## Domain Proyek

Industri asuransi kesehatan menghadapi tantangan besar dalam menentukan premi yang sesuai untuk setiap pelanggan. Penetapan premi yang tidak akurat dapat menimbulkan risiko finansial, baik bagi perusahaan asuransi maupun bagi nasabah. Oleh karena itu, pendekatan berbasis data dan algoritma machine learning menjadi solusi yang penting dalam membangun sistem prediksi premi asuransi yang lebih akurat dan adil.

Seiring meningkatnya kompleksitas data pelanggan dan kebutuhan akan evaluasi risiko yang lebih presisi, banyak perusahaan asuransi mulai mengadopsi teknologi kecerdasan buatan (AI) untuk mempersonalisasi biaya premi berdasarkan karakteristik individu, seperti usia, indeks massa tubuh (BMI), kebiasaan merokok, jumlah tanggungan, dan lokasi geografis [1]. Dengan bantuan model prediktif, risiko pelanggan dapat dievaluasi secara objektif dan efisien, sehingga meningkatkan profitabilitas dan keadilan dalam proses underwriting.

[1] M. A. Khan, S. A. Khan, and A. A. Khan, “Machine Learning-Based Regression Framework to Predict Health Insurance Premiums,” *Healthcare*, vol. 10, no. 7, pp. 1–15, 2022. https://doi.org/10.3390/healthcare10071288

Dataset yang digunakan dalam proyek ini diambil dari Kaggle: https://www.kaggle.com/datasets/mirichoi0218/insurance

## Business Understanding


### Problem Statements
- Bagaimana memprediksi biaya asuransi seseorang berdasarkan faktor-faktor seperti usia, BMI, jumlah anak, kebiasaan merokok, jenis kelamin, dan wilayah tempat tinggal?
- Algoritma machine learning mana yang menghasilkan prediksi biaya asuransi paling akurat?


### Goals
- Membangun model prediktif biaya asuransi berdasarkan data demografis dan gaya hidup.
- Membandingkan performa beberapa algoritma regresi dan memilih model terbaik.


### Solution Statements
- Menggunakan model baseline Linear Regression untuk memberikan gambaran performa awal.
- Meningkatkan performa prediksi dengan menggunakan Random Forest Regressor dan Gradient Boosting Regressor.
- Mengukur performa setiap model menggunakan metrik MAE untuk memilih model terbaik.



## Data Understanding
Dataset yang digunakan dalam proyek ini merupakan kumpulan data individu di Amerika Serikat yang berisi informasi yang berkaitan dengan premi asuransi kesehatan. Data ini banyak digunakan dalam konteks prediksi biaya klaim asuransi berdasarkan karakteristik demografis dan gaya hidup seseorang. Dataset terdiri dari 1.338 baris data (sampel) dan 7 kolom, yang mencakup:
- 6 fitur (input): karakteristik individu yang mungkin memengaruhi biaya asuransi.
- 1 target (output): biaya asuransi atau `charges`, yaitu jumlah yang harus dibayar individu untuk layanan asuransi kesehatan.


### Fitur:
- `age`: Usia individu (numeric)
- `sex`: Jenis kelamin (male/female)
- `bmi`: Indeks massa tubuh (numeric)
- `children`: Jumlah anak tanggungan (numeric)
- `smoker`: Status merokok (yes/no)
- `region`: Wilayah tempat tinggal (northeast/southeast/southwest/northwest)
- `charges`: Biaya asuransi (target)


### Kondisi data:
Untuk memahami kualitas dan karakteristik data, dilakukan pemeriksaan awal terhadap nilai kosong, data duplikat, dan outlier. Berikut adalah temuan dari eksplorasi tersebut:
- **Missing values:** Tidak ditemukan nilai kosong pada seluruh kolom. Semua entri dalam dataset terisi dengan lengkap, sehingga tidak diperlukan penanganan nilai hilang.
- **Duplikasi:** Ditemukan 1 baris duplikat yang memiliki nilai identik pada seluruh kolom. Keberadaan duplikasi ini menunjukkan bahwa kemungkinan terdapat entri yang terekam lebih dari satu kali dalam proses pengumpulan data.
- **Outlier:** Teridentifikasi keberadaan outlier pada kolom charges dan bmi, berdasarkan distribusi nilai yang terlihat tidak normal dan jangkauan nilai ekstrem. Beberapa individu memiliki biaya asuransi (charges) jauh lebih tinggi dibandingkan mayoritas, terutama pada kelompok perokok dan individu dengan BMI sangat tinggi.


### EDA yang dilakukan:
Exploratory Data Analysis (EDA) dilakukan untuk memahami karakteristik dan hubungan antar variabel dalam dataset. EDA mencakup analisis statistik dasar serta visualisasi data untuk membantu mengidentifikasi pola, anomali, dan relasi antar fitur. Berikut langkah-langkah yang dilakukan:

1. Analisis Univariat (Satu Variabel)
- Fitur numerik (`age`, `bmi`, `children`, `charges`):
    - Digunakan histogram dan boxplot untuk melihat sebaran nilai dan mendeteksi outlier.
    - Ditemukan bahwa `charges` memiliki distribusi skewed (miring ke kanan), menunjukkan bahwa sebagian besar individu memiliki biaya asuransi lebih rendah, sementara sebagian kecil memiliki biaya sangat tinggi.
    - Fitur `bmi` juga menunjukkan nilai ekstrem pada beberapa sampel.

- Fitur kategorikal (`sex`, `smoker`, `region`):
    - Digunakan bar chart untuk melihat proporsi kategori.
    - Terlihat distribusi tidak seimbang, misalnya jumlah perokok jauh lebih sedikit dibandingkan non-perokok.

2. Analisis Multivariat (Hubungan Antar Variabel)
- Korelasi antar fitur numerik dengan `charges`:
    - Menggunakan heatmap korelasi untuk melihat hubungan antar variabel numerik.
    - Fitur `age`, `bmi`, dan `smoker` memiliki korelasi yang cukup kuat dengan `charges`.

- Korelasi antar fitur kategorikal dengan `charges`:
    - Menggunakan bar plot untuk melihat rata-rata charges per kategori.
    - Terlihat bahwa:
        - Perokok (`smoker`=`yes`) memiliki biaya asuransi jauh lebih tinggi.
        - Wilayah (`region`) dan gender (`sex`) tidak menunjukkan perbedaan signifikan terhadap `charges`.

## Data Preparation
Tahapan ini bertujuan untuk mempersiapkan data agar siap digunakan oleh algoritma machine learning. Seluruh langkah dilakukan secara sistematis berdasarkan karakteristik data yang telah dianalisis sebelumnya.

Langkah-Langkah yang Dilakukan:
1. **Menghapus Duplikasi:** 
Setelah dilakukan pemeriksaan, ditemukan 1 baris data yang memiliki nilai identik pada semua kolom. Baris ini dihapus agar tidak mengganggu proses pelatihan model atau menyebabkan bobot berlebih terhadap sampel tertentu. Menghapus duplikasi membantu menjaga representasi data tetap valid dan tidak bias.

2. **Menangani Outlier:**
Outlier terdeteksi pada kolom charges dan bmi berdasarkan hasil visualisasi dengan boxplot dan analisis distribusi. Nilai-nilai ekstrem yang teridentifikasi dihapus sebaik mungkin agar tidak terdapat Outlier. Sisa Outlier yang ada dibiarkan karena nilainya sudah tidak terlalu ekstrem karena data tersebut merepresentasikan kondisi nyata dan tidak mengganggu distribusi mayoritas. Hal ini membantu model belajar dari variasi yang wajar dalam biaya asuransi.

3. **Encoding Fitur Kategorikal:**
Fitur kategorikal (`sex`, `smoker`, dan `region`) tidak dapat langsung digunakan oleh sebagian besar algoritma machine learning yang hanya menerima input numerik. Oleh karena itu, digunakan teknik One-Hot Encoding untuk mengubah nilai kategorikal menjadi representasi numerik biner (0 atau 1). Setelah itu, tipe data boolean hasil encoding diubah menjadi integer agar lebih sesuai untuk model.

4. **Split Data (Train-Test Split):**
Dataset dibagi menjadi dua bagian: 80% untuk data latih (X_train, y_train) dan 20% untuk data uji (X_test, y_test). Pembagian ini dilakukan menggunakan fungsi train_test_split dari scikit-learn, dengan random_state ditentukan agar hasil pembagian konsisten. Tujuannya adalah untuk menguji performa model pada data yang belum pernah dilihat, sehingga evaluasi menjadi lebih objektif dan realistis.

5. **Scaling Fitur Numerikal pada Dataset Train:**
Fitur numerikal (`age`, `bmi`, dan `children`) memiliki skala yang berbeda-beda, yang dapat memengaruhi performa model. Digunakan `StandardScaler` untuk melakukan standardisasi, sehingga setiap fitur memiliki rata-rata = 0 dan standar deviasi = 1. Scaling ini penting agar model seperti **Linear Regression** dan **Gradient Boosting** dapat bekerja lebih stabil dan optimal.


### Alasan Setiap Tahap:
- **Menghapus duplikasi:** Mencegah bias akibat pengulangan data yang tidak disengaja.
- **Outlier:** Dihapus sebaik mungkin dan sisanya dibiarkan untuk merepresentasikan kasus nyata yang relevan secara domain asuransi.
- **Encoding:** Algoritma machine learning tidak dapat membaca data non-numerik, sehingga perlu diubah.
- **Split data:** Penting untuk menghindari overfitting dan mengukur kemampuan generalisasi model pada data nyata.
- **Scaling:** Model sensitif terhadap skala fitur numerik; scaling mempercepat dan menstabilkan pelatihan.



## Modeling
Tiga model yang digunakan:

1. **Linear Regression**
Linear Regression adalah salah satu metode paling sederhana dan paling dikenal dalam machine learning. Model ini digunakan sebagai baseline atau titik awal pembanding dengan model lain yang lebih kompleks.

- Cara kerja: Model ini mencoba menggambar garis lurus terbaik yang merepresentasikan hubungan antara fitur (seperti usia, BMI, dll.) dengan target (charges). Model ini mengasumsikan bahwa hubungan antara semua fitur dengan target bersifat linear atau lurus.

- Kelebihan:
    - Mudah dipahami dan dijelaskan.
    - Cepat saat dijalankan, bahkan untuk dataset yang cukup besar.

- Kekurangan:
    - Kurang akurat jika data memiliki pola hubungan yang tidak lurus (non-linear).


2. **Random Forest Regressor**
Random Forest adalah model yang terdiri dari banyak pohon keputusan (decision trees) yang digabung menjadi satu. Model ini memiliki parameter yang disesuaikan, seperti:
- `n_estimators=50`: jumlah pohon yang digunakan.
- `max_depth=16`: kedalaman maksimal setiap pohon untuk mencegah overfitting.

- Cara kerja: Setiap pohon mencoba membuat prediksi berdasarkan bagian kecil dari data. Hasil akhir dari model merupakan rata-rata dari semua pohon, sehingga lebih stabil dan akurat. Model ini tidak mengasumsikan adanya hubungan linear, sehingga bisa menangkap pola yang kompleks.

- Kelebihan:
    - Sangat baik dalam menangani data dengan pola yang rumit.
    - Tahan terhadap overfitting karena prediksi berasal dari banyak pohon yang berbeda.

- Kekurangan:
    - Lebih lambat dibanding Linear Regression.
    - Kurang transparan (lebih sulit dijelaskan secara sederhana).


3. **Gradient Boosting Regressor**
Gradient Boosting adalah metode boosting, yaitu teknik di mana model dibangun secara bertahap untuk memperbaiki kesalahan dari model sebelumnya. Parameter yang digunakan dalam proyek ini:
- `n_estimators=100`: Model membangun 100 pohon secara bertahap.
- `learning_rate=0.1`: Mengontrol seberapa besar kontribusi setiap pohon baru terhadap model keseluruhan. Semakin kecil, semakin halus perbaikan model.
- `max_depth=4`: Kedalaman maksimum setiap pohon. Ini membantu membatasi kompleksitas model agar tidak overfitting.

- Cara kerja: Alih-alih membangun semua pohon sekaligus seperti Random Forest, Gradient Boosting membangun pohon satu per satu. Setiap pohon baru mencoba memperbaiki kesalahan yang dibuat oleh pohon sebelumnya.

- Kelebihan:
    - Mampu menghasilkan prediksi yang sangat akurat untuk data tabular.

- Kekurangan:
    - Lebih sensitif terhadap overfitting jika tidak dikontrol dengan parameter yang tepat.
    - Proses pelatihan bisa lebih lambat karena model dibangun secara bertahap.


### Alasan Pemilihan dan Perbandingan
Setiap model di atas diuji menggunakan metrik MAE (Mean Absolute Error) untuk menilai seberapa besar kesalahan rata-rata prediksi mereka. Hasilnya:
- **Random Forest** memberikan hasil prediksi terbaik dengan MAE paling kecil dan stabil, baik di data latih maupun data uji.
- **Linear Regression** cenderung kurang akurat karena hubungan antar variabel tidak sepenuhnya linear.
- **Gradient Boosting** cukup akurat, tetapi tidak sebaik Random Forest pada dataset ini.

Oleh karena itu, Random Forest Regressor dipilih sebagai model terbaik untuk memprediksi biaya asuransi.



## Evaluation


### Metrik Evaluasi
Untuk mengukur performa model dalam proyek ini, digunakan metrik MAE (Mean Absolute Error) karena:
- MAE menghitung rata-rata kesalahan absolut antara nilai aktual dan prediksi.
- Metrik ini cocok untuk kasus regresi karena memberikan estimasi berapa jauh prediksi rata-rata model dari nilai sebenarnya, dalam satuan asli target (yaitu dolar AS).
- MAE juga tidak terlalu sensitif terhadap outlier, sehingga memberikan penilaian yang lebih stabil terhadap performa model.

Formula MAE:

$$
MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$


### Hasil Evaluasi:

| Model              | MAE (Train) | MAE (Test) |
|--------------------|-------------|------------|
| Linear Regression  | 2478.46     | 2689.48    |
| Random Forest      | 1024.24     | 2396.47    |
| Gradient Boosting  | 1608.51     | 2425.43    |

- Random Forest Regressor menghasilkan MAE paling rendah pada data pelatihan dan juga cukup kompetitif pada data pengujian.
- Linear Regression, meskipun sederhana, menghasilkan kesalahan prediksi yang lebih tinggi dibanding model lain.
- Gradient Boosting memiliki performa mendekati Random Forest, namun tidak sebaik Random Forest dalam konteks dataset ini.


### Analisis Hasil dengan Bisnis Understanding:
Dari hasil di atas, dapat disimpulkan bahwa:

**Problem Statement**
- Problem Statement 1: Bagaimana memprediksi biaya asuransi seseorang berdasarkan berbagai faktor? 
Model berhasil dibangun dan mampu memprediksi biaya asuransi dengan tingkat kesalahan rata-rata yang dapat diterima (sekitar $2.396).
- Problem Statement 2: Algoritma machine learning mana yang menghasilkan prediksi paling akurat? 
Random Forest Regressor terbukti menjadi model dengan performa terbaik berdasarkan evaluasi metrik MAE.

**Goals**
- Goal 1: Membangun model prediktif biaya asuransi berdasarkan data demografis dan gaya hidup. 
Model telah dibangun dan diuji, dengan hasil evaluasi yang mendukung penggunaannya dalam konteks prediktif.
- Goal 2: Membandingkan performa beberapa algoritma regresi dan memilih model terbaik. 
Tiga model diuji secara adil, dan Random Forest dipilih berdasarkan hasil evaluasi kuantitatif.

Solution Statement:
- Baseline model Linear Regression digunakan sebagai pembanding awal. 
Memberikan gambaran dasar performa.
- Random Forest dan Gradient Boosting digunakan untuk meningkatkan performa. 
Kedua model meningkatkan akurasi, dan Random Forest secara konsisten menghasilkan MAE lebih rendah.
- Metrik MAE digunakan untuk mengevaluasi performa model secara objektif. 
Metrik ini telah menunjukkan keefektifan pendekatan yang diambil.

## Kesimpulan
Proyek ini berhasil membangun sistem prediksi biaya asuransi kesehatan menggunakan pendekatan machine learning. Tiga algoritma regresi diuji: Linear Regression, Random Forest Regressor, dan Gradient Boosting Regressor. Berdasarkan metrik MAE, Random Forest Regressor menunjukkan performa terbaik dengan kesalahan rata-rata yang paling kecil dan stabil pada data uji. Secara bisnis, model ini memiliki dampak penting:
- Efisiensi: Memungkinkan perusahaan asuransi untuk secara otomatis dan akurat memprediksi biaya klaim.
- Keadilan: Penetapan premi dapat dipersonalisasi secara adil berdasarkan profil risiko individu.
- Pengambilan Keputusan: Memberikan dasar yang lebih kuat untuk pengambilan keputusan dalam proses underwriting.

Dengan model ini, perusahaan asuransi dapat meningkatkan akurasi prediksi premi, mengurangi risiko kerugian, dan memberikan layanan yang lebih transparan dan kompetitif kepada pelanggan.
