# Laporan Proyek Machine Learning - Muhammad Rayhan Rasyad (MC006D5Y1862)

## Domain Proyek

Industri asuransi kesehatan menghadapi tantangan besar dalam menentukan premi yang sesuai untuk setiap pelanggan. Penetapan premi yang tidak akurat dapat menimbulkan risiko finansial, baik bagi perusahaan asuransi maupun bagi nasabah. Oleh karena itu, pendekatan berbasis data dan algoritma machine learning menjadi solusi yang penting dalam membangun sistem prediksi premi asuransi yang lebih akurat dan adil.

Seiring meningkatnya kompleksitas data pelanggan dan kebutuhan akan evaluasi risiko yang lebih presisi, banyak perusahaan asuransi mulai mengadopsi teknologi kecerdasan buatan (AI) untuk mempersonalisasi biaya premi berdasarkan karakteristik individu, seperti usia, indeks massa tubuh (BMI), kebiasaan merokok, jumlah tanggungan, dan lokasi geografis [1]. Dengan bantuan model prediktif, risiko pelanggan dapat dievaluasi secara objektif dan efisien, sehingga meningkatkan profitabilitas dan keadilan dalam proses underwriting.

[1] M. A. Khan, S. A. Khan, and A. A. Khan, “Machine Learning-Based Regression Framework to Predict Health Insurance Premiums,” *Healthcare*, vol. 10, no. 7, pp. 1–15, 2022. https://doi.org/10.3390/healthcare10071288



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
Dataset yang digunakan dalam proyek ini diambil dari Kaggle: https://www.kaggle.com/datasets/mirichoi0218/insurance

Dataset yang digunakan dalam proyek ini merupakan kumpulan data individu di Amerika Serikat yang berisi informasi yang berkaitan dengan premi asuransi kesehatan. Data ini banyak digunakan dalam konteks prediksi biaya klaim asuransi berdasarkan karakteristik demografis dan gaya hidup seseorang. Dataset terdiri dari 1.338 baris data (sampel) dan 7 kolom, yang mencakup:
- 6 fitur (input): karakteristik individu yang mungkin memengaruhi biaya asuransi.
- 1 target (output): biaya asuransi atau `charges`, yaitu jumlah yang harus dibayar individu untuk layanan asuransi kesehatan.


### Deskripsi Fitur:
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
- Fitur kategorikal (`sex`, `smoker`, `region`):
    - Digunakan bar chart untuk melihat proporsi kategori.
    - Terlihat distribusi tidak seimbang, misalnya jumlah perokok jauh lebih sedikit dibandingkan non-perokok.

- Fitur numerik (`age`, `bmi`, `children`, `charges`):
    - Digunakan boxplot untuk melihat sebaran nilai.
    - Ditemukan bahwa `age`, `children`, dan `charges`, memiliki distribusi skewed (miring ke kanan).

2. Analisis Multivariat (Hubungan Antar Variabel)
- Korelasi antar fitur kategorikal dengan `charges`:
    - Menggunakan bar plot untuk melihat rata-rata charges per kategori.
    - Terlihat bahwa:
        - Perokok (`smoker`=`yes`) memiliki biaya asuransi jauh lebih tinggi.
        - Wilayah (`region`) dan gender (`sex`) tidak menunjukkan perbedaan signifikan terhadap `charges`.

- Korelasi antar fitur numerik dengan `charges`:
    - Menggunakan heatmap korelasi untuk melihat hubungan antar variabel numerik.
    - Fitur `age`, `bmi`, dan `smoker` memiliki korelasi yang cukup kuat dengan `charges`.



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



## Model Development

### Model 1 — **Linear Regression**

#### Cara Kerja  
Linear Regression adalah algoritma regresi paling sederhana yang berusaha memodelkan hubungan antara variabel input (fitur) dan output (target) dengan sebuah garis lurus. Model ini menghitung koefisien linear dari setiap fitur untuk meminimalkan selisih kuadrat antara nilai aktual dan nilai prediksi.

#### Parameter
Model digunakan dengan parameter **default** dari `sklearn.linear_model.LinearRegression`, yaitu:
- `fit_intercept=True`: model menghitung intercept (titik potong).
- `normalize='deprecated'`: normalisasi tidak dilakukan (fitur telah distandardisasi sebelumnya).
- `n_jobs=None`: pelatihan dilakukan tanpa paralelisasi eksplisit.

#### Kelebihan
- Cepat, sederhana, dan mudah diinterpretasi.
- Cocok sebagai baseline model.

#### Kekurangan
- Kurang akurat jika hubungan antar fitur tidak linear.
- Rentan terhadap outlier.


### Model 2 — **Random Forest Regressor**

#### Cara Kerja  
Random Forest adalah algoritma ensemble yang terdiri dari banyak decision tree. Setiap pohon dibangun dari subset data dan subset fitur (teknik bagging), dan hasil akhir merupakan rata-rata prediksi dari seluruh pohon. Pendekatan ini mengurangi overfitting dan meningkatkan akurasi.

#### Parameter
Model digunakan dengan konfigurasi sebagai berikut:
- `n_estimators=50`: jumlah pohon keputusan yang digunakan.
- `max_depth=16`: kedalaman maksimum setiap pohon untuk mencegah overfitting.
- `random_state=55`: untuk menghasilkan hasil yang konsisten.
- `n_jobs=-1`: memanfaatkan semua inti CPU untuk pelatihan secara paralel.

#### Kelebihan
- Akurat pada data dengan pola kompleks.
- Tahan terhadap overfitting karena menggunakan banyak pohon.

#### Kekurangan
- Pelatihan lebih lambat dibanding model linear.
- Kurang transparan (sulit diinterpretasi).


### Model 3 — **Gradient Boosting Regressor**

#### Cara Kerja  
Gradient Boosting adalah teknik boosting yang membangun model secara bertahap, di mana setiap pohon berikutnya berusaha memperbaiki kesalahan dari model sebelumnya. Model ini mengoptimalkan fungsi loss dengan cara iteratif untuk menghasilkan prediksi yang lebih akurat.

#### Parameter
Model digunakan dengan konfigurasi:
- `n_estimators=100`: jumlah total pohon.
- `learning_rate=0.1`: tingkat kontribusi masing-masing pohon terhadap model akhir.
- `max_depth=4`: kedalaman maksimum setiap pohon.
- `random_state=55`: untuk memastikan replikasi hasil.

#### Kelebihan
- Sering kali menghasilkan prediksi yang sangat akurat, terutama pada data tabular.
- Dapat menangkap interaksi kompleks antar fitur.

#### Kekurangan
- Lebih rentan terhadap overfitting jika tidak diatur dengan baik.
- Pelatihan lebih lambat karena model dibangun secara bertahap.


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

- Linear Regression, meskipun sederhana, menghasilkan kesalahan prediksi yang lebih tinggi dibanding model lain.
- Random Forest Regressor menghasilkan MAE paling rendah pada data pelatihan dan juga cukup kompetitif pada data pengujian.
- Gradient Boosting memiliki performa mendekati Random Forest, namun tidak sebaik Random Forest dalam konteks dataset ini.


### Analisis Hasil dengan Bisnis Understanding:
Dari hasil di atas, dapat disimpulkan bahwa:

**Problem Statement**
- Problem Statement 1: Bagaimana memprediksi biaya asuransi seseorang berdasarkan berbagai faktor? 
    - Model berhasil dibangun dan mampu memprediksi biaya asuransi dengan tingkat kesalahan rata-rata yang dapat diterima (sekitar $2.396).
- Problem Statement 2: Algoritma machine learning mana yang menghasilkan prediksi paling akurat? 
    - Random Forest Regressor terbukti menjadi model dengan performa terbaik berdasarkan evaluasi metrik MAE.

**Goals**
- Goal 1: Membangun model prediktif biaya asuransi berdasarkan data demografis dan gaya hidup. 
    - Model telah dibangun dan diuji, dengan hasil evaluasi yang mendukung penggunaannya dalam konteks prediktif.
- Goal 2: Membandingkan performa beberapa algoritma regresi dan memilih model terbaik. 
    - Tiga model diuji secara adil, dan Random Forest dipilih berdasarkan hasil evaluasi kuantitatif.

**Solution Statement**
- Baseline model Linear Regression digunakan sebagai pembanding awal. 
    - Memberikan gambaran dasar performa.
- Random Forest dan Gradient Boosting digunakan untuk meningkatkan performa. 
    - Kedua model meningkatkan akurasi, dan Random Forest secara konsisten menghasilkan MAE lebih rendah.
- Metrik MAE digunakan untuk mengevaluasi performa model secara objektif. 
    - Metrik ini telah menunjukkan keefektifan pendekatan yang diambil.

## Kesimpulan
Proyek ini berhasil membangun sistem prediksi biaya asuransi kesehatan menggunakan pendekatan machine learning. Tiga algoritma regresi diuji: Linear Regression, Random Forest Regressor, dan Gradient Boosting Regressor. Berdasarkan metrik MAE, Random Forest Regressor menunjukkan performa terbaik dengan kesalahan rata-rata yang paling kecil dan stabil pada data uji. Secara bisnis, model ini memiliki dampak penting:
- Efisiensi: Memungkinkan perusahaan asuransi untuk secara otomatis dan akurat memprediksi biaya klaim.
- Keadilan: Penetapan premi dapat dipersonalisasi secara adil berdasarkan profil risiko individu.
- Pengambilan Keputusan: Memberikan dasar yang lebih kuat untuk pengambilan keputusan dalam proses underwriting.

Dengan model ini, perusahaan asuransi dapat meningkatkan akurasi prediksi premi, mengurangi risiko kerugian, dan memberikan layanan yang lebih transparan dan kompetitif kepada pelanggan.
