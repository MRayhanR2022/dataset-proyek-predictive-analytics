# Laporan Proyek Machine Learning - Muhammad Rayhan Rasyad (MC006D5Y1862)

## Domain Proyek

Pertumbuhan konten digital dan kebutuhan personalisasi mendorong banyak platform digital untuk mengembangkan sistem rekomendasi, termasuk dalam industri hiburan seperti film. Sistem ini tidak hanya membantu pengguna menemukan konten yang relevan, tetapi juga meningkatkan engagement, retensi, dan konversi bisnis.

Dalam proyek ini, dibangun sistem rekomendasi film menggunakan dua pendekatan populer: Content-Based Filtering dan Collaborative Filtering berbasis Neural Network. Pendekatan ini dipilih karena keduanya saling melengkapi — content-based efektif pada cold start, sedangkan collaborative lebih akurat pada data besar.

Referensi:
[1] Ricci, F., Rokach, L., & Shapira, B. (2011). Introduction to Recommender Systems Handbook. Springer.



## Business Understanding
### Problem Statements
- Bagaimana sistem dapat merekomendasikan film berdasarkan preferensi pengguna sebelumnya?
- Bagaimana mengatasi tantangan cold start dan sparsity dalam sistem rekomendasi film?

### Goals
- Mengembangkan sistem rekomendasi film yang mampu memberikan top-N rekomendasi berdasarkan histori pengguna.
- Membandingkan efektivitas dua pendekatan: content-based filtering dan collaborative filtering berbasis neural network.

### Solution Approach
#### Solution 1: Content Based Filtering
- Menggunakan metadata (genre) dari film.
- Mengubah genre menjadi representasi numerik dengan TF-IDF.
- Mengukur kemiripan antarfilm dengan Cosine Similarity.

#### Solution 2: Collaborative Filtering (Neural Network)
- Membangun model berdasarkan pola interaksi user-item.
- Melakukan embedding pada user dan movie ID.
- Model dilatih menggunakan arsitektur neural network sederhana dengan fungsi aktivasi sigmoid.

## Data Understanding
Dataset yang digunakan dalam proyek ini diambil dari Kaggle: https://www.kaggle.com/datasets/parasharmanas/movie-recommendation-system?select=ratings.csv

Dataset yang digunakan dalam proyek ini merupakan kumpulan data interaksi pengguna terhadap film, yang mencakup informasi rating yang diberikan oleh pengguna terhadap berbagai judul film. Dataset terdiri dari dua file utama: `ratings.csv` dan `movies.csv`. Setlah digabungkan dan kolom `timestamp` di drop dapat dilihat bahwa dataset terdiri dari 25.000.095 baris data (sample) dan 5 kolom.

### Deskripsi Fitur:
- `movieId`: ID unik untuk setiap film (numeric)
- `title`: Judul film, termasuk tahun rilis (object)
- `genres`: Daftar genre film (object)
- `userId`: ID unik pengguna (numeric)
- `movieId`: ID film yang dirating (numeric)
- `rating`: Nilai rating dari pengguna (0.5 sampai 5.0)

### Kondisi data:
Untuk memahami kualitas dan karakteristik data, dilakukan pemeriksaan awal terhadap nilai kosong, data duplikat, dan outlier. Berikut adalah temuan dari eksplorasi tersebut:
- **Missing values:** Tidak ditemukan nilai kosong pada seluruh kolom. Semua entri dalam dataset terisi dengan lengkap, sehingga tidak diperlukan penanganan nilai hilang.
- **Duplikasi:** Tidak ditemukan data duplikat pada seluruh kolom. Semua entri dalam dataset terisi dengan benar, sehingga tidak diperlukan penanganan data duplikat.

### EDA yang dilakukan:
Exploratory Data Analysis (EDA) dilakukan untuk memahami lebih lanjut dataset yang digunakan, seperti tipe data setiap variabel, distribusi data, dan karakteristik masing-masing variabel. Berikut EDA yang dilakukan:

#### Analisis Univariat
Dilakukan pengecekan isi dari masing-masing fitur `userId`, `movieId`, `title`, `rating`, dan `genres`. Berdasarkan hasil analisis diketahui informasi berikut:
1. Terdapat 162.541 user unik dalam dataset
2. Terdapat 59.047 film unik dalam dataset
3. Terdapat 58.958 judul film unik dalam dataset
4. Rating memiliki rentang nilai mulai dari 0,5 sampai dengan 5,0 dengan interval 0,5
5. Terdapat 20 genre berbeda dalam dataset, yaitu: (no genres listed), Action, Adventure, Animation, Children, Comedy, Crime, Documentary, Drama, Fantasy, Film-Noir, Horror, IMAX, Musical, Mystery, Romance, Sci-Fi, Thriller, War, dan Western.



## Data Preparation
Tahapan ini bertujuan untuk mempersiapkan data agar siap digunakan oleh algoritma machine learning. Seluruh langkah dilakukan secara sistematis berdasarkan karakteristik data yang telah dianalisis sebelumnya.

Langkah-Langkah yang Dilakukan:
1. **Mensample dataset:** 
Jumlah data yang tersedia sangat besar, yakni 25 juta baris. Untuk menghemat waktu komputasi dan sumber daya selama proses eksplorasi dan pelatihan model, dilakukan pengambilan sampel sebanyak 500.000 baris secara acak menggunakan fungsi sample() dari pandas, dan disimpan dalam variabel df_small.

2. **Mengubah genre film menjadi representasi vektor TF-IDF:**
Genre film yang berupa string dikonversi menjadi representasi vektor numerik menggunakan teknik TF-IDF (`TfidfVectorizer`). Representasi ini digunakan dalam content-based filtering untuk menghitung kemiripan antar film.

3. **Menyalin data:**
Data yang sudah disampling disalin ke dalam variabel baru `cf_df` yang akan digunakan secara khusus untuk collaborative filtering.

4. **Mapping dan encoding:**
ID pengguna (`userId`) dan ID film (`movieId`) dikonversi ke indeks numerik berturut-turut (0 hingga N) menggunakan dictionary mapping. Hasil mapping ini ditambahkan sebagai kolom `user` dan `movie`.

5. **Membagi Data untuk Training dan Validasi:**
Data dipecah menjadi data pelatihan dan data validasi dengan rasio 80:20 menggunakan `train_test_split`. Rating juga dinormalisasi ke rentang 0–1 agar sesuai dengan output fungsi aktivasi sigmoid yang digunakan dalam model.

### Alasan Setiap Tahap:
- **Mensample dataset:** Untuk mengurangi beban komputasi dan mempercepat proses pemodelan.
- **Mengubah genre film menjadi representasi vektor TF-IDF:** Agar informasi genre dapat dihitung kemiripannya secara numerik dalam content-based filtering.
- **Menyalin data:** Untuk memisahkan proses antara content-based dan collaborative filtering agar tidak saling memengaruhi.
- **Mapping dan encoding:** Karena model neural network hanya menerima input numerik berupa indeks, bukan ID asli.
- **Membagi Data untuk Training dan Validasi:** Untuk mengukur performa model secara objektif di data yang tidak digunakan saat pelatihan.



## Model Development
### Model 1 — **Content Based Filtering**
#### Cara Kerja  
Model ini merekomendasikan film berdasarkan kesamaan konten, yaitu informasi genre dari film. Genre diubah menjadi representasi numerik menggunakan TF-IDF vectorizer, kemudian kemiripan antar film dihitung menggunakan cosine similarity. Pengguna dapat diberikan rekomendasi film serupa dengan film yang pernah disukai atau berdasarkan genre yang diinputkan.

#### Rekomendasi
Ketika pengguna memilih film "King Arthur: Legend of the Sword (2017)", maka hasilnya adalah:
Rekomendasi untuk: King Arthur: Legend of the Sword (2017) | Genre: Action|Adventure|Drama|Fantasy
1. Lord of the Rings: The Return of the King, The (2003) | Genre: Action|Adventure|Drama|Fantasy
2. D-War (Dragon Wars) (2007) | Genre: Action|Adventure|Drama|Fantasy
3. Beowulf & Grendel (2005) | Genre: Action|Adventure|Drama|Fantasy
4. Clash of the Titans (2010) | Genre: Action|Adventure|Drama|Fantasy
5. The Huntsman Winter's War (2016) | Genre: Action|Adventure|Drama|Fantasy

#### Kelebihan
- Tidak memerlukan data pengguna lain, cocok untuk cold-start pada pengguna baru.
- Interpretasi hasil lebih mudah karena didasarkan pada konten film yang jelas.

#### Kekurangan
- Hanya merekomendasikan film dengan genre yang mirip, tidak memperhitungkan selera kolektif.
- Tidak bisa menangani preferensi pengguna yang kompleks secara eksplisit.


### Model 2 — **Collaborative Filtering (Neural Network)**
#### Cara Kerja  
Model ini memanfaatkan user-item interaction matrix dan dilatih menggunakan neural network. ID pengguna dan film diubah menjadi indeks numerik dan dipetakan ke dalam embedding layer, kemudian hubungan laten antara pengguna dan film dipelajari. Output model berupa skor prediksi rating (skala 0–1, hasil dari sigmoid), yang digunakan untuk memilih film terbaik bagi masing-masing pengguna.

#### Rekomendasi
Ketika ingin mengetahui rekomendasi yang cocok untuk seseorang pengguna tertentu, misalnya pengguna dengan ID 126379, maka akan tampil hasil berikut:
Rekomendasi 10 film untuk user 126379:
- Shawshank Redemption, The (1994)
- Schindler's List (1993)
- Matrix, The (1999)
- Star Wars: Episode IV - A New Hope (1977)
- Sixth Sense, The (1999)
- Godfather, The (1972)
- Forrest Gump (1994)
- Back to the Future (1985)
- Goodfellas (1990)
- Interstellar (2014)

#### Kelebihan
- Mampu menangkap hubungan kompleks antar pengguna dan film secara otomatis.
- Memberikan rekomendasi personal tanpa melihat isi atau genre film secara eksplisit.

#### Kekurangan
- Tidak bisa memberikan rekomendasi untuk user atau item baru yang belum pernah muncul dalam data pelatihan (cold-start problem).
- Memerlukan data interaksi dalam jumlah besar dan waktu pelatihan yang relatif lama.



## Evaluation
Tahap evaluasi bertujuan untuk mengukur performa dari sistem rekomendasi yang dibangun. Karena sistem ini terdiri dari dua pendekatan yang berbeda, maka metrik evaluasi yang digunakan pun berbeda, disesuaikan dengan karakteristik masing-masing model.

### Model 1 — **Content Based Filtering**
#### Metrik Evaluasi
Untuk mengevaluasi model Content-Based Filtering, digunakan metrik `Precision at K`, yaitu persentase item yang relevan di antara K rekomendasi teratas yang diberikan kepada pengguna.

Formula:
$%
\text{Precision at K} = \frac{\text{Jumlah item relevan dalam top-K}}{K}
$$
#### Hasil
Dalam pengujian sederhana menggunakan input film "King Arthur: Legend of the Sword (2017)", sistem merekomendasikan 5 film dengan genre yang konsisten yaitu Action|Adventure|Drama|Fantasy. Karena genre yang dihasilkan seragam dan sesuai dengan input, dapat disimpulkan bahwa precision-nya cukup tinggi (mendekati 1).

### Model 2 — **Collaborative Filtering (Neural Network)**
#### Metrik Evaluasi
Model ini memprediksi rating yang diberikan pengguna terhadap film tertentu. Oleh karena itu, digunakan metrik `Root Mean Squared Error (RMSE)` untuk mengukur seberapa jauh prediksi model dibandingkan nilai aktualnya.

Formula: 
$$
\text{RMSE} = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 }
$$

### Hasil Evaluasi:
Model dilatih selama 3 epoch dengan hasil sebagai berikut:
| Epoch  | RMSE (Train) | RMSE (Validation) |
|--------|--------------|-------------------|
|   1    |    0.2673    |      0.2242       |
|   2    |    0.2342    |      0.2573       |
|   3    |    0.2552    |      0.2447       |

Dari hasil tersebut terlihat bahwa model mencapai performa terbaik pada epoch ke-1 (val_RMSE: 0.2242). Meskipun terdapat fluktuasi, nilai RMSE tetap berada di kisaran rendah, yang menunjukkan bahwa prediksi model cukup dekat dengan rating aktual pengguna.


### Analisis Hasil dengan Bisnis Understanding:
Dari hasil di atas, dapat disimpulkan bahwa:

**Problem Statement**
- Problem Statement 1: Bagaimana sistem dapat merekomendasikan film berdasarkan preferensi pengguna sebelumnya?
    - Telah terjawab dengan pendekatan collaborative filtering yang menggunakan histori interaksi pengguna dan content-based filtering yang memanfaatkan genre film yang disukai.

- Problem Statement 2: Bagaimana mengatasi kendala cold start dan sparsity dalam sistem rekomendasi?
    - Untuk cold-start pada pengguna baru, content-based filtering dapat digunakan karena tidak membutuhkan data interaksi sebelumnya.
    - Untuk pengguna lama, collaborative filtering memberikan hasil rekomendasi yang lebih personal berdasarkan pola global.

**Goals**
- Goal 1: Mengembangkan sistem rekomendasi film yang mampu memberikan top-N rekomendasi berdasarkan histori pengguna.
    - Sistem telah dibangun dengan dua pendekatan dan berhasil memberikan rekomendasi film relevan berdasarkan preferensi pengguna.

- Goal 2: Membandingkan efektivitas dua pendekatan: content-based filtering dan collaborative filtering berbasis neural network.
    - Evaluasi dilakukan dengan metrik yang sesuai (Precision dan RMSE), dan hasilnya digunakan untuk menilai kelebihan serta kekurangan masing-masing pendekatan.

**Solution Statement**
- Solution 1 (Content-Based Filtering) 

Efektif dalam menangani cold start untuk item karena hanya membutuhkan fitur konten film, namun kurang mampu melakukan personalisasi mendalam karena tidak mempertimbangkan pola interaksi pengguna lain.

- Solution 2 (Collaborative Filtering)

Mampu menangkap preferensi kolektif pengguna dan memberikan rekomendasi yang lebih personal, namun memerlukan banyak data historis dan belum ideal untuk pengguna baru tanpa histori (cold start user).

## Kesimpulan
- Content-Based Filtering sangat efektif untuk rekomendasi berbasis genre, namun terbatas untuk pengguna dengan preferensi eksplisit.
- Collaborative Filtering unggul dalam rekomendasi personal, namun memerlukan data interaksi yang cukup dan tidak cocok untuk pengguna atau item baru.
- Kombinasi keduanya (hybrid system) dapat menjadi solusi masa depan untuk mencapai personalisasi maksimal dan ketahanan terhadap cold start.
