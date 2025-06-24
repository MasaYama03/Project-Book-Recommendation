# Laporan Project Machine Learning - Masahiro Gerarudo Yamazaki

## Project Overview

Di era digital masa kini, banyak sektor bisnis memanfaatkan machine learning untuk meningkatkan efisiensi dan produktivitas mereka. Salah satu penerapan paling sederhana namun efektif adalah sistem rekomendasi, yang tidak hanya mendorong penjualan lewat produk yang relevan, tetapi juga membantu menarik lebih banyak pelanggan. Teknologi ini juga sangat cocok diterapkan pada toko buku maupun perpustakaan, guna memberikan rekomendasi buku yang lebih personal dan relevan bagi pengguna.

Selain manfaat komersial, sistem rekomendasi buku juga berpotensi besar dalam meningkatkan literasi masyarakat Indonesia, yang masih menjadi tantangan serius. Menurut data dari hasil PISA 2022 yang dirilis akhir 2023, peringkat literasi membaca Indonesia naik sekitar 5–6 posisi dibanding tahun 2018, meski skor rata-ratanya belum setara negara OECD [[1](https://www.polibatam.ac.id/en/indonesias-ranking-in-pisa-2022-has-increased-by-5-to-6-positions-compared-to-2018/?utm_source=chatgpt.com)]. Kenaikan ini menjadi indikasi bahwa inovasi seperti sistem rekomendasi dapat menjadi salah satu solusi strategis untuk mendukung budaya membaca yang lebih luas.

Dengan latar belakang tersebut, saya mengangkat topik **Pembuatan Sistem Rekomendasi Buku dengan menggunakan Metode Collaborative Filtering**, sebagai kontribusi dalam membantu pengguna menemukan buku yang cocok sekaligus mendorong peningkatan literasi secara lebih luas.

## Business Understanding

### Problem Statements
Berdasarkan latar belakang yang telah dijelaskan, berikut merupakan rumusan masalah yang ingin diselesaikan dalam proyek ini:
- Bagaimana cara memanfaatkan data yang tersedia untuk menghasilkan rekomendasi buku yang sesuai dengan preferensi pengguna, baik untuk dibaca maupun dibeli, dengan menggunakan pendekatan Collaborative Filtering?

### Goals
Proyek ini bertujuan untuk:
- Memberikan rekomendasi buku kepada pengguna dengan memanfaatkan metode Collaborative Filtering, berdasarkan rating yang sebelumnya telah diberikan oleh pengguna terhadap buku-buku tertentu.

### Solution Statement
Solusi yang diusulkan dalam proyek ini adalah penerapan metode [Collaborative Filtering](https://developers.google.com/machine-learning/recommendation/collaborative/basics).

Metode ini akan membantu merekomendasikan buku kepada pengguna berdasarkan rating yang sudah mereka berikan sebelumnya. Dari data rating tersebut, sistem bisa menemukan buku lain yang mirip dan cocok, lalu merekomendasikannya ke pengguna — terutama buku yang belum pernah mereka beri rating sebelumnya.

**Kelebihan** :  
- Tidak membutuhkan pengetahuan khusus tentang domain tertentu, karena proses embedding mampu mempelajari hubungan antar data secara otomatis.
- Membantu pengguna dalam menemukan ketertarikan atau preferensi yang baru
- Model hanya memerlukan data berupa feedback pengguna (seperti rating) dalam bentuk matriks untuk proses pelatihan, tanpa memerlukan informasi tambahan seperti fitur kontekstual.

**Kekurangan** : 
- Sistem ini tidak bisa memberikan rekomendasi kepada pengguna baru karena bergantung pada riwayat aktivitas pengguna sebelumnya, seperti rating atau ulasan yang pernah diberikan (cold-start problem).
- kemudian sulit untuk mengikut sertakan fitur lain (side-features) pada kueri atau item.


## Data Understanding

Informasi Dataset:

Jenis | Keterangan
--- | ---
Title | Book Recommendation Dataset
Source | [Kaggle](https://www.kaggle.com/arashnic/book-recommendation-dataset)
Maintainer | [Möbius](https://www.kaggle.com/arashnic)
License | [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)
Usability | 10.0

Dataset yang digunakan terdiri dari tiga file CSV, yaitu `Books.csv` , `Ratings.csv` , dan `Users.csv`

Pada file `Books.csv` menyimpan informasi terkait buku sebanyak 271.360 baris dan memiliki 8 kolom, diantaranya adalah :  

- `ISBN` : berisi kode ISBN dari buku  
- `Book-Title` : berisi judul buku
- `Book-Author` : berisi penulis buku
- `Year-Of-Publication` : tahun terbit buku  
- `Publisher` : penerbit buku  
- `Image-URL-S` : URL menuju gambar buku berukuran kecil
- `Image-URL-M` : URL menuju gambar buku berukuran sedang
- `Image-URL-L` : URL menuju gambar buku berukuran besar

Sementara itu, file Ratings.csv berisi data penilaian buku oleh pengguna, dengan jumlah 1.149.780 baris dan memiliki 3 kolom, yaitu :  

 - `User-ID` : berisi ID unik pengguna
 - `ISBN` : berisi kode ISBN buku yang diberi rating oleh pengguna
 - `Book-Rating` : berisi nilai rating yang diberikan oleh pengguna berkisar antara 0-10

![Cuplikan Data Rating](https://github.com/user-attachments/assets/1d9dc151-22e6-4d7d-af36-f9e6e6d828a4)

Dalam data `Rating`, ditemukan bahwa `User-ID` berupa angka dengan nilai yang cukup besar, sedangkan `ISBN` adalah string unik yang terdiri dari kombinasi huruf dan angka sebagai identitas buku. Kedua data ini nantinya perlu di-encoding agar bisa diproses dalam pembuatan sistem rekomendasi. Data rating ini berperan sebagai sumber utama dalam membangun sistem rekomendasi menggunakan metode Collaborative Filtering pada proyek ini.
  
Sementara itu, file `Users.csv` menyimpan informasi mengenai pengguna. Dataset ini berisikan 278.858 baris dan memiliki 3 kolom, yaitu : 

- `User-ID` : berisi ID unik pengguna
- `Location` : berisi data lokasi pengguna
- `Age` : berisi data usia pengguna

Berikut ini adalah hasil dari visualiasi jumlah rating buku yang diberikan oleh user.

![Jumlah Rating Buku yang Diberikan Pengguna](https://github.com/user-attachments/assets/48b3d751-62a5-4dbe-8305-f7c90dbf95b0)

Berdasarkan visualisasi diagram di atas, terlihat bahwa sebagian besar pengguna lebih dari 700 ribu memberikan rating 0 terhadap buku. Kondisi ini menunjukkan bahwa data bersifat tidak seimbang *(imbalanced)*. Oleh karena itu, perlu dilakukan proses penanganan khusus pada data tersebut agar distribusinya menjadi lebih seimbang.

## Data Preparation
Berikut merupakan tahapan yang dilakukan dalam proses penyiapan data *(Data Preparation)* pada proyek ini:

- **Menangani Ketidakseimbangan Data (Handling Imbalanced Data)**  
  Seperti yang telah dijelaskan sebelumnya, sebagian besar pengguna memberikan rating 0 terhadap buku, sehingga distribusi data menjadi tidak seimbang. Ketidakseimbangan ini berpotensi menurunkan performa model. Oleh karena itu, pada tahap ini seluruh data dengan rating 0 akan dihapus. Meskipun jumlah data menjadi jauh lebih sedikit, namun distribusinya menjadi lebih merata dan diharapkan dapat meningkatkan kualitas prediksi model.

- **Encoding**  
  Proses ini dilakukan untuk mengubah nilai `User-ID` dan `ISBN` ke dalam bentuk indeks bilangan bulat (integer). Hal ini diperlukan karena `User-ID` merupakan angka acak berukuran besar, dan `ISBN` adalah kombinasi huruf dan angka. Dengan mengubahnya ke dalam bentuk indeks, data menjadi lebih mudah diproses oleh model.

- **Pengacakan Data (Randomize Dataset)**  
  Dataset akan diacak secara acak agar distribusinya tidak berpola. Tujuan dari pengacakan ini adalah untuk mengurangi varians, mencegah model menjadi *overfit*, serta memastikan bahwa data validasi mewakili keseluruhan distribusi data secara adil.

- **Standarisasi Data (Data Standardization)**  
  Nilai rating awal berada dalam rentang 0 hingga 10. Untuk memudahkan proses pelatihan model, rating ini akan dinormalisasi ke dalam rentang 0 hingga 1. Standarisasi ini penting agar semua variabel berada pada skala yang sama dan tidak memberikan bobot berlebih terhadap model, yang bisa menyebabkan bias.

- **Pembagian Dataset (Data Splitting)**  
  Dataset akan dibagi menjadi dua bagian utama: 80% untuk proses pelatihan (*training*) dan 20% untuk validasi (*validation*). Pembagian ini bertujuan untuk melatih model sekaligus mengevaluasi performanya terhadap data yang tidak dilihat selama pelatihan.

## Modeling
Pada tahap ini, model akan menghitung tingkat kecocokan antara pengguna dan buku menggunakan pendekatan *embedding*.

Beberapa properti penting yang digunakan dalam kelas `RecommenderNet` sebagai parameter dalam layer embedding antara lain:

- `num_users` : total jumlah pengguna dalam data
- `num_isbn` : total jumlah buku berdasarkan kode ISBN
- `embedding_size` : dimensi vektor embedding yang digunakan untuk pengguna dan buku

Langkah pertama adalah melakukan proses *embedding* terhadap data pengguna dan buku. Nilai `num_users` dan `num_isbn` digunakan sebagai input untuk membentuk vektor embedding masing-masing entitas. Sementara itu, `embedding_size` menentukan seberapa besar dimensi vektor yang dihasilkan. Ukuran embedding yang lebih besar cenderung meningkatkan akurasi model, namun jika terlalu besar justru bisa menyebabkan model mengalami *overfitting*. 

Oleh karena itu, dalam proyek ini digunakan bantuan *Optuna* untuk menemukan nilai `embedding_size` yang paling optimal.

Setelah proses *embedding* selesai, dilakukan operasi *dot product* antara embedding pengguna dan buku untuk menghasilkan skor kecocokan. Selain itu, model juga memungkinkan penambahan nilai *bias* pada masing-masing pengguna dan buku. Nilai kecocokan ini kemudian disesuaikan dalam skala [0, 1] menggunakan fungsi aktivasi sigmoid.

Model ini dikompilasi menggunakan fungsi *loss* `BinaryCrossentropy` dan mengadopsi algoritma `Adam` sebagai optimizer, dengan nilai *learning rate* sebesar 0.001.

Setelah proses pelatihan selesai, model mampu menghasilkan rekomendasi 10 buku teratas (*top-10*) untuk setiap pengguna, seperti yang ditampilkan pada hasil berikut.

![Top-10 Book Recommendation](https://github.com/user-attachments/assets/7204a714-4fab-46ad-86f7-59ed539af36a)

## Evaluation
Pada proyek ini, **Root Mean Square Error (RMSE)** digunakan sebagai metrik utama untuk mengevaluasi performa model. RMSE merupakan metrik standar dalam pengukuran kesalahan prediksi pada nilai kuantitatif. RMSE dihitung sebagai **akar kuadrat dari rata-rata kuadrat selisih antara nilai prediksi dan nilai aktual** [[2](https://medium.com/%40wl8380/demystifying-regression-evaluation-metrics-mse-rmse-and-mae-75c32121c6a3)].

Perhitungan RMSE ditunjukkan pada rumus berikut ini:

![RMSE](https://github.com/user-attachments/assets/1953a376-14e0-4b6a-90b0-25dacc9e5a50)

`RMSE` = nilai root mean square error

`y`  = nilai hasil observasi

`ŷ`  = nilai hasil prediksi

`i`  = urutan data

`n`  = jumlah data

Nilai RMSE yang rendah mengindikasikan bahwa hasil prediksi model memiliki tingkat kesalahan yang kecil dan mendekati nilai sebenarnya dari data observasi. RMSE digunakan untuk mengukur sejauh mana perbedaan antara nilai prediksi dan nilai aktual. Semakin kecil nilai RMSE, maka semakin akurat dan sesuai prediksi model terhadap data yang diamati.

Berikut adalah visualisasi plot metrik RMSE setelah model selesai dilatih.

![Model Metrics]((https://github.com/user-attachments/assets/1953a376-14e0-4b6a-90b0-25dacc9e5a50)

Berdasarkan plot di atas, terlihat bahwa model menghasilkan skor RMSE sebesar 0.184, yang menunjukkan performa prediksi yang sudah cukup baik. Meskipun demikian, nilai error tersebut masih bisa ditekan lebih rendah lagi melalui pengembangan dan penyempurnaan model lebih lanjut.

## Referensi

[[1](https://www.polibatam.ac.id/en/indonesias-ranking-in-pisa-2022-has-increased-by-5-to-6-positions-compared-to-2018/?utm_source=chatgpt.com)] Politeknik Negeri Batam. (2023). Indonesia’s ranking in PISA 2022 has increased by 5 to 6 positions compared to 2018. Polibatam News. https://www.polibatam.ac.id/en/indonesias-ranking-in-pisa-2022-has-increased-by-5-to-6-positions-compared-to-2018/?utm_source=chatgpt.com

[[2](https://medium.com/%40wl8380/demystifying-regression-evaluation-metrics-mse-rmse-and-mae-75c32121c6a3)] Liu, W. (2021). Demystifying Regression Evaluation Metrics: MSE, RMSE, and MAE. Medium. https://medium.com/@wl8380/demystifying-regression-evaluation-metrics-mse-rmse-and-mae-75c32121c6a3

