# Linear Regression from Scratch
### Latar Belakang
Linear Regression adalah salah satu teknik dasar dalam statistika dan pembelajaran mesin yang digunakan untuk mengukur dan memodelkan hubungan linier antara satu atau lebih variabel independen (input) dan variabel dependen (output). Ini adalah salah satu pendekatan paling sederhana untuk memahami dan memodelkan hubungan antara variabel-variabel tersebut.

Pada dasarnya, Linear Regression mencoba untuk menemukan garis lurus (dalam kasus regresi satu variabel) atau datar hyperplane (dalam kasus regresi lebih dari satu variabel) yang paling baik mewakili pola hubungan antara variabel independen dan dependen. Tujuan utamanya adalah untuk meminimalkan selisih antara nilai yang diprediksi oleh model dan nilai yang sebenarnya dari data pelatihan.

Berikut adalah beberapa konsep penting terkait Linear Regression:

1. Variabel Independen dan Dependen: 

Variabel independen (input) adalah variabel yang digunakan untuk memprediksi nilai variabel dependen (output). Misalnya, dalam prediksi harga rumah, variabel independen bisa berupa luas tanah, jumlah kamar, dll., sementara harga rumah menjadi variabel dependen.

2. Garis Regresi: 

Dalam regresi satu variabel, garis regresi adalah garis lurus yang mencoba menyesuaikan data sedemikian rupa sehingga selisih antara nilai yang diprediksi oleh garis dan nilai sebenarnya dari data di minimalkan. Dalam regresi lebih dari satu variabel, kita menggunakan hyperplane sebagai generalisasi dari garis.

3. Konsep Least Squares: 

Pendekatan yang umum digunakan dalam Linear Regression adalah Least Squares. Ini berarti kita mencari garis atau hyperplane yang menghasilkan jumlah kuadrat terkecil dari selisih antara nilai yang diprediksi dan nilai sebenarnya dari data pelatihan.

4. Koefisien Regresi: 

Dalam regresi linear, koefisien (slope) dari garis regresi menggambarkan perubahan rata-rata dalam variabel dependen untuk setiap perubahan satu unit dalam variabel independen. Koefisien intercept mengindikasikan nilai variabel dependen ketika variabel independen nol.

5. Evaluasi Model: 

Evaluasi model Linear Regression melibatkan analisis residu (selisih antara nilai sebenarnya dan nilai yang diprediksi). Metrik evaluasi yang umum digunakan termasuk Mean Squared Error (MSE), Root Mean Squared Error (RMSE), dan R-squared (koefisien determinasi).

Membangun Linear Regression dari awal melibatkan menghitung koefisien regresi dan intercept secara matematis dengan meminimalkan fungsi biaya (seperti MSE) menggunakan teknik optimasi seperti metode gradien turun. Namun, dalam implementasi nyata, library atau framework pembelajaran mesin seperti scikit-learn atau TensorFlow sering digunakan untuk menghemat waktu dan usaha.

Pemahaman tentang Linear Regression penting karena memberikan dasar yang kuat untuk memahami konsep-konsep yang lebih kompleks dalam pembelajaran mesin dan statistika.

### Kelebihan dan Kelemahan Model

#### Kelebihan Linear Regression:

1. Sederhana dan Interpretatif: 

Linear regression adalah model yang relatif sederhana dan mudah diinterpretasikan. Koefisien dalam model dapat memberikan wawasan tentang hubungan antara variabel input dan output.

2. Penggunaan Luas: 

Linear regression dapat digunakan dalam berbagai konteks, baik untuk analisis prediktif maupun pemahaman hubungan antar variabel.

3. Stabilitas: 

Linear regression cenderung stabil dan memiliki risiko overfitting yang lebih rendah dibandingkan dengan model yang lebih kompleks.

4. Deteksi Outlier: 

Outlier dapat terdeteksi dengan mudah dalam linear regression melalui analisis residual.

5. Penghitungan Efisien: 

Perhitungan koefisien dalam linear regression memiliki solusi matematis tertutup (closed-form solution), yang memungkinkan perhitungan yang relatif efisien.

#### Kelemahan Linear Regression:

1. Asumsi Linieritas: 

Linear regression hanya efektif jika hubungan antara variabel independen dan dependen bersifat linier. Jika hubungan bersifat non-linier, hasil prediksi dapat menjadi tidak akurat.

2. Sensitif terhadap Outlier: 

Meskipun dapat mendeteksi outlier, outlier yang signifikan dapat memiliki pengaruh besar pada hasil model linear regression.

3. Asumsi Kemandirian Variabel: 

Model ini mengasumsikan bahwa variabel input adalah mandiri satu sama lain. Jika ada korelasi atau interaksi antara variabel, model ini mungkin tidak akurat.

4. Heteroskedastisitas: 

Model ini mengasumsikan homoskedastisitas, yaitu variasi residual konstan. Jika variasi residual tidak konstan (heteroskedastisitas), hasil prediksi dan interval kepercayaan menjadi tidak akurat.

5. Overfitting Kecil: 

Linear regression cenderung memiliki kemampuan prediksi yang lebih rendah jika pola yang kompleks ada dalam data. Ini dapat menghasilkan hasil yang kurang akurat dalam situasi seperti itu.

6. Terbatas pada Masalah Regresi: 

Linear regression hanya cocok untuk masalah regresi, yaitu ketika kita mencoba memprediksi nilai numerik. Ini tidak cocok untuk masalah klasifikasi di mana output adalah kategori diskrit.

Secara keseluruhan, linear regression adalah model yang kuat dan sederhana, tetapi keefektifannya tergantung pada sejauh mana asumsi-asumsi yang mendasarinya terpenuhi dalam data yang digunakan. Jika asumsi-asumsi ini tidak terpenuhi, model yang lebih kompleks atau teknik lain mungkin lebih cocok.
