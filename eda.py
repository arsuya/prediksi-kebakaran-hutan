# Import library
import streamlit as st
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import glob

def run():
    st.title('Exploratory Data Analysis')

    st.markdown('EDA akan difokuskan untuk menganalisis karakteristik dari gambar-gambar kebakaran hutan dan tidak, sehingga dapat memberikan pemahaman awal mengenai pola visual yang membedakan keduanya')
    # tranval_fire = glob.glob("Training and Validation/fire/*")
    # tranval_nofire = glob.glob("Training and Validation/nofire/*")
    tranval_fire = sorted(glob.glob("Training and Validation/fire/*"))
    tranval_nofire = sorted(glob.glob("Training and Validation/nofire/*"))

    fields = ['Bagaimana distribusi data training dan validation ?', 'Bagaimana distribusi ukuran gambar ?',
              'Bagaimana contoh gambar kelas fire dan no fire ?', 'Bagaimana distribusi warna kelas fire dan no fire ?',
              'Bagaimana membedakan ciri-ciri hutan kebakaran dan hutan saat senja ?']
    
    pilihan = st.selectbox('Pilih Bagian EDA', fields)

    if pilihan == fields[0]:
        st.subheader('- Bagaimana distribusi data training dan validation ?')
        num_fire_traval = len(tranval_fire)
        num_nofire_traval = len(tranval_nofire)
        num_traval = [num_fire_traval, num_nofire_traval]

        label = ['Fire', 'No Fire']

        fig, ax = plt.subplots(figsize=(6, 5))
        bars = ax.bar(label, num_traval, color=['red', 'green'])
        ax.set_title('Jumlah data target untuk training dan validation')
        ax.set_ylabel('Jumlah data')

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 5, str(height), ha='center', va='bottom')

        st.pyplot(fig)
        st.markdown('Jumlah data pada training and validation adalah 928 gambar untuk kebakaran hutan dan 904 gambar untuk tidak kebakaran hutan. Dengan perbedaan jumlah yang sangat kecil (1.3%), dataset ini dapat dikatakan seimbang')
    
    elif pilihan == fields[1]:
        st.subheader('- Bagaimana distribusi ukuran gambar ?')
        st.markdown('Analisis distribusi gambar perlu dianalisis karena model CNN memerlukan input gambar dengan ukuran yang seragam agar dapat diproses secara efisien selama training. Selain itu, ukuran gambar juga berdampak langsung pada kompleksitas komputasi dan kecepatan pelatihan model. Hasil dari analisis ini akan menjadi dasar pertimbangan dalam menentukan apakah gambar perlu di resize, serta ukuran berapa yang paling optimal untuk digunakan dalam arsitektur CNN')

        all_images = tranval_fire + tranval_nofire
        widths, heights = [], []

        for img_path in all_images:
            with Image.open(img_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(widths, heights, color='blue')
        ax.set_title('Distribusi Ukuran Gambar (Width × Height)')
        ax.set_xlabel('Width (pixel)')
        ax.set_ylabel('Height (pixel)')
        st.pyplot(fig)
        st.markdown('Hasil analisis distribusi ukuran gambar menunjukkan bahwa seluruh gambar pada dataset memiliki ukuran yang konsisten, yakni 250 pixel x 250 pixel')

    elif pilihan == fields[2]:
        st.subheader('- Bagaimana contoh gambar kelas fire dan no fire ?')
        st.markdown('Analisis contoh visualisasi gambar dari masing-masing kelas perlu dianalisis untuk memverifikasi kualitas visual data, sekaligus memperoleh gambaran awal mengenai karakteristik visual yang membedakan kedua kelas')

        img_fire = mpimg.imread(tranval_fire[27])
        img_nofire = mpimg.imread(tranval_nofire[27])

        fig = plt.figure(figsize=(10, 5.5))
        plt.subplot(1, 2, 1)
        plt.title('Gambar kelas fire')
        plt.imshow(img_fire)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title('Gambar kelas no fire')
        plt.imshow(img_nofire)
        plt.axis('off')

        st.pyplot(fig)

        st.markdown('Gambar yang ditampilkan di atas merupakan contoh representatif dari masing-masing kelas dalam dataset, yaitu kelas fire dan no fire')
        st.markdown('Terlihat pada kelas fire, adanya nyala api berwarna oranye kemerahan yang mendominasi latar depan, terlihat juga asap dan semprotan air yang untuk pemadaman api')
        st.markdown('Sementara itu pada kelas no fire memperlihatkan kondisi hutan yang didominasi oleh warna hijau dari pepohonan, serta tidak menunjukkan adanya unsur api dan asap')
        st.markdown('Perbedaan warna dominan jika dilihat melalui pengamatan mata langsung antar kedua kelas cukup jelas, sehingga dataset ini memiliki kejelasan visual yang baik dan secara pengamatan mata langsung sudah mengandung fitur pembeda antar kelas yang signifikan')

    elif pilihan == fields[3]:
        st.subheader('- Bagaimana distribusi warna kelas fire dan no fire ?')
        st.markdown('Analisis berikutnya mengidentifikasi apakah terdapat pola warna yang khas yang dapat membedakan antar kelas. Analisis ini bertujuan untuk mengidentifikasi apakah terdapat pola warna yang khas yang dapat membedakan antara kelas fire dan no fire')
        st.markdown('Visualisasi dilakukan dengan membuat boxplot atau histogram RGB untuk masing-masing kelas, guna mengetahui sebaran intensitas pixel pada warna red, green, dan blue, serta mendeteksi adanya dominasi warna tertentu di masing-masing kategori. Gambar yang digunakan dalam analisis ini adalah gambar yang sama seperti pada langkah EDA sebelumnya')

        def get_rgb_flattened(img_path):
            img_bgr = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            return img_rgb[:, :, 0].flatten(), img_rgb[:, :, 1].flatten(), img_rgb[:, :, 2].flatten()

        R_fire, G_fire, B_fire = get_rgb_flattened(tranval_fire[27])
        R_nofire, G_nofire, B_nofire = get_rgb_flattened(tranval_nofire[27])

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        ax[0].boxplot([R_fire, G_fire, B_fire], labels=['Red', 'Green', 'Blue'],
                      patch_artist=True, boxprops=dict(facecolor='white'),
                      medianprops=dict(color='black'))
        ax[0].set_title('Boxplot Warna - Kelas Fire')
        ax[0].set_ylabel('Nilai Intensitas Pixel (0–255)')
        ax[0].grid(axis='y', linestyle='--', alpha=0.6)

        ax[1].boxplot([R_nofire, G_nofire, B_nofire], labels=['Red', 'Green', 'Blue'],
                      patch_artist=True, boxprops=dict(facecolor='white'),
                      medianprops=dict(color='black'))
        ax[1].set_title('Boxplot Warna - Kelas No Fire')
        ax[1].grid(axis='y', linestyle='--', alpha=0.6)

        st.pyplot(fig)

        st.markdown('Hasilnya,  Pada gambar kelas fire, distribusi warna merah cenderung memiliki median dan rentang IQR yang lebih tinggi dibandingkan warna lainnya. Hal ini mengindikasikan bahwa warna merah mendominasi pada gambar kebakaran')
        st.markdown('Sebaliknya, pada gambar kelas no fire, warna hijau menunjukkan distribusi yang lebih tinggi dan luas, dengan median yang juga lebih besar dibanding channel lainnya. Ini mencerminkan keberadaan pepohonan yang cenderung berwarna hijau')

    if pilihan == fields[4]:
        st.subheader('- Bagaimana membedakan ciri-ciri hutan kebakaran dan hutan saat senja ?')
        st.markdown('Saat  melakukan analisis gambar, ditemukan gambar-gambar pohon saat senja hari. Dimana pada waktu senja, langit dan lingkungan sekitar cenderung didominasi oleh warna jingga atau kemerahan. Hal ini dapat menyebabkan gambar pepohonan pada waktu senja memiliki warna yang menyerupai kebakaran, seperti yang terlihat pada contoh gambar dibawah. Jika dilihat secara mata langsung, warna jingga yang dominan pada gambar hutan saat senja dapat menyerupai warna nyala api, sehingga berpotensi membingungkan model dalam proses klasifikasi')

        def get_rgb_flattened(img_path):
            img_bgr = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            return img_rgb[:, :, 0].flatten(), img_rgb[:, :, 1].flatten(), img_rgb[:, :, 2].flatten()

        R_fire, G_fire, B_fire = get_rgb_flattened(tranval_fire[30])
        R_nofire, G_nofire, B_nofire = get_rgb_flattened(tranval_nofire[30])

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        ax[0].boxplot([R_fire, G_fire, B_fire], labels=['Red', 'Green', 'Blue'],
                      patch_artist=True, boxprops=dict(facecolor='white'),
                      medianprops=dict(color='black'))
        ax[0].set_title('Boxplot Warna - Kelas Fire')
        ax[0].set_ylabel('Nilai Intensitas Pixel (0–255)')
        ax[0].grid(axis='y', linestyle='--', alpha=0.6)

        ax[1].boxplot([R_nofire, G_nofire, B_nofire], labels=['Red', 'Green', 'Blue'],
                      patch_artist=True, boxprops=dict(facecolor='white'),
                      medianprops=dict(color='black'))
        ax[1].set_title('Boxplot Warna - Kelas No Fire')
        ax[1].grid(axis='y', linestyle='--', alpha=0.6)

        st.pyplot(fig)

        st.markdown('Jika dilihat dari hasil visualisasi boxplot, kelas no fire tampak memiliki pola distribusi warna yang mirip dengan kelas fire, terutama pada warna merah yang memiliki nilai median lebih tinggi dibandingkan warna lainnya. Pola ini menyerupai kecenderungan warna pada gambar kebakaran hutan, yang umumnya memang didominasi oleh warna merah yang tinggi. Oleh karena itu, boxplot pada gambar kelas no fire kali ini tampak menyerupai rata-rata pola boxplot pada kelas fire')
        st.markdown('Sehingga diperlukan visualisasi lain yang dapat membedakan karakteristik dari masing masing kelas, seperti tekstur dan pola tepi objek')
        st.markdown('Maka dilakukan uji coba dengan menerapkan metode deteksi tepi menggunakan filter Sobel dan Canny. Pendekatan ini bertujuan supaya kita dapat melihat struktur dalam gambar yang bisa menjadi pembeda antara pola nyala api dan pencahayaan matahari senja')

        def detect_edges(image_path):
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
            sobel_combined = cv2.magnitude(sobelx, sobely)
            canny = cv2.Canny(img, 100, 200)
            return sobel_combined, canny

        fire_path = tranval_fire[30]
        nofire_path = tranval_nofire[30]
        sobel_fire, canny_fire = detect_edges(fire_path)
        sobel_nofire, canny_nofire = detect_edges(nofire_path)

        fig, ax = plt.subplots(2, 3, figsize=(15, 8))

        ax[0, 0].imshow(cv2.imread(fire_path)[..., ::-1])
        ax[0, 0].set_title('Original Fire')
        ax[1, 0].imshow(cv2.imread(nofire_path)[..., ::-1])
        ax[1, 0].set_title('Original No Fire')

        ax[0, 1].imshow(sobel_fire, cmap='gray')
        ax[0, 1].set_title('Sobel - Fire')
        ax[1, 1].imshow(sobel_nofire, cmap='gray')
        ax[1, 1].set_title('Sobel - No Fire')

        ax[0, 2].imshow(canny_fire, cmap='gray')
        ax[0, 2].set_title('Canny - Fire')
        ax[1, 2].imshow(canny_nofire, cmap='gray')
        ax[1, 2].set_title('Canny - No Fire')

        for a in ax.flat:
            a.axis('off')

        st.pyplot(fig)
        st.markdown('Berdasarkan hasil deteksi tepi menggunakan metode sobel dan canny, dapat disimpulkan bahwa terdapat perbedaan karakteristik visual antara gambar pada kelas fire dan no fire. Pada gambar kelas fire, pola tepi yang dihasilkan oleh filter sobel tampak lebih tegas dan terarah, terutama pada bagian atas api dan objek yang tersinari cahaya api. Hal ini juga diperkuat oleh hasil canny yang menunjukkan batas-batas objek yang lebih jelas dan terstruktur. Karakteristik ini menunjukkan bahwa nyala api memiliki kontras yang tinggi dan membentuk pola arah yang spesifik')
        st.markdown('Sedangkan, pada gambar kelas no fire, hasil deteksi tepi menunjukkan banyak pola acak dan tidak terarah. Sobel menghasilkan kontur yang tersebar di antara dedaunan, sementara canny menunjukkan pola tepi yang lebih padat namun tidak membentuk struktur menyala')

if __name__ == '__main__':
    run()
