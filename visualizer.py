import os
import numpy as np
from pyaudio import PyAudio, paInt16
import cv2
from PIL import Image, ImageDraw, ImageFont

# Menentukan backend OpenGL untuk visualisasi
os.environ['VISPY_GL_BACKEND'] = 'PyQt5'

class Visualizer:
    def __init__(self):
        # Konfigurasi audio
        self.CHUNK = 1024  # Ukuran buffer, semakin besar semakin banyak data yang diproses
        self.FORMAT = paInt16  # Format data audio
        self.CHANNELS = 1  # Audio mono
        self.RATE = 44100  # Frekuensi sampel audio (Hz)

        # Inisialisasi PyAudio untuk menangani stream audio
        self.p = PyAudio()
        self.stream = None
        self.init_audio_stream()

        # Konfigurasi visualisasi
        self.NUM_BARS = 10  # Jumlah bar dalam visualisasi spektrum
        self.BAR_WIDTH = 2  # Lebar setiap bar
        self.BAR_SPACING = 2  # Jarak antar bar
        self.SPECTRUM_HEIGHT = 40  # Tinggi maksimum spektrum
        self.BG_COLOR = (30, 30, 30)  # Warna latar belakang
        self.LINE_COLOR = (255, 255, 255)  # Warna garis spektrum
        self.TEXT_COLOR = (255, 255, 255)  # Warna teks
        self.FONT_PATH = "Poppins-Regular.ttf"  # Path ke file font
        self.FONT_SIZE = 18  # Ukuran font untuk teks

    def init_audio_stream(self):
        """
        Inisialisasi stream audio dengan memilih perangkat input (Stereo Mix atau perangkat default).
        """
        try:
            # Coba cari perangkat Stereo Mix
            device_index = None
            for i in range(self.p.get_device_count()):
                device_info = self.p.get_device_info_by_index(i)
                if "Stereo Mix" in device_info["name"]:
                    print(f"Stereo Mix ditemukan pada device index {i}.")
                    device_index = i
                    break

            # Jika Stereo Mix tidak ditemukan, gunakan perangkat input default
            if device_index is None:
                print("Stereo Mix tidak ditemukan. Menggunakan perangkat input default.")
                device_index = self.p.get_default_input_device_info()["index"]

            # Buka stream audio dengan perangkat yang ditemukan
            self.stream = self.p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK,
                input_device_index=device_index
            )
            print("Stream audio berhasil dibuka.")

        except Exception as e:
            print(f"Error saat membuka stream audio: {e}")
            self.stream = None

    def update_spectrum(self):
        """
        Menghitung FFT dari data audio untuk mendapatkan spektrum frekuensi.
        """
        try:
            if self.stream is None:
                print("Stream audio tidak tersedia.")
                return np.zeros(self.NUM_BARS)

            # Membaca data audio dari stream
            data = self.stream.read(self.CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)

            # Menghitung FFT (Fast Fourier Transform) untuk mendapatkan spektrum frekuensi
            fft_data = np.abs(np.fft.fft(audio_data))[:self.CHUNK // 2]
            normalized_fft = fft_data / np.max(fft_data) if np.max(fft_data) > 0 else np.zeros_like(fft_data)

            # Menyesuaikan ukuran spektrum dengan jumlah bar
            x_original = np.linspace(0, 1, len(normalized_fft))
            x_new = np.linspace(0, 1, self.NUM_BARS)
            resized_fft = np.interp(x_new, x_original, normalized_fft)

            return resized_fft

        except Exception as e:
            print(f"Error saat memperbarui spektrum: {e}")
            return np.zeros(self.NUM_BARS)

    def draw_visuals(self, frame, hands_data, volume, filter_level, speed, audio_data=None):
        """
        Menggambar elemen-elemen visual pada frame dari kamera.
        """
        height, width = frame.shape[:2]

        # Menggambar elemen visual untuk setiap tangan
        lines = []  # Menyimpan titik-titik tengah garis antara ibu jari dan telunjuk
        for i, hand in enumerate(hands_data):
            thumb = hand["thumb"]
            index = hand["index"]

            # Gambar lingkaran pada ibu jari dan telunjuk
            self.draw_stroke_circle(frame, thumb, 10, (0, 0, 255), thickness=1)  # Merah pada ibu jari
            self.draw_stroke_circle(frame, index, 10, (0, 255, 255), thickness=1)  # Kuning pada telunjuk

            # Gambar garis antara ibu jari dan telunjuk
            self.draw_smooth_line(frame, thumb, index, (255, 255, 255), thickness=3)

            # Menyimpan titik tengah garis antara ibu jari dan telunjuk
            mid_x = (thumb[0] + index[0]) // 2
            mid_y = (thumb[1] + index[1]) // 2
            lines.append((mid_x, mid_y))

            # Posisi teks di atas telunjuk
            text_offset = 35  # Jarak teks dari telunjuk
            position = (index[0], index[1] - text_offset)
            if i == 0:  # Tangan kanan untuk volume
                text = f"Vol: {int(volume)}"
            elif i == 1:  # Tangan kiri untuk filter
                text = f"Filter: {int(filter_level)}"

            # Gambar teks dengan font Poppins
            self.draw_text_with_poppins(frame, text, position, font_size=self.FONT_SIZE, color=self.TEXT_COLOR)

        # Menggambar spektrum jika ada dua tangan
        if len(lines) == 2 and audio_data is not None:
            # Titik tengah garis antara ibu jari dan telunjuk di kedua tangan
            line1_mid = lines[0]
            line2_mid = lines[1]

            # Menghitung jarak antara kedua garis untuk menentukan lebar spektrum
            distance = self.calculate_distance(line1_mid, line2_mid)

            # Menentukan posisi spektrum berdasarkan jarak tangan
            x_start = min(line1_mid[0], line2_mid[0])
            x_end = max(line1_mid[0], line2_mid[0])
            spectrum_width = x_end - x_start

            # Menentukan titik tengah antara kedua garis
            mid_x = (line1_mid[0] + line2_mid[0]) // 2
            mid_y = (line1_mid[1] + line2_mid[1]) // 2

            # Gambar spektrum responsif yang bergerak mengikuti tangan
            self.draw_responsive_spectrum(frame, mid_x, mid_y, spectrum_width, audio_data, volume)

        return frame

    def calculate_distance(self, point1, point2):
        """
        Menghitung jarak Euclidean antara dua titik.
        """
        return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5

    def draw_stroke_circle(self, frame, center, radius, color, thickness=2):
        """
        Menggambar lingkaran dengan garis tepi tanpa isi.
        """
        cv2.circle(frame, center, radius, color, thickness=thickness, lineType=cv2.LINE_AA)

    def draw_smooth_line(self, frame, start, end, color, thickness=2):
        """
        Menggambar garis dengan anti-aliasing agar lebih halus.
        """
        cv2.line(frame, start, end, color, thickness=thickness, lineType=cv2.LINE_AA)

    def draw_text_with_poppins(self, frame, text, position, font_size=24, color=(255, 255, 255)):
        """
        Menggambar teks dengan font Poppins dan memberikan efek stroke hitam tipis di sekelilingnya.
        """
        x, y = position
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        try:
            font = ImageFont.truetype(self.FONT_PATH, font_size)
        except IOError:
            print(f"Font '{self.FONT_PATH}' tidak ditemukan. Menggunakan font default.")
            font = ImageFont.load_default()

        draw.text((x, y), text, font=font, fill=color)

        frame[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def draw_responsive_spectrum(self, frame, x_center, y_center, spectrum_width, audio_data, volume):
        """
        Menggambar spektrum audio yang menyusut atau melebar sesuai dengan jarak antara kedua tangan.
        """
        max_height = int(self.SPECTRUM_HEIGHT * (volume / 100))  # Tinggi spektrum berdasarkan volume
        total_width = spectrum_width
        x_start = x_center - total_width // 2

        if audio_data is None or len(audio_data) == 0:
            audio_data = np.zeros(self.NUM_BARS)  # Nilai default jika audio_data kosong

        # Normalisasi data audio
        normalized_audio = audio_data / np.max(audio_data) if np.max(audio_data) > 0 else np.zeros_like(audio_data)

        # Gambar spektrum
        for i in range(self.NUM_BARS):
            x_bar = x_start + i * (total_width // self.NUM_BARS)
            bar_height = int(normalized_audio[i] * max_height)
            y_top_up = y_center - bar_height  # Posisi atas
            y_bottom_down = y_center + bar_height  # Posisi bawah

            # Pastikan koordinat tidak negatif atau melebihi batas gambar
            y_top_up = max(0, y_top_up)
            y_bottom_down = min(frame.shape[0], y_bottom_down)

            # Gambar garis spektrum
            cv2.line(frame, (int(x_bar), int(y_center)), (int(x_bar), int(y_top_up)), self.LINE_COLOR, thickness=self.BAR_WIDTH, lineType=cv2.LINE_AA)
            cv2.line(frame, (int(x_bar), int(y_center)), (int(x_bar), int(y_bottom_down)), self.LINE_COLOR, thickness=self.BAR_WIDTH, lineType=cv2.LINE_AA)

    def close(self):
        """
        Menutup stream audio dan mengosongkan resource.
        """
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()