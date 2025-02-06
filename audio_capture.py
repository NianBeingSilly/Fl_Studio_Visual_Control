import numpy as np
import pyaudio

class AudioCapture:
    def __init__(self, rate=44100, chunk=1024):
        self.rate = rate  # Sampling rate (Hz)
        self.chunk = chunk  # Jumlah frame per buffer
        self.p = pyaudio.PyAudio()  # Inisialisasi PyAudio
        self.stream = None
        self.device_index = self.find_stereo_mix_device()  # Mencari perangkat Stereo Mix atau menggunakan perangkat default
        self.init_audio_stream()  # Menginisialisasi aliran audio

    def find_stereo_mix_device(self):
        """
        Mencari perangkat Stereo Mix atau menggunakan perangkat input default.
        """
        for i in range(self.p.get_device_count()):
            device_info = self.p.get_device_info_by_index(i)
            if "Stereo Mix" in device_info["name"]:  # Cek apakah perangkat Stereo Mix ada
                print(f"Stereo Mix ditemukan pada indeks perangkat {i}.")
                return i
        print("Stereo Mix tidak ditemukan. Menggunakan perangkat input default.")
        return self.p.get_default_input_device_info()["index"]

    def init_audio_stream(self):
        """
        Menginisialisasi aliran audio dengan perangkat yang dipilih.
        """
        try:
            self.stream = self.p.open(
                format=pyaudio.paInt16,  # Format audio 16-bit
                channels=1,  # 1 saluran (mono)
                rate=self.rate,  # Sampling rate
                input=True,  # Mode input (menerima audio)
                frames_per_buffer=self.chunk,  # Jumlah frame per buffer
                input_device_index=self.device_index  # Indeks perangkat input
            )
            print("Aliran audio berhasil dibuka.")
        except Exception as e:
            print(f"Gagal menginisialisasi aliran audio: {e}")
            self.stream = None

    def get_audio_data(self):
        """
        Menangkap data audio secara real-time dari desktop.
        
        Menggunakan FFT (Fast Fourier Transform) untuk mendapatkan spektrum frekuensi.
        """
        try:
            if self.stream is None:
                print("Aliran audio tidak tersedia.")
                return None

            # Membaca data audio dari aliran
            data = self.stream.read(self.chunk, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)  # Mengonversi byte menjadi integer 16-bit

            # Menghitung FFT untuk mendapatkan spektrum frekuensi
            fft_data = np.abs(np.fft.fft(audio_data))[:self.chunk // 2]  # Hanya mengambil setengah spektrum (frekuensi positif)
            return fft_data
        except Exception as e:
            print(f"Gagal menangkap audio: {e}")
            return None

    def close(self):
        """
        Menutup aliran audio.
        """
        if self.stream:
            self.stream.stop_stream()  # Menghentikan aliran
            self.stream.close()  # Menutup aliran
        self.p.terminate()  # Menghentikan PyAudio
