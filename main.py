import cv2
from hand_tracking import HandTracker
from midi_control import MidiController
from visualizer import Visualizer
from audio_capture import AudioCapture

def main():
    # Buka kamera default (indeks 0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera tidak dapat dibuka!")
        return

    # Inisialisasi komponen
    hand_tracker = HandTracker()  # Pelacak tangan
    midi_controller = MidiController()  # Kontroler MIDI
    visualizer = Visualizer()  # Visualisasi
    audio_capture = AudioCapture()  # Penangkap audio

    while True:
        # Baca frame dari kamera
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            print("Gagal membaca frame dari kamera!")
            break

        # Deteksi tangan dan dapatkan data (posisi tangan dan kecepatan)
        hands_data, speed = hand_tracker.track_hands(frame)

        # Kirim sinyal MIDI berdasarkan data tangan
        midi_controller.send_midi_signals(hands_data, speed)

        # Ambil nilai volume dan filter dari data tangan
        volume = hands_data[0]["distance"] if hands_data else 0  # Volume dari tangan kanan
        filter_level = hands_data[1]["distance"] if len(hands_data) > 1 else 0  # Filter dari tangan kiri

        # Ambil data audio dari desktop
        audio_data = audio_capture.get_audio_data()

        # Gambar visualisasi pada frame
        frame = visualizer.draw_visuals(frame, hands_data, volume, filter_level, speed, audio_data)

        # Tampilkan frame yang telah diproses
        cv2.imshow("FL Studio Controller", frame)

        # Keluar dari loop jika tombol 'q' ditekan
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Bersihkan sumber daya
    cap.release()  # Tutup kamera
    audio_capture.close()  # Tutup penangkap audio
    visualizer.close()  # Tutup visualizer
    cv2.destroyAllWindows()  # Tutup semua jendela OpenCV

if __name__ == "__main__":
    main()