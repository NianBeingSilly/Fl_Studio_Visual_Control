import cv2
import mediapipe as mp
import math

class HandTracker:
    def __init__(self):
        # Inisialisasi MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        print("MediaPipe Hands berhasil diimpor!")  # Pesan debugging

        try:
            # Inisialisasi objek Hands dengan maksimal 2 tangan yang terdeteksi
            self.hands = self.mp_hands.Hands(max_num_hands=2)
            print(f"Atribut 'hands' berhasil diinisialisasi! Tipe: {type(self.hands)}")  # Pesan debugging
        except Exception as e:
            # Jika terjadi error, tampilkan pesan dan hentikan program
            print(f"Error saat inisialisasi 'hands': {e}")
            raise

        # Inisialisasi utilitas menggambar landmark
        self.mp_draw = mp.solutions.drawing_utils

        # Parameter untuk normalisasi dan smoothing
        self.max_distance = 500  # Jarak maksimum untuk normalisasi
        self.smoothed_speed = 127  # Nilai awal untuk smoothing eksponensial
        self.alpha = 0.2  # Faktor smoothing (0 < alpha <= 1)

    def calculate_distance(self, x1, y1, x2, y2):
        # Menghitung jarak Euclidean antara dua titik (x1, y1) dan (x2, y2)
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def track_hands(self, frame):
        # Konversi frame dari BGR ke RGB (MediaPipe memerlukan format RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Pastikan atribut 'hands' sudah terinisialisasi
        if not hasattr(self, 'hands'):
            print("Error: Atribut 'hands' tidak terinisialisasi!")
            return [], 127  # Kembalikan data kosong dan nilai speed default

        # Proses frame untuk mendeteksi tangan
        results = self.hands.process(rgb_frame)
        hands_data = []  # List untuk menyimpan data tangan yang terdeteksi

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Ambil landmark untuk ujung ibu jari, ujung telunjuk, dan pergelangan tangan
                thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]

                # Dapatkan dimensi frame
                h, w, _ = frame.shape

                # Konversi koordinat landmark ke piksel
                thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
                index_x, index_y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)

                # Hitung jarak antara ujung ibu jari dan ujung telunjuk
                distance_thumb_index = self.calculate_distance(thumb_x, thumb_y, index_x, index_y)

                # Normalisasi jarak ke rentang 0-127
                normalized_distance = min(int((distance_thumb_index / self.max_distance) * 127 * 4), 127)

                # Simpan data tangan ke dalam list
                hands_data.append({
                    "thumb": (thumb_x, thumb_y),  # Koordinat ujung ibu jari
                    "index": (index_x, index_y),  # Koordinat ujung telunjuk
                    "wrist": (wrist_x, wrist_y),  # Koordinat pergelangan tangan
                    "distance": normalized_distance  # Jarak normalisasi
                })

                # Gambar landmark dan koneksi tangan pada frame
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            # Hitung nilai speed jika dua tangan terdeteksi
            speed_value = self.smoothed_speed  # Nilai default jika hanya satu tangan
            if len(results.multi_hand_landmarks) == 2:
                # Ambil landmark ujung ibu jari dari kedua tangan
                hand1_thumb = results.multi_hand_landmarks[0].landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                hand2_thumb = results.multi_hand_landmarks[1].landmark[self.mp_hands.HandLandmark.THUMB_TIP]

                # Konversi koordinat landmark ke piksel
                hand1_thumb_x, hand1_thumb_y = int(hand1_thumb.x * w), int(hand1_thumb.y * h)
                hand2_thumb_x, hand2_thumb_y = int(hand2_thumb.x * w), int(hand2_thumb.y * h)

                # Hitung jarak antara kedua ujung ibu jari
                distance_between_thumbs = self.calculate_distance(hand1_thumb_x, hand1_thumb_y, hand2_thumb_x, hand2_thumb_y)

                # Debugging: Tampilkan jarak antara kedua ibu jari
                print(f"Jarak Antara Ibu Jari: {distance_between_thumbs}")

                # Hitung nilai speed berdasarkan jarak antara ibu jari
                max_speed_distance = 500  # Jarak maksimum untuk normalisasi speed
                normalized_speed = int((1 - distance_between_thumbs / max_speed_distance) * 60)  # Faktor pengali 60
                speed_value = max(0, min(127, normalized_speed))  # Batasi nilai speed antara 0 dan 127

                # Smoothing eksponensial untuk menghaluskan perubahan speed
                self.smoothed_speed = int(self.alpha * speed_value + (1 - self.alpha) * self.smoothed_speed)
                speed_value = self.smoothed_speed

                # Debugging: Tampilkan nilai speed
                print(f"Nilai Speed: {speed_value}")

            return hands_data, speed_value  # Kembalikan data tangan dan nilai speed

        return [], 127  # Kembalikan data kosong dan nilai speed default jika tidak ada tangan terdeteksi