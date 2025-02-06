import mido
from mido import Message

print("File midi_control.py berhasil diimpor!")

class MidiController:
    def __init__(self):
        # Menampilkan daftar port MIDI output yang tersedia
        print("Available MIDI Output Ports:", mido.get_output_names())
        
        try:
            # Membuka port MIDI dengan nama 'visualDj 1'
            self.outport = mido.open_output('visualDj 1')  # Ganti dengan nama port Anda
            print(f"Port MIDI '{self.outport.name}' berhasil dibuka!")
        except Exception as e:
            # Menangani error jika gagal membuka port MIDI
            print(f"Error membuka port MIDI: {e}")
            raise

    def send_midi_signals(self, hands_data, speed):
        # Keluar dari fungsi jika tidak ada data tangan
        if not hands_data:
            return

        # Kontrol Volume menggunakan tangan kanan
        right_hand = hands_data[0]
        volume = right_hand["distance"]  # Nilai sudah dinormalisasi ke 0-127
        volume_message = Message('control_change', control=7, value=int(volume))  # CC 7 untuk volume
        self.outport.send(volume_message)

        # Kontrol EQ menggunakan tangan kiri
        eq_level = 0
        if len(hands_data) > 1:
            left_hand = hands_data[1]
            eq_level = left_hand["distance"]  # Nilai sudah dinormalisasi ke 0-127
            eq_message = Message('control_change', control=10, value=int(eq_level))  # CC 10 untuk EQ
            self.outport.send(eq_message)

        # Kontrol Speed
        speed_value = int(speed)  # Pastikan speed_value adalah integer
        speed_message = Message('control_change', control=22, value=speed_value)  # CC 22 untuk speed
        self.outport.send(speed_message)

        # Menampilkan nilai volume, EQ, dan speed untuk debugging
        print(f"Volume: {volume}, EQ Level: {eq_level}, Speed: {speed_value}")