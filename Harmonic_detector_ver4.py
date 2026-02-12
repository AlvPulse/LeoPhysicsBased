import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from matplotlib.animation import FuncAnimation
import os

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
FILENAME = 'single_mic3.wav'
WINDOW_DURATION = 2.0  
STEP_SIZE = 0.4
REFRESH_INTERVAL = 100
MIN_FREQ_FILTER = 150.0  
MAX_FREQ_FILTER = 2000.0
NUM_PEAKS_TO_CHECK = 7  
TOLERANCE = 0.2          
# ==========================================

def run_analysis():
    if not os.path.exists(FILENAME):
        print(f"❌ Error: '{FILENAME}' not found.")
        return

    # Load Audio
    fs, audio = wavfile.read(FILENAME)
    if len(audio.shape) > 1: audio = audio[:, 0]
    duration = len(audio) / fs
    
    # Pre-calc Spectrogram
    f_spec, t_spec, Sxx = signal.spectrogram(audio, fs, nperseg=1024)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)

    # --- Setup Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.canvas.manager.set_window_title('Harmonic Monitor')
    plt.subplots_adjust(hspace=0.3, wspace=0.15)

    # 1. Probability History Plot
    ax_prob = axes[0, 0]
    line_prob, = ax_prob.plot([], [], color='green', lw=2)
    ax_prob.set_title("1. Probability", fontweight='bold')
    ax_prob.set_ylim(0, 105)
    ax_prob.set_xlim(0, duration)
    ax_prob.set_ylabel("Probability (%)")
    ax_prob.set_xlabel("Time (s)")
    ax_prob.grid(True, alpha=0.3)
    
    # Fill area under curve
    fill_prob = ax_prob.fill_between([], [], color='green', alpha=0.1)

    # 2. Spectrogram
    ax_spec = axes[0, 1]
    ax_spec.pcolormesh(t_spec, f_spec, Sxx_db, shading='gouraud', cmap='inferno')
    cursor_line = ax_spec.axvline(x=0, color='cyan', linestyle='--')
    ax_spec.set_title("2. Spectrogram", fontweight='bold')
    ax_spec.set_ylabel("Freq (Hz)")

    # 3. PSD Curve
    ax_psd = axes[1, 0]
    line_psd, = ax_psd.plot([], [], color='navy', lw=1.5)
    scatter_peaks = ax_psd.scatter([], [], color='red', s=50, zorder=5)
    ax_psd.axvline(x=MIN_FREQ_FILTER, color='orange', linestyle=':', linewidth=2, label='Filter Cutoff')
    ax_psd.set_title("3. Contours", fontweight='bold')
    ax_psd.set_xlim(0, 1500)
    ax_psd.set_ylim(-100, 0)
    ax_psd.grid(alpha=0.3)
    ax_psd.legend(loc='upper right')

    # 4. LIVE TABLE
    ax_table = axes[1, 1]
    ax_table.axis('off')
    ax_table.set_title(f"4. Harmonic Table", fontweight='bold')
    
    col_labels = ["Frequency", "Power", "Ratio", "Status"]
    table_data = [["-", "-", "-", "-"] for _ in range(NUM_PEAKS_TO_CHECK)]
    the_table = ax_table.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(11)
    the_table.scale(1, 2.0)

    # Store history for probability plot
    prob_history_t = []
    prob_history_y = []

    def init():
        line_prob.set_data([], [])
        line_psd.set_data([], [])
        scatter_peaks.set_offsets(np.empty((0, 2)))
        cursor_line.set_xdata([0])
        return line_prob, line_psd, scatter_peaks, cursor_line

    def update(frame):
        start_t = frame * STEP_SIZE
        end_t = start_t + WINDOW_DURATION
        if end_t > duration: return init()

        # Slice
        s_idx, e_idx = int(start_t*fs), int(end_t*fs)
        slice_audio = audio[s_idx:e_idx]
        if len(slice_audio) < 2048: return init()

        # Update Spectrogram Cursor
        cursor_line.set_xdata([start_t + WINDOW_DURATION/2])

        # PSD
        f_welch, Pxx = signal.welch(slice_audio, fs, nperseg=2048, noverlap=0.75*2048)
        Pxx_db = 10 * np.log10(Pxx + 1e-10)
        line_psd.set_data(f_welch, Pxx_db)

        # Peak Detection
        max_h = np.max(Pxx_db)
        peaks, _ = signal.find_peaks(Pxx_db, height=max_h-100, prominence=4, distance=40)
        pf = f_welch[peaks]
        pa = Pxx_db[peaks]

        # FILTER > 150Hz
        mask = (pf > MIN_FREQ_FILTER) & (pf < MAX_FREQ_FILTER)
        pf = pf[mask]
        pa = pa[mask]

        # SELECTION (Top 5 Loudest)
        loudness_sort = np.argsort(pa)[::-1]
        top_indices = loudness_sort[:NUM_PEAKS_TO_CHECK]
        pf_top = pf[top_indices]
        pa_top = pa[top_indices]

        # RE-SORT by Frequency
        freq_sort = np.argsort(pf_top)
        pf_final = pf_top[freq_sort]
        pa_final = pa_top[freq_sort]
        
        scatter_peaks.set_offsets(np.c_[pf_final, pa_final])

        # --- PROBABILITY CALCULATION ---
        current_score = 0
        matches = 0
        
        # Table Updates
        for r in range(NUM_PEAKS_TO_CHECK):
            for c in range(4):
                the_table[r+1, c].get_text().set_text("-")
                the_table[r+1, c].set_facecolor('white')

        if len(pf_final) > 0:
            # We have at least a Base
            matches = 0
            base_freq = pf_final[0]

            for i in range(len(pf_final)):
                f = pf_final[i]
                a = pa_final[i]
                
                ratio = f / base_freq
                coeff = round(ratio)
                drift = abs(ratio - coeff)
                
                # Table Logic
                if i == 0:
                    status = "BASE"
                    color = "#e6f2ff"
                    ratio_str = "1.00"
                else:
                    ratio_str = f"{ratio:.2f}"
                    # Check Harmonic
                    if drift < TOLERANCE and coeff > 1:
                        status = f"Harmonic #{coeff}"
                        color = "#e6ffe6"
                        matches += 1 # Found a harmonic!
                    else:
                        status = "Non-Harmonic"
                        color = "#ffe6e6"

                the_table[i+1, 0].get_text().set_text(f"{f:.1f}")
                the_table[i+1, 1].get_text().set_text(f"{a:.1f}")
                the_table[i+1, 2].get_text().set_text(ratio_str)
                the_table[i+1, 3].get_text().set_text(status)
                for c in range(4): the_table[i+1, c].set_facecolor(color)

            # Calculate Final Score
            current_score = (matches / 4) * 100
        else:
            current_score = 0

        # Update Probability Plot
        prob_history_t.append(start_t)
        prob_history_y.append(current_score)
        line_prob.set_data(prob_history_t, prob_history_y)
        
        # Redraw fill (a bit expensive, so simple trick: remove old and add new)
        # For simplicity in animation, we just stick to the line or update polygon
        # (Updating Polygon efficiently in Matplotlib animation is complex, keeping line for speed)
        print(f"probability = {current_score}%")
        return line_prob, line_psd, scatter_peaks, cursor_line, the_table

    print(f"Starting Analysis... Max Score possible: 100% (if {NUM_PEAKS_TO_CHECK} harmonics found)")
    anim = FuncAnimation(fig, update, frames=range(int((duration-WINDOW_DURATION)/STEP_SIZE)), 
                         init_func=init, blit=False, interval=REFRESH_INTERVAL)
    plt.show()

if __name__ == "__main__":
    run_analysis()