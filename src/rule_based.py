
def evaluate_expert_rules_v1(features,rules):
        """
        Stage 1: Apply physics-based rules.
        Returns: (Decision_String, Confidence, Triggered_Rule_Name)
                 or None if no rule matches.
        """
        r=rules
        # Rule 3: Clear Tonal/Harmonic Signal (Low flatness, clear dominant frequency)
        #if features['spectral_flatness'] < r['FLATNESS_MAX_TONAL'] and features['harmonic_score']> r['HARMONIC_SCORE_THRESHOLD']:
        if features['harmonic_score']> r['HARMONIC_SCORE_THRESHOLD'] and features['spectral_flatness'] < r['FLATNESS_MAX_TONAL']:
            # Could be siren, bird, harmonic call
            return "Harmonic/Tonal", features['spectral_flatness'], "Highly Tonal"
        
        # Rule 1: Wind Noise (Very high low-frequency energy, high flatness/noise)
        # if features['low_band_ratio'] > r['WIND_LOW_BAND_RATIO_MIN'] and features['spectral_flatness'] > r['FLATNESS_MAX_TONAL'] and features['high_band_ratio'] < r['AIRPLANE_HIGH_BAND_RATIO_MIN']:
        #     return "Wind", features['low_band_ratio'], "High Low-Band Energy + High Flatness"

        # # Rule 2: Airplane Noise (Significant high-frequency energy, broadband)
        # if features['high_band_ratio'] > r['AIRPLANE_HIGH_BAND_RATIO_MIN'] and features['spectral_flatness'] > r['FLATNESS_MAX_TONAL'] and features['low_band_ratio'] < r['WIND_LOW_BAND_RATIO_MIN']:
        #     return "Airplane", features['high_band_ratio'], "High High-Band Energy + High Flatness"


        # Rule 4: Generic Broadband Noise
        if features['spectral_flatness'] > r['FLATNESS_MIN_NOISE'] and features['zcr_mean'] > r['ZCR_MIN_NOISE']:
            return "Broadband Noise", features['zcr_mean'], "High Flatness + High ZCR"

        return None # Falls through to ML Stage


def evaluate_expert_rules_v2(features, rules):
    """
    Classify sound using EXACT thresholds from your decision tree.
    Features required: 
        high_band_ratio, dominant_freq, low_band_ratio, 
        zcr_mean, spectral_kurtosis
    """
    # 1. Root: high_band_ratio <= 0.00
    if features["high_band_ratio"] <= rules["HIGH_BAND_RATIO_ROOT"]:
        if features["dominant_freq"] <= rules["DOMINANT_FREQ_WIND"]:
            return "wind"
        else:
            return "Airplane"
    
    # 2. high_band_ratio > 0.00
    else:
        # 2.1 high_band_ratio <= 0.47
        if features["high_band_ratio"] <= rules["HIGH_BAND_RATIO_MID"]:
            if features["low_band_ratio"] <= rules["LOW_BAND_RATIO_KITCHEN"]:
                # kitchen (removed) → map to noise
                return rules["KITCHEN_MAPPED_TO"]
            else:
                if features["zcr_mean"] <= rules["ZCR_MEAN_VEHICLE"]:
                    if features["zcr_mean"] <= rules["ZCR_MEAN_VEHICLE_SUB"]:
                        return "vehicle"
                    else:
                        return "Train"
                else:
                    if features["dominant_freq"] <= rules["DOMINANT_FREQ_NOISE"]:
                        return "noise"
                    else:
                        return "speech"
        
        # 2.2 high_band_ratio > 0.47
        else:
            # 2.2.1 high_band_ratio <= 1.60
            if features["high_band_ratio"] <= rules["HIGH_BAND_RATIO_MOTOR"]:
                if features["zcr_mean"] <= rules["ZCR_MEAN_MOTOR"]:
                    if features["spectral_kurtosis"] <= rules["SPECTRAL_KURTOSIS_MOTOR"]:
                        return rules["KITCHEN_MAPPED_TO"]  # kitchen → noise
                    else:
                        return "Motor"
                else:
                    return rules["BIKE_MAPPED_TO"]  # bike → vehicle
            
            # 2.2.2 high_band_ratio > 1.60
            else:
                if features["low_band_ratio"] <= rules["LOW_BAND_RATIO_RAIN"]:
                    return "rain"
                else:
                    return rules["SIREN_MAPPED_TO"]  # siren → noise