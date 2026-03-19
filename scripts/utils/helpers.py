def get_bilateral_trials(trial_names):
    return [t for t in trial_names if "Left" not in t and "Right" not in t]


def apply_s08_scaling(df, subject_id):
    if subject_id == "S08":
        return df * 1_000_000
    return df