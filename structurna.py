import sys
import os
import pandas as pd
import re
import plotly.express as px
import random
import traceback
import time
from seqfold import fold

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
run_df_path = os.path.join(BASE_DIR, "data", "runs.csv")

# Ensure the root directory is in the path
sys.path.append(os.path.join(os.getcwd(), 'RiboNN'))

# Now the library will correctly see 'src'
from RiboNN.src.predict import predict_using_nested_cross_validation_models
import streamlit as st

CALIBRATION_DEFAULTS = {
    'mfe_offset': 25.0,
    'cap_mult': 2.0,
    'kozak_mult': 3.0,
    'cds_mult': 0.75,
    'full_results': None
}

for key, default_value in CALIBRATION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

if 'reset_counter' not in st.session_state:
    st.session_state.reset_counter = 0

def handle_calibration_reset():
    st.session_state.reset_counter += 1
    st.session_state['full_results'] = None

# 1. Cache heavy execution by isolating a single individual fold row index safely
@st.cache_data
def run_single_fold_cached(utr5_seq, cds_seq, utr3_seq, species, run_df_path, fold_idx, target_columns=None):
    # Read the run configuration matrix
    run_df_object = pd.read_csv(run_df_path)
    max_available_folds = len(run_df_object)
    
    # Strictly bind the fold index to what actually exists in the file to prevent out-of-bounds crashes
    safe_idx = max(1, min(fold_idx, max_available_folds))
    single_fold_row = run_df_object.iloc[[safe_idx - 1]].reset_index(drop=True)
        
    df = pd.DataFrame({
        'tx_id': ['query_seq'],
        'tx_sequence': [utr5_seq + cds_seq + utr3_seq],
        'utr5_sequence': [utr5_seq],
        'cds_sequence': [cds_seq],
        'utr3_sequence': [utr3_seq],
        'utr5_size': [len(utr5_seq)],
        'cds_size': [len(cds_seq)],
        'utr3_size': [len(utr3_seq)],
        'tx_size': [len(utr5_seq) + len(cds_seq) + len(utr3_seq)]
    })
    
    temp_path = os.path.abspath(f"temp_ribonn_input_fold_{fold_idx}_{random.randint(0,100000)}.csv")
    df.to_csv(temp_path, sep="\t", index=False)
    
    start_dir = os.getcwd()
    lib_dir = os.path.join(start_dir, "RiboNN")
    
    try:
        os.chdir(lib_dir)
        return predict_using_nested_cross_validation_models(temp_path, species, single_fold_row)
    except Exception as e:
        # Gracefully log any underlying framework errors without crashing the Streamlit interface
        st.sidebar.error(f"Error executing fold {fold_idx}: {str(e)}")
        return pd.DataFrame()
    finally:
        os.chdir(start_dir)
        if os.path.exists(temp_path):
            os.remove(temp_path)

# 2. Loop coordinator for tracking progress across folds with shape validation
def run_ribonn_prediction_cached(utr5_seq, cds_seq, utr3_seq, species, run_df_path, target_columns=None, ensemble_depth=None, progress_callback=None):
    raw_results = []
    
    # Dynamically clamp total requested folds to the physical row limit of the configuration matrix
    try:
        run_df_object = pd.read_csv(run_df_path)
        actual_limit = len(run_df_object)
    except:
        actual_limit = 90
        
    requested_folds = ensemble_depth if ensemble_depth is not None else 90
    total_folds = min(requested_folds, actual_limit)
    
    for fold_idx in range(1, total_folds + 1):
        fold_data = run_single_fold_cached(
            utr5_seq, cds_seq, utr3_seq, species, run_df_path, fold_idx, target_columns
        )
        if not fold_data.empty:
            raw_results.append(fold_data)
        
        if progress_callback is not None:
            progress_callback(fold_idx, total_folds)
            
    return raw_results

def clean_and_validate_sequences(utr5, cds, utr3):
    """
    Cleans whitespaces and verifies biology constraints.
    Returns (is_valid, data_or_message)
    """
    clean_utr5 = re.sub(r'\s+', '', utr5).upper()
    clean_cds = re.sub(r'\s+', '', cds).upper()
    clean_utr3 = re.sub(r'\s+', '', utr3).upper()
    
    if not clean_utr5 or not clean_cds or not clean_utr3:
        return False, "Empty Fields Detected: Please ensure all three sequence fields (5' UTR, CDS, 3' UTR) are filled out."
        
    clean_utr5 = clean_utr5.replace('T', 'U')
    clean_cds = clean_cds.replace('T', 'U')
    clean_utr3 = clean_utr3.replace('T', 'U')
    
    allowed = set("ACGU")
    if not set(clean_utr5).issubset(allowed) or not set(clean_cds).issubset(allowed) or not set(clean_utr3).issubset(allowed):
        return False, "Invalid Characters: Non-nucleotide bases found. Please check your sequences for non-alphabetic characters or degenerates (like 'N')."

    cds_len = len(clean_cds)
    if cds_len % 3 != 0:
        remainder = cds_len % 3
        return False, f"CDS Reading Frame Alignment Error: RiboNN requires the Coding Sequence (CDS) to be divisible by 3 (triplet codons). Your current CDS length is {cds_len} (off by {remainder} bases)."

    return True, (clean_utr5, clean_cds, clean_utr3)

if 'tag_index' not in st.session_state:
    st.session_state.tag_index = 0
    
if 'tag_len' not in st.session_state:
    st.session_state.tag_len = 12
    
ORGAN_MAP = {
    "Kidney": [
        "predicted_TE_Kidney_normal_tissue", 
        "predicted_TE_ccRCC"
    ],
    "Lung": [
        "predicted_TE_A549", "predicted_TE_Calu.3", "predicted_TE_PC9", 
        "predicted_TE_Primary_human_bronchial_epithelial_cells"
    ],
    "Brain/Nervous System": [
        "predicted_TE_early_neurons", "predicted_TE_neuronal_precursor_cells", 
        "predicted_TE_neurons", "predicted_TE_normal_brain_tissue", 
        "predicted_TE_human_brain_tumor", "predicted_TE_U.251", 
        "predicted_TE_U.343", "predicted_TE_SH.SY5Y"
    ],
    "Blood/Immune": [
        "predicted_TE_K562", "predicted_TE_THP.1", "predicted_TE_Primary_CD4._T.cells", 
        "predicted_TE_LCL", "predicted_TE_MOLM.13", "predicted_TE_Molt.3", 
        "predicted_TE_MM1.S", "predicted_TE_primary_macrophages"
    ],
    "Liver": [
        "predicted_TE_HepG2", "predicted_TE_Huh.7.5", "predicted_TE_Huh7"
    ],
    "Breast": [
        "predicted_TE_MCF7", "predicted_TE_MCF10A", "predicted_TE_MDA.MB.231", 
        "predicted_TE_T47D", "predicted_TE_ZR75.1", "predicted_TE_SUM159PT"
    ],
    "Prostate": [
        "predicted_TE_PC3", "predicted_TE_normal_prostate"
    ],
    "Muscle/Connective": [
        "predicted_TE_muscle_tissue", "predicted_TE_skeletal_muscle", 
        "predicted_TE_cardiac_fibroblasts", "predicted_TE_fibroblast"
    ],
    "General Cancer/Cell Lines (Other)": [
        "predicted_TE_HeLa", "predicted_TE_HeLa_S3", "predicted_TE_HCT116", 
        "predicted_TE_HEK293", "predicted_TE_HEK293T", "predicted_TE_PANC1", 
        "predicted_TE_U2OS", "predicted_TE_Vero_6"
    ]
}

def validate_seq(seq):
    clean = re.sub(r'[^ACGT]', '', seq.upper())
    return clean

def get_mfe(seq):
    dna_seq = seq.replace('U', 'T')
    structures = fold(dna_seq)
    return round(abs(structures[0].e), 2) if structures else 0.0

def calculate_kozak_penalty(tag_seq):
    clean_tag = validate_seq(tag_seq) 
    consensus = "GCCGCCACCAUGG"
    seq_window = clean_tag[:13].upper()
    return sum(1 for a, b in zip(seq_window, consensus) if a != b)

def calculate_penalty(utr, cds, use_full, mfe_off, cap_w, kozak_w, cds_w, start_excluded):
    utr = validate_seq(utr)
    cds = validate_seq(cds)
    tag = cds if start_excluded else cds[3:]
    raw_mfe = get_mfe(utr)
    mfe = raw_mfe + mfe_off 
    cap_window = (utr + 'NNNNNNNNNN')[:10]
    cap = round((cap_window.count('G') + cap_window.count('C')) / 10, 2)
    kozak = calculate_kozak_penalty(tag)
    tag_gc = (tag[:13].count('G') + tag[:13].count('C')) / 13 if len(tag) >= 13 else 0.5
    gc_cds = (cds.count('G') + cds.count('C')) / len(cds) if len(cds) > 0 else 0
    cds_p = round(abs(tag_gc - gc_cds), 2)
    total = abs(mfe) + (cap_w * cap) + (kozak_w * kozak) + (cds_w * cds_p)
    return round(total, 2), mfe, cap, kozak, cds_p

def generate_optimal_tags(cds_seq, utr_seq, mfe_off, cap_w, kozak_w, cds_w, start_excluded, tag_len):
    base_mfe = get_mfe(utr_seq) + mfe_off
    target_gc = (cds_seq.upper().count('G') + cds_seq.upper().count('C')) / len(cds_seq) if len(cds_seq) > 0 else 0
    
    def get_detailed_score(tag):
        kozak = calculate_kozak_penalty(tag)
        tag_gc = (tag.count('G') + tag.count('C')) / tag_len
        cds_p = round(abs(tag_gc - target_gc), 2)
        cap_window = (utr_seq + 'NNNNNNNNNN')[:10]
        cap = round((cap_window.count('G') + cap_window.count('C')) / 10, 2)
        total = abs(base_mfe) + (float(cap_w) * cap) + (float(kozak_w) * kozak) + (float(cds_w) * cds_p)
        return round(total, 2), base_mfe, cap, kozak, cds_p

    def hamming_distance(s1, s2):
        return sum(1 for a, b in zip(s1, s2) if a != b)

    random.seed(hash(cds_seq + utr_seq))
    pool = set()
    for _ in range(1000):
        curr = list("".join(random.choices("ACGT", k=tag_len)))
        curr_score, _, _, _, _ = get_detailed_score("".join(curr))
        improved = True
        while improved:
            improved = False
            for i in range(tag_len):
                orig = curr[i]
                for base in "ACGT":
                    if base == orig: continue
                    curr[i] = base
                    new_score, _, _, _, _ = get_detailed_score("".join(curr))
                    if new_score < curr_score:
                        curr_score = new_score
                        improved = True
                    else:
                        curr[i] = orig
        pool.add(("".join(curr), get_detailed_score("".join(curr))))
    
    sorted_pool = sorted(list(pool), key=lambda x: x[1][0])
    optimized_results = []
    for tag, stats in sorted_pool:
        if all(hamming_distance(tag, accepted_tag) >= 2 for accepted_tag, _ in optimized_results):
            optimized_results.append((tag, stats))
        if len(optimized_results) == 5: break
    return optimized_results

def run_ribonn_prediction(utr5_seq, cds_seq, utr3_seq, species, run_df_path, target_columns=None, ensemble_depth=None):
    df = pd.DataFrame({
        'tx_id': ['query_seq'],
        'tx_sequence': [utr5_seq + cds_seq + utr3_seq],
        'utr5_sequence': [utr5_seq],
        'cds_sequence': [cds_seq],
        'utr3_sequence': [utr3_seq],
        'utr5_size': [len(utr5_seq)],
        'cds_size': [len(cds_seq)],
        'utr3_size': [len(utr3_seq)],
        'tx_size': [len(utr5_seq) + len(cds_seq) + len(utr3_seq)]
    })
    
    temp_path = os.path.abspath("temp_ribonn_input.csv")
    df.to_csv(temp_path, sep="\t", index=False)
    
    run_df_object = pd.read_csv(run_df_path)
    
    if ensemble_depth and ensemble_depth < len(run_df_object):
        run_df_object = run_df_object.head(ensemble_depth).reset_index(drop=True)
    
    start_dir = os.getcwd()
    lib_dir = os.path.join(start_dir, "RiboNN")
    
    try:
        os.chdir(lib_dir)
        return predict_using_nested_cross_validation_models(temp_path, species, run_df_object)
    finally:
        os.chdir(start_dir)
        if os.path.exists(temp_path):
            os.remove(temp_path)

# --- UI SETUP ---
st.set_page_config(layout="wide")
st.title("Structurna")

st.sidebar.header("Select Mode")
mode = st.sidebar.radio(
    "Navigation", 
    ["Penalty Benchmark", "Batch Processing", "Optimal Tag Generator", "RiboNN Prediction", "Pareto Frontier Optimization"],
    key="main_app_navigation",
    label_visibility="collapsed" 
)

if mode in ["Penalty Benchmark", "Batch Processing", "Optimal Tag Generator", "Pareto Frontier Optimization"]:
    st.sidebar.header("Calibration Settings")
    mfe_offset_input = st.sidebar.slider("MFE Offset", 0.0, 50.0, 25.0, key=f"mfe_offset_{st.session_state.reset_counter}")
    cap_multiplier_input = st.sidebar.slider("Cap Multiplier", 0.0, 5.0, 2.0, key=f"cap_mult_{st.session_state.reset_counter}")
    kozak_multiplier_input = st.sidebar.slider("Kozak Multiplier", 0.0, 5.0, 3.0, key=f"kozak_mult_{st.session_state.reset_counter}")
    cds_multiplier_input = st.sidebar.slider("CDS Multiplier", 0.0, 5.0, 0.75, key=f"cds_mult_{st.session_state.reset_counter}")
    st.sidebar.button("Reset Calibration", key="reset_calibration_btn", on_click=handle_calibration_reset)

start_excl = st.sidebar.checkbox("CDS sequence already excludes start codon (e.g. ATG)?", value=True)

if mode == "Penalty Benchmark":
    st.subheader("Penalty Benchmark Mode")
    utr_in = st.text_area("Paste 5' UTR:")
    cds_in = st.text_area("Paste CDS:")
    full = st.checkbox("Use Full CDS", value=True)
    
    if st.button("Calculate"):
        tot, mfe, cap, kozak, cds_p = calculate_penalty(
            utr_in, cds_in, full, mfe_offset_input, 
            cap_multiplier_input, kozak_multiplier_input, cds_multiplier_input, start_excl
        )
        st.metric("Total Penalty", f"{tot:.2f}")
        st.markdown("### Specific Penalty Breakdown Stats")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1: st.metric("Minimum Free Energy (MFE)", f"{mfe:.2f}")
        with m_col2: st.metric("Cap GC Bias", f"{cap:.2f}")
        with m_col3: st.metric("Kozak Sequence Alignment", f"{kozak:.0f}")
        with m_col4: st.metric("CDS Deviation Penalty", f"{cds_p:.2f}")

elif mode == "Batch Processing":
    st.subheader("Batch Processing Mode")
    uploaded = st.file_uploader("Upload CSV (tx_id, utr5_sequence, cds_sequence)", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded, dtype=str).fillna('')
        if st.button("Process Batch"):
            res = []
            for _, row in df.iterrows():
                tot, mfe, cap, kozak, cds_p = calculate_penalty(row['utr5_sequence'], row['cds_sequence'], True, mfe_offset_input, cap_multiplier_input, kozak_multiplier_input, cds_multiplier_input, start_excl)
                res.append({"tx_id": row['tx_id'], "Total Penalty": tot, "MFE": mfe, "Cap": cap, "Kozak": kozak, "CDS": cds_p})
            out_df = pd.DataFrame(res)
            st.dataframe(out_df)
            st.download_button("Download CSV", out_df.to_csv(index=False), "results.csv", "text/csv")

elif mode == "Optimal Tag Generator":
    st.subheader("Optimal Tag Generator")
    utr_in = st.text_area("Paste 5' UTR:")
    cds_in = st.text_area("Paste Full CDS:")
    st.session_state.tag_len = st.slider("Tag Length (bp)", 13, 100, st.session_state.tag_len)
    
    col1, col2, col3, col4 = st.columns([2, 4, 0.5, 0.5])
    with col1:
        generate_btn = st.button("Generate Tags", type="primary")
        
    if generate_btn:
        st.session_state.results = generate_optimal_tags(
            cds_in, utr_in, mfe_offset_input, cap_multiplier_input, 
            kozak_multiplier_input, cds_multiplier_input, start_excl, 
            st.session_state.tag_len
        )

    if 'results' in st.session_state and st.session_state.results:
        with col3:
            if st.button("<-"):
                st.session_state.tag_index = (st.session_state.tag_index - 1) % len(st.session_state.results)
                st.rerun()
        with col4:
            if st.button("->"):
                st.session_state.tag_index = (st.session_state.tag_index + 1) % len(st.session_state.results)
                st.rerun()
        
        idx = st.session_state.tag_index
        tag, stats = st.session_state.results[idx]
        st.write(f"### Result {idx + 1} of {len(st.session_state.results)} | Total Penalty: {stats[0]}")
        st.markdown("#### Segment Architecture View")
        st.markdown(f"**Full Blueprint:** :blue[{utr_in}] :green[__[{tag}]__] :orange[{cds_in}]")

elif mode == "RiboNN Prediction":
    st.subheader("RiboNN Prediction")
    try:
        if 'last_processed_df' not in st.session_state: st.session_state['last_processed_df'] = None
        if 'ribonn_complete' not in st.session_state: st.session_state['ribonn_complete'] = False
        if 'accumulated_execution_time' not in st.session_state: st.session_state['accumulated_execution_time'] = 0.0
        if 'persisted_folds_count' not in st.session_state: st.session_state['persisted_folds_count'] = 0
        if 'persisted_species' not in st.session_state: st.session_state['persisted_species'] = ""
        if 'persisted_organs' not in st.session_state: st.session_state['persisted_organs'] = []

        seq_name = st.text_input("Name your sequence set", value="query_seq", key="ribonn_seq_name")
        col_seq1, col_seq2, col_seq3 = st.columns(3)
        with col_seq1: utr_in = st.text_area("Paste 5' UTR:", key="ribonn_utr5")
        with col_seq2: cds_in = st.text_area("Paste CDS:", key="ribonn_cds")
        with col_seq3: utr3_in = st.text_area("Paste 3' UTR:", key="ribonn_utr3")
            
        species = st.selectbox("Species", ["human", "mouse"], key="ribonn_species_select")
        bundled_path = os.path.join(BASE_DIR, "RiboNN", "models", species, "runs.csv")

        st.markdown("### Execution & Optimization Filters")
        run_speed_mode = st.selectbox(
            "Execution Compute Mode & Ensemble Depth:",
            options=[
                "Fast Draft (3 cross-validation folds)",
                "Medium Verification (5 cross-validation folds)",
                "Robust Validation (10 cross-validation folds)",
                "Full Production Ensemble (All 90 cross-validation folds)"
            ], index=0, key="ribonn_compute_speed_mode"
        )
        
        depth_value = 3 if "Fast Draft" in run_speed_mode else 5 if "Medium Verification" in run_speed_mode else 10 if "Robust Validation" in run_speed_mode else 90
        selected_organs = st.multiselect("Isolate Output Matrix to specific Organs/Tissues:", options=list(ORGAN_MAP.keys()), key="ribonn_organ_filter")

        if st.button("Run Prediction", key="run_ribonn_btn"):
            st.session_state['ribonn_complete'] = False
            is_valid, validation_output = clean_and_validate_sequences(utr_in, cds_in, utr3_in)
            
            if not is_valid:
                st.error(validation_output)
            else:
                sanitized_utr5, sanitized_cds, sanitized_utr3 = validation_output
                progress_bar = st.progress(0, text="Initializing production ensemble pipeline...")
                
                start_time = time.time()
                target_columns_list = []
                if selected_organs:
                    for organ in selected_organs: target_columns_list.extend(ORGAN_MAP.get(organ, []))
                target_tuple = tuple(target_columns_list) if target_columns_list else None
                
                def update_progress(current_fold, total_folds):
                    percent = int((current_fold / total_folds) * 100)
                    progress_bar.progress(min(percent, 99), text=f"Processing fold {current_fold}/{total_folds}...")

                raw_data = run_ribonn_prediction_cached(
                    sanitized_utr5, sanitized_cds, sanitized_utr3, species, bundled_path,
                    target_columns=target_tuple, ensemble_depth=depth_value, progress_callback=update_progress  
                )
                
                progress_bar.progress(100, text="Structuring production matrices...")
                flattened_frames = []
                for fold_idx, df_entry in enumerate(raw_data):
                    df_clean = df_entry.copy() if isinstance(df_entry, pd.DataFrame) else pd.DataFrame(df_entry)
                    if 'fold' not in df_clean.columns: df_clean['fold'] = fold_idx
                    flattened_frames.append(df_clean)

                processed_df = pd.concat(flattened_frames, ignore_index=True)                
                processed_df['tx_id'] = [f"{seq_name}-{i+1:02d}" for i in range(len(processed_df))]
                
                tissue_cols = [c for c in processed_df.columns if c.startswith('predicted_TE_')]
                display_tissues = [t for t in target_columns_list if t in processed_df.columns]

                if tissue_cols: processed_df['Global_Mean_TE'] = processed_df[tissue_cols].mean(axis=1)
                if display_tissues: processed_df['Regional_Mean_TE'] = processed_df[display_tissues].mean(axis=1)

                numeric_cols = processed_df.select_dtypes(include=['number']).columns
                mean_row_series = processed_df[numeric_cols].mean()
                summary_row_df = pd.DataFrame(mean_row_series).T
                summary_row_df.insert(0, 'tx_id', 'Query Mean Summary')
                processed_df = pd.concat([processed_df, summary_row_df], ignore_index=True)

                st.session_state['accumulated_execution_time'] = time.time() - start_time
                st.session_state['persisted_folds_count'] = depth_value
                st.session_state['persisted_species'] = species.capitalize()
                st.session_state['persisted_organs'] = selected_organs
                st.session_state['last_processed_df'] = processed_df
                st.session_state['ribonn_complete'] = True
                progress_bar.empty()
                st.rerun()

    except Exception as e:
        st.error("The model pipeline crashed internally:")
        st.exception(e)

    if st.session_state.get('ribonn_complete', False) and st.session_state.get('last_processed_df') is not None:
        persisted_data = st.session_state['last_processed_df'].copy()
        
        st.markdown("### Run Execution Summary")
        with st.container(border=True):
            col1, col2, col3 = st.columns(3)
            with col1: st.markdown(f"**Sequence Model:** RiboNN Ensemble ({st.session_state['persisted_folds_count']} Folds)")
            with col2:
                active_organs = st.session_state['persisted_organs']
                st.markdown(f"**Target Backgrounds:** {st.session_state['persisted_species']} ({', '.join(active_organs) if active_organs else 'All Tissues'})")
            with col3:
                td = st.session_state['accumulated_execution_time']
                st.markdown(f"**Compute Status:** `{f'Completed in {td:.2f}s' if td < 60 else f'Completed in {int(td//60)}m {int(td%60)}s'}`")

        tissue_cols = [c for c in persisted_data.columns if c.startswith('predicted_TE_')]
        meta_cols = [c for c in persisted_data.columns if not c.startswith('predicted_TE_')]
        user_target_tissues = []
        for org in active_organs: user_target_tissues.extend(ORGAN_MAP.get(org, []))
        display_tissues = [t for t in user_target_tissues if t in persisted_data.columns]

        st.markdown("### Matrix Optimization & Views")
        mvf = st.selectbox("Select Matrix Projection:", ["Show Aggregated Means Only", "Show Targeted Organ Columns + Means", "Show Full 89-Tissue Set"], index=1)
        if mvf == "Show Aggregated Means Only": st.dataframe(persisted_data[meta_cols], hide_index=True)
        elif mvf == "Show Targeted Organ Columns + Means": st.dataframe(persisted_data[meta_cols + (display_tissues if display_tissues else tissue_cols[:5])], hide_index=True)
        else: st.dataframe(persisted_data[meta_cols + tissue_cols], hide_index=True)

        st.download_button(label="Download results as CSV", data=persisted_data.to_csv(index=False).encode('utf-8'), file_name=f"{seq_name}_results.csv", mime="text/csv")

        st.markdown("### Tissue-Specific Translation Efficiency (TE) Distribution Profiles")
        if tissue_cols:
            plot_source_data = persisted_data[persisted_data['tx_id'] != 'Query Mean Summary']
            mean_series = plot_source_data[tissue_cols].mean().reset_index()
            mean_series.columns = ["Tissue / Target Organ", "Predicted TE"]
            mean_series["Tissue / Target Organ"] = mean_series["Tissue / Target Organ"].astype(str).str.replace("predicted_TE_", "", case=False).str.replace("_", " ").str.title()
            mean_series = mean_series.sort_values(by="Predicted TE", ascending=False).reset_index(drop=True)
            
            sds = [[0.0, "#A50026"], [0.25, "#FFFFFF"], [0.38, "#4575B4"], [0.50, "#313695"], [1.0, "#1C1E52"]]

            fig_static = px.bar(mean_series.head(15), x="Tissue / Target Organ", y="Predicted TE", labels={"Predicted TE": "Relative TE (Mean)"}, color="Predicted TE", color_continuous_scale=sds, range_color=[-1, 3], title="Top 15 Highest Predicted Expression Domains")
            fig_static.update_layout(template="plotly_white", xaxis_tickangle=-45)
            
            fig_full_static = px.bar(mean_series, x="Tissue / Target Organ", y="Predicted TE", labels={"Predicted TE": "Relative TE (Mean)"}, color="Predicted TE", color_continuous_scale=sds, range_color=[-1, 3], title="Exhaustive Expression Matrix Background (Full 89 Systems)")
            fig_full_static.update_layout(template="plotly_white", xaxis_tickangle=-45)

            graph_view_format = st.radio("Select Chart View Layout:", ["Show Top 15 Highest Expression Domains", "Show Exhaustive 89-Tissue Expression Matrix Background"], horizontal=True, key="ribonn_graph_layout")
            if "Top 15" in graph_view_format: st.plotly_chart(fig_static, use_container_width=True)
            else: st.plotly_chart(fig_full_static, use_container_width=True)

        if st.button("Clear Active Results"):
            st.session_state['ribonn_complete'] = False
            st.session_state['last_processed_df'] = None
            st.rerun()

elif mode == "Pareto Frontier Optimization":
    st.subheader("Pareto Frontier Optimization")
    
    seq_name = st.text_input("Name your optimization set", value="pareto_query", key="pareto_seq_name")
    col_seq1, col_seq2, col_seq3 = st.columns(3)
    with col_seq1: st.text_area("Paste 5' UTR:", key="pareto_utr5")
    with col_seq2: st.text_area("Paste CDS:", key="pareto_cds")
    with col_seq3: st.text_area("Paste 3' UTR:", key="pareto_utr3")
        
    tag_len_selected = st.slider("Tag Length (bp):", min_value=3, max_value=99, value=12, step=3, key="pareto_tag_len_slider")
    species = st.selectbox("Species Background Model", ["human", "mouse"], key="pareto_species_select")
    bundled_path = os.path.join(BASE_DIR, "RiboNN", "models", species, "runs.csv")

    st.markdown("### Expression Verification Filters")
    run_speed_mode = st.selectbox(
        "Execution Compute Mode & Ensemble Depth:",
        options=[
            "Fast Draft (3 cross-validation folds)",
            "Medium Verification (5 cross-validation folds)",
            "Robust Validation (10 cross-validation folds)",
            "Full Production Ensemble (All 90 cross-validation folds)"
        ], index=0, key="pareto_compute_speed_mode"
    )
    depth_value = 3 if "Fast Draft" in run_speed_mode else 5 if "Medium Verification" in run_speed_mode else 10 if "Robust Validation" in run_speed_mode else 90
    selected_organs = st.multiselect("Isolate Target Evaluation Organs:", options=list(ORGAN_MAP.keys()), key="pareto_organ_filter")

    # --- SESSION STATE PERSISTENCE ARRAYS ---
    if "pareto_complete" not in st.session_state: st.session_state["pareto_complete"] = False
    if "pareto_df" not in st.session_state: st.session_state["pareto_df"] = None
    if "best_compromise" not in st.session_state: st.session_state["best_compromise"] = None
    if "pareto_processed_df" not in st.session_state: st.session_state["pareto_processed_df"] = None
    if "base_x" not in st.session_state: st.session_state["base_x"] = 0.0
    if "base_y" not in st.session_state: st.session_state["base_y"] = 0.0

    if st.button("Generate Pareto Frontier Pipeline", key="run_pareto_btn"):
        is_valid, validation_output = clean_and_validate_sequences(
            st.session_state["pareto_utr5"], 
            st.session_state["pareto_cds"], 
            st.session_state["pareto_utr3"]
        )
        if not is_valid:
            st.error(validation_output)
        else:
            sanitized_utr5, sanitized_cds, sanitized_utr3 = validation_output
            
            # 1. Calculate genuine structural baseline penalty score
            orig_tot, _, _, _, _ = calculate_penalty(
                sanitized_utr5, sanitized_cds, True, mfe_offset_input, 
                cap_multiplier_input, kozak_multiplier_input, cds_multiplier_input, start_excl
            )
            
            st.info("Generating optimal tags...")
            generated_tags = generate_optimal_tags(
                sanitized_cds, sanitized_utr5, mfe_offset_input, cap_multiplier_input, 
                kozak_multiplier_input, cds_multiplier_input, start_excl, tag_len_selected
            )
            
            target_columns_list = []
            if selected_organs:
                for organ in selected_organs: target_columns_list.extend(ORGAN_MAP.get(organ, []))
            target_tuple = tuple(target_columns_list) if target_columns_list else None
            
            total_candidates = 1 + len(generated_tags)
            progress_bar = st.progress(0, text="Initializing production multi-variant validation pipeline...")
            
            # --- PHASE A: BASELINE PROFILE EXECUTION ---
            def update_progress_base(current_fold, total_folds):
                percent = int(((current_fold / total_folds) / total_candidates) * 100)
                progress_bar.progress(min(percent, 99), text=f"Evaluating Baseline Set: fold {current_fold}/{total_folds}...")

            raw_base_data = run_ribonn_prediction_cached(
                sanitized_utr5, sanitized_cds, sanitized_utr3, species, bundled_path,
                target_columns=target_tuple, ensemble_depth=depth_value, progress_callback=update_progress_base  
            )
            
            base_frames = []
            for fold_idx, df_entry in enumerate(raw_base_data):
                df_clean = df_entry.copy() if isinstance(df_entry, pd.DataFrame) else pd.DataFrame(df_entry)
                base_frames.append(df_clean)
            processed_base_df = pd.concat(base_frames, ignore_index=True)
            
            tissue_cols = [c for c in processed_base_df.columns if c.startswith('predicted_TE_')]
            display_tissues = [t for t in target_columns_list if t in processed_base_df.columns]
            active_mean_col = 'Regional_Mean_TE' if display_tissues else 'Global_Mean_TE'
            
            if tissue_cols: processed_base_df['Global_Mean_TE'] = processed_base_df[tissue_cols].mean(axis=1)
            if display_tissues: processed_base_df['Regional_Mean_TE'] = processed_base_df[display_tissues].mean(axis=1)
            base_te = processed_base_df[active_mean_col].mean() if len(processed_base_df) > 0 else 0.0
            
            pareto_data = [{
                "Candidate": "Original Sequence (Reference)",
                "Sequence": "Original Baseline Context",
                "Local Penalty (X)": orig_tot,
                "Predicted TE (Y)": base_te
            }]
            
            # --- PHASE B: TRUE VARIANT ENSEMBLE TESTING ---
            for i, (tag_seq, stats) in enumerate(generated_tags):
                local_penalty = stats[0]
                cand_idx = i + 1
                
                if start_excl:
                    variant_cds = tag_seq + sanitized_cds
                else:
                    variant_cds = tag_seq + sanitized_cds[3:]
                
                def update_progress_var(current_fold, total_folds):
                    base_percent = (cand_idx / total_candidates)
                    fold_fraction = (current_fold / total_folds) / total_candidates
                    percent = int((base_percent + fold_fraction) * 100)
                    progress_bar.progress(min(percent, 99), text=f"Evaluating Variant {cand_idx}/{len(generated_tags)}: fold {current_fold}/{total_folds}...")

                raw_var_data = run_ribonn_prediction_cached(
                    sanitized_utr5, variant_cds, sanitized_utr3, species, bundled_path,
                    target_columns=target_tuple, ensemble_depth=depth_value, progress_callback=update_progress_var  
                )
                
                var_frames = []
                for df_entry in raw_var_data:
                    df_clean = df_entry.copy() if isinstance(df_entry, pd.DataFrame) else pd.DataFrame(df_entry)
                    var_frames.append(df_clean)
                processed_var_df = pd.concat(var_frames, ignore_index=True)
                
                if tissue_cols: processed_var_df['Global_Mean_TE'] = processed_var_df[tissue_cols].mean(axis=1)
                if display_tissues: processed_var_df['Regional_Mean_TE'] = processed_var_df[display_tissues].mean(axis=1)
                
                true_var_te = processed_var_df[active_mean_col].mean() if len(processed_var_df) > 0 else 0.0
                
                pareto_data.append({
                    "Candidate": f"Tag Candidate {cand_idx}",
                    "Sequence": tag_seq,
                    "Local Penalty (X)": local_penalty,
                    "Predicted TE (Y)": true_var_te
                })
                
            progress_bar.progress(100, text="Compiling exact multi-parametric structures...")
            pareto_df = pd.DataFrame(pareto_data).sort_values(by="Local Penalty (X)").reset_index(drop=True)
            
            frontier_indices = []
            for idx, row in pareto_df.iterrows():
                dominated = False
                for other_idx, other_row in pareto_df.iterrows():
                    if other_idx == idx: continue
                    if (other_row["Local Penalty (X)"] <= row["Local Penalty (X)"] and 
                        other_row["Predicted TE (Y)"] >= row["Predicted TE (Y)"] and 
                        (other_row["Local Penalty (X)"] < row["Local Penalty (X)"] or 
                         other_row["Predicted TE (Y)"] > row["Predicted TE (Y)"])):
                        dominated = True
                        break
                if not dominated:
                    frontier_indices.append(idx)
            
            pareto_df["Classification"] = "Dominated Candidate (Suboptimal)"
            pareto_df.loc[frontier_indices, "Classification"] = "Pareto Frontier (Elite Optimal)"

            st.session_state["base_x"] = orig_tot
            st.session_state["base_y"] = base_te

            def evaluate_quadrant(row):
                if row["Candidate"] == "Original Sequence (Reference)":
                    return "Original Reference Baseline"
                elif row["Local Penalty (X)"] <= st.session_state["base_x"] and row["Predicted TE (Y)"] >= st.session_state["base_y"]:
                    return "Strict Win (Lower Penalty & Higher TE)"
                elif row["Predicted TE (Y)"] >= st.session_state["base_y"]:
                    return "Expression Upgrade (Higher TE, Higher Penalty)"
                else:
                    return "Sub-Baseline Candidate"

            pareto_df["Design_Quadrant"] = pareto_df.apply(evaluate_quadrant, axis=1)

            st.session_state["pareto_df"] = pareto_df
            st.session_state["best_compromise"] = pareto_df[pareto_df["Classification"] == "Pareto Frontier (Elite Optimal)"].sort_values(by="Predicted TE (Y)", ascending=False).iloc[0]
            st.session_state["pareto_processed_df"] = processed_base_df
            st.session_state["pareto_complete"] = True
            
            progress_bar.empty()
            st.rerun()

    # --- RENDER INTERFACE OUTSIDE THE BUTTON LOOP ---
    if st.session_state.get("pareto_complete") and st.session_state.get("pareto_df") is not None:
        pareto_df = st.session_state["pareto_df"]
        best_compromise = st.session_state["best_compromise"]
        processed_df = st.session_state["pareto_processed_df"]
        tissue_cols = [c for c in processed_df.columns if c.startswith('predicted_TE_')]

        frontier_df = pareto_df[pareto_df["Classification"] == "Pareto Frontier (Elite Optimal)"].sort_values(by="Local Penalty (X)")

        fig_pareto = px.scatter(
            pareto_df, x="Local Penalty (X)", y="Predicted TE (Y)", 
            color="Design_Quadrant",
            symbol="Classification",
            hover_data={
                "Candidate": True,
                "Design_Quadrant": True,
                "Classification": True,
                "Local Penalty (X)": ":.2f",
                "Predicted TE (Y)": ":.4f",
                "Sequence": False  
            },
            color_discrete_map={
                "Original Reference Baseline": "#1E293B",
                "Strict Win (Lower Penalty & Higher TE)": "#059669",
                "Expression Upgrade (Higher TE, Higher Penalty)": "#3B82F6",
                "Sub-Baseline Candidate": "#94A3B8"
            },
            symbol_map={
                "Pareto Frontier (Elite Optimal)": "star",
                "Dominated Candidate (Suboptimal)": "circle"
            },
            labels={
                "Local Penalty (X)": "Local Biophysical Penalty", 
                "Predicted TE (Y)": "True RiboNN Predicted TE",
                "Design_Quadrant": "Design Quadrant",
                "Classification": "Frontier Status"
            },
            title="Design Optimization Architecture Space (Reference Threshold Quadrants)"
        )
        
        fig_pareto.add_scatter(
            x=frontier_df["Local Penalty (X)"], y=frontier_df["Predicted TE (Y)"], 
            mode="lines", name="Pareto Optimal Frontier", 
            line=dict(color="#EF4444", width=2, dash="dash")
        )
        
        fig_pareto.add_vline(
            x=st.session_state["base_x"], line_width=1.5, line_dash="dot", line_color="#64748B",
            annotation_text="Original Penalty Baseline", annotation_position="top left"
        )
        
        fig_pareto.add_hline(
            y=st.session_state["base_y"], line_width=1.5, line_dash="dot", line_color="#64748B",
            annotation_text="Original TE Baseline", annotation_position="bottom right"
        )

        fig_pareto.update_layout(template="plotly_white", legend=dict(title="Candidate Evaluation Metrics"))
        st.plotly_chart(fig_pareto, use_container_width=True)
        
        upgraded_options = pareto_df[
            (pareto_df["Predicted TE (Y)"] > st.session_state["base_y"]) & 
            (pareto_df["Candidate"] != "Original Sequence (Reference)")
        ]
        
        st.markdown("### Optimization Strategy Insights")
        if not upgraded_options.empty:
            highest_te_variant = upgraded_options.sort_values(by="Predicted TE (Y)", ascending=False).iloc[0]
            col_rec1, col_rec2 = st.columns(2)
            with col_rec1:
                st.success(
                    f"**Maximum Expression Target**: `{highest_te_variant['Candidate']}` "
                    f"yielded the absolute highest TE outcome of **{highest_te_variant['Predicted TE (Y)']:.4f}** "
                    f"(a **+{((highest_te_variant['Predicted TE (Y)'] - st.session_state['base_y'])/st.session_state['base_y'])*100:.1f}%** extension improvement)."
                )
            with col_rec2:
                st.info(
                    f"**Balanced Frontier Target**: `{best_compromise['Candidate']}` "
                    f"maintains a TE output score of "
                    f"**{best_compromise['Predicted TE (Y)']:.4f}** with a structural penalty of **{best_compromise['Local Penalty (X)']}**."
                )
        else:
            st.warning("No generated variants outperformed the base sequence expression threshold.")

        st.markdown("### Base Model Tissue Profile Metrics")
        if tissue_cols:
            mean_series = processed_df[tissue_cols].mean().reset_index()
            mean_series.columns = ["Tissue / Target Organ", "Predicted TE"]
            mean_series["Tissue / Target Organ"] = mean_series["Tissue / Target Organ"].astype(str).str.replace("predicted_TE_", "", case=False).str.replace("_", " ").str.title()
            mean_series = mean_series.sort_values(by="Predicted TE", ascending=False).reset_index(drop=True)
            
            sds = [[0.0, "#A50026"], [0.25, "#FFFFFF"], [0.38, "#4575B4"], [0.50, "#313695"], [1.0, "#1C1E52"]]
            
            fig_p_static = px.bar(mean_series.head(15), x="Tissue / Target Organ", y="Predicted TE", color="Predicted TE", color_continuous_scale=sds, range_color=[-1, 3], title="Top 15 Highest Expression Background Targets")
            fig_p_static.update_layout(template="plotly_white", xaxis_tickangle=-45)
            
            fig_p_full = px.bar(mean_series, x="Tissue / Target Organ", y="Predicted TE", color="Predicted TE", color_continuous_scale=sds, range_color=[-1, 3], title="Exhaustive Base Target Set Profile (Full 89)")
            fig_p_full.update_layout(template="plotly_white", xaxis_tickangle=-45)
            
            graph_toggle_pareto = st.radio("Select Expression Graph Projection:", ["Show Top 15 Highest Expression Domains", "Show 89-Tissue Expression Background"], horizontal=True, key="pareto_graph_display_layout_toggle")
            if "Top 15" in graph_toggle_pareto: 
                st.plotly_chart(fig_p_static, use_container_width=True)
            else: 
                st.plotly_chart(fig_p_full, use_container_width=True)

        if st.button("Clear Optimization Results", key="clear_pareto_btn"):
            st.session_state["pareto_complete"] = False
            st.session_state["pareto_df"] = None
            st.session_state["best_compromise"] = None
            st.session_state["pareto_processed_df"] = None
            st.session_state["base_x"] = 0.0
            st.session_state["base_y"] = 0.0
            st.rerun()