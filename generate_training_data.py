import json
import os
import re
import time
import torch

import multiprocessing as mp
import pandas as pd

from sentence_transformers import SentenceTransformer, util
from torch.utils.data import DataLoader

from language_classifier.language_classifier import LanguageClassifier


def clean_text(text, skip_cleaning=False):
    allow_numbers = True

    if not skip_cleaning:
        if allow_numbers:
            allowed_chars = r"[^a-zA-ZÀ-ÖØ-öø-ÿ0-9.,;:!?()'\"-]"
        else:
            allowed_chars = r"[^a-zA-ZÀ-ÖØ-öø-ÿ.,;:!?()'\"-]"
        text = re.sub(allowed_chars, ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

    return text


def get_files_for_publication(pub_number, fr_eng_correlation_df):
    row = fr_eng_correlation_df.loc[fr_eng_correlation_df['pub_number'] == pub_number]
    if not row.empty:
        filename_fr = row['filename_fr'].values[0]
        filename_en = row['filename_en'].values[0]
        return filename_fr, filename_en
    return None, None


def get_json_file_link(parsed_docs_folder, pdf_filename):
    if pdf_filename.endswith(".pdf"):
        json_filename = pdf_filename + ".json"
        for root, _, files in os.walk(parsed_docs_folder):
            if json_filename in files:
                return os.path.join(root, json_filename)
    return None


def extract_text_from_single_file(json_file, target_language, clf, skip_cleaning=False):
    min_block_length = 10
    max_block_length = 500

    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    if 'text' not in data:
        raise KeyError(f"The key 'text' is missing in the JSON file: {json_file}")

    full_text = clean_text(data['text'], skip_cleaning)
    text_blocks = re.split(r'(?<![;,])[.?!]\s|\n\n', full_text)
    text = []

    for block in text_blocks:
        block = block.strip()
        if len(block) < min_block_length or len(block) > max_block_length:
            continue

        if clf.classify(block) == target_language:
            text.append(block + '. ')

    return " ".join(text)


def extract_both_languages_from_two_files(json_file_fr, json_file_en, clf):
    return (extract_text_from_single_file(json_file_fr, "fr", clf),
            extract_text_from_single_file(json_file_en, "en", clf))


def extract_both_languages_from_single_file(json_file, clf):
    min_block_length = 10
    max_block_length = 500

    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    if 'text' not in data:
        raise KeyError(f"The key 'text' is missing in the JSON file: {json_file}")

    full_text = clean_text(data['text'], skip_cleaning=False)
    text_blocks = re.split(r'(?<![;,])[.?!]\s|\n\n', full_text)
    text_fr, text_en = [], []

    for block in text_blocks:
        block = block.strip()
        if len(block) < min_block_length or len(block) > max_block_length:
            continue

        if clf.classify(block) == "fr":
            text_fr.append(block + '. ')
        elif clf.classify(block) == "en":
            text_en.append(block + '. ')

    return " ".join(text_fr), " ".join(text_en)


def create_sentences(text_fr, text_en):  # TODO: if we forget about \n\n, maybe this would exclude a bunch of appendix junk
    sentences_fr = [x.strip() for x in re.split(r'(?<![;,])[.?!]\s|\n\n', text_fr) if x != ""]
    sentences_en = [x.strip() for x in re.split(r'(?<![;,])[.?!]\s|\n\n', text_en) if x != ""]

    return sentences_fr, sentences_en


def create_similarity_matrix(sentences_fr, sentences_en, sentence_encoder, device):
    max_batch_size = 1024
    min_batch_size = 8

    embeddings_fr = sentence_encoder.encode(
        sentences_fr,
        convert_to_tensor=True,
        batch_size=min(max_batch_size, max(min_batch_size, len(sentences_fr))),
        device=device
    )
    embeddings_en = sentence_encoder.encode(
        sentences_en,
        convert_to_tensor=True,
        batch_size=min(max_batch_size, max(min_batch_size, len(sentences_en))),
        device=device
    )

    return util.pytorch_cos_sim(embeddings_fr, embeddings_en)


def align_sentences(sim_matrix, device):
    threshold = 0.7
    n, m = sim_matrix.shape

    weights = torch.where(sim_matrix >= threshold, sim_matrix, torch.tensor(0.0, device=device))
    dp = torch.zeros((n + 1, m + 1), dtype=torch.float32, device=device)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            score_match = dp[i - 1, j - 1] + weights[i - 1, j - 1]
            score_skip_fr = dp[i - 1, j]
            score_skip_en = dp[i, j - 1]

            dp[i, j] = torch.max(torch.tensor([score_match, score_skip_fr, score_skip_en], device=device))

    aligned_pairs = []
    i, j = n, m
    while i > 0 and j > 0:
        current_val = dp[i, j]
        if torch.isclose(current_val, dp[i - 1, j]):
            i -= 1
        elif torch.isclose(current_val, dp[i, j - 1]):
            j -= 1
        else:
            similarity_score = sim_matrix[i - 1, j - 1].item()
            if weights[i - 1, j - 1] > 0:
                aligned_pairs.append((i - 1, j - 1, similarity_score))
            i -= 1
            j -= 1

    aligned_pairs.reverse()

    return aligned_pairs


def text_from_coordinates(aligned_pairs, sentences_fr, sentences_en, pub_number):
    correlated_list = list()
    for i, j, similarity in aligned_pairs:
        correlated_list.append((pub_number, sentences_fr[i], sentences_en[j], round(similarity, 3)))

    return correlated_list


def correlate_and_clean_text(text_fr, text_en, pub_number, sentence_encoder, device):
    sentences_fr, sentences_en = create_sentences(text_fr, text_en)
    similarity_matrix = create_similarity_matrix(sentences_fr, sentences_en, sentence_encoder, device)
    aligned_pairs = align_sentences(similarity_matrix, device)

    return text_from_coordinates(aligned_pairs, sentences_fr, sentences_en, pub_number)


def process_row(row_tuple, device, language_classifier, sentence_encoder, skip_abstract_only_translations=False):
    parsed_docs_folder = os.path.join("..", "ParsedPublications")
    index, row = row_tuple
    pub_number = row['pub_number']
    filename_fr, filename_en = row['filename_fr'], row['filename_en']

    if filename_fr == "WITHDRAWN" and filename_en == "WITHDRAWN":
        return None

    fr_link = get_json_file_link(parsed_docs_folder, filename_fr)
    if fr_link is None:
        return None

    if filename_fr == filename_en:
        text_fr, text_en = extract_both_languages_from_single_file(fr_link, language_classifier)
    else:
        en_link = get_json_file_link(parsed_docs_folder, filename_en)
        if en_link is None:
            return None
        text_fr, text_en = extract_both_languages_from_two_files(fr_link, en_link, language_classifier)

    # low-quality text criteria
    max_ratio = 2  # abstract only translations to (potentially) exclude
    min_char = 1000  # low quality, bad OCR, or incomplete transcription / parsing
    len_fr, len_en = len(text_fr), len(text_en)

    if len_fr == 0 or len_en == 0:
        return None
    elif skip_abstract_only_translations:
        if len(text_fr) / len(text_en) > max_ratio or len(text_en) / len(text_fr) > max_ratio:
            return None
    elif len(text_fr) < min_char or len(text_en) < min_char:
        return None

    return correlate_and_clean_text(text_fr, text_en, pub_number, sentence_encoder, device)


def process_row_wrapper(args):
    row, device, language_classifier, sentence_encoder, skip_abstracts = args
    return process_row(row, device, language_classifier, sentence_encoder, skip_abstracts)


def print_time_estimate(start_time, n, n_total):
    if n == 0:
        print(f"\n{n}/{n_total} complete.", end="... ")
        return

    time_elapsed = int(time.time() - start_time)
    time_remaining = int((n_total / n) * time_elapsed)

    time_elapsed_text = f"{time_elapsed // 3600}h:{(time_elapsed % 3600) // 60:02d}m"
    time_remaining_text = f"{time_remaining // 3600}h:{(time_remaining % 3600) // 60:02d}m"

    print(f"\n{n}/{n_total} complete at {time_elapsed_text}. Estimated {time_remaining_text} remaining.", end="... ")


def print_status(start_time, n, n_total):
    if n % 10 == 0:
        if n % 100 == 0:
            print_time_estimate(start_time, n, n_total)
        else:
            print(f"{n}", end="... ")


def main():
    start_time = time.time()

    fr_eng_correlation_df = pd.read_csv("fr_eng_correlation_data.csv")
    fr_eng_correlation_df = fr_eng_correlation_df[['pub_number', 'filename_fr', 'filename_en']]
    rows = list(fr_eng_correlation_df.iterrows())
    n_rows = len(rows)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = max(1, os.cpu_count() - 2)

    language_classifier = LanguageClassifier()
    sentence_encoder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2').to(device)

    print(f'\nUsing device: {device}')
    print(f"Using {num_workers} CPU cores.\n")
    
    args_list = [(row, device, language_classifier, sentence_encoder, False) for row in rows]

    print("=========== PROCESSING matched_df ===========")

    with mp.Pool(num_workers) as pool:
        results = []
        for i, result in enumerate(pool.imap_unordered(process_row_wrapper, args_list)):
            if result:
                results.extend(result)

            print_status(start_time, i, n_rows * 2)

    matched_df = pd.DataFrame(results, columns=['pub_number', 'fr', 'en', 'similarity'])
    matched_df.to_pickle("matched_data.pickle")
    print(f"\nProcessing matched_df complete!\n")

    args_list_wo = [(row, device, language_classifier, sentence_encoder, True) for row in rows]

    print("=========== PROCESSING matched_df_wo_abstracts ===========")

    with mp.Pool(num_workers) as pool:
        results_wo = []
        for i, result in enumerate(pool.imap_unordered(process_row_wrapper, args_list_wo)):
            if result:
                results_wo.extend(result)

            print_status(start_time, i, n_rows * 2)

    matched_df_wo_abstracts = pd.DataFrame(results_wo, columns=['pub_number', 'fr', 'en', 'similarity'])
    matched_df_wo_abstracts.to_pickle("matched_data_wo_abstracts.pickle")
    print(f"\nProcessing matched_df_wo_abstracts complete!\n")


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
