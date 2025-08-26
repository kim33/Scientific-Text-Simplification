import nltk
import textstat
from nltk.translate.bleu_score import sentence_bleu
from lens import download_model, LENS_SALSA
from evaluate import load
import argparse
import torch
from bert_score import score
from sentence_transformers import SentenceTransformer, util
import numpy as np
from simalign import SentenceAligner
import spacy


#DEVICES = [0] if torch.cuda.is_available() else [-1]

#lens_path = download_model("davidheineman/lens-salsa")
#lens_eval = LENS_SALSA(lens_path)
sari = load("sari")

print(torch.__version__)                  # PyTorch version
print(torch.version.cuda)                 # CUDA version PyTorch was built with
print(torch.backends.cudnn.version())     # cuDNN version
print(torch.cuda.is_available())          # Whether CUDA GPU is available

max_len_allowed = 128  # or get from model/config if known

# Function to compute BLEU score between reference and candidate
def compute_bleu(references, candidates):
    assert len(references) == len(candidates), "Line count mismatch"

    total_score = 0
    for ref, cand in zip(references, candidates):
        ref_tokens = [nltk.word_tokenize(ref.lower())]
        cand_tokens = nltk.word_tokenize(cand.lower())
        score = sentence_bleu(ref_tokens, cand_tokens)
        total_score += score

    return total_score / len(references)

# Function to compute Flesch-Kincaid Readability and Grade level
def compute_fk_metrics(lines):
    full_text = " ".join(lines)
    fk_grade_level = textstat.flesch_kincaid_grade(full_text)
    fk_reading_ease = textstat.flesch_reading_ease(full_text)
    return fk_grade_level, fk_reading_ease

# Read files
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def read_file_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file if line.strip()]

def read_file_sari(file_path: str) -> list:
    """Read a text file and return a list of sentences."""
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file.readlines()]
    return lines

def calculate_sari_from_files(original_sentences: list, simplified_sentences: list, references: list = None) -> float:
    if references is None:
        # Create dummy references for each sentence
        dummy_refs = [[[]] for _ in range(len(original_sentences))]
        sari_score = sari.compute(sources=original_sentences, predictions=simplified_sentences, references=dummy_refs)
        print("Warning: No references provided. SARI score is calculated without the 'keep' operation.")
    else:
        sari_score = sari.compute(sources=original_sentences, predictions=simplified_sentences, references=references)
    return sari_score

def truncate_sentence_nltk(sentence, max_len):
    tokens = nltk.word_tokenize(sentence)
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    return " ".join(tokens)

def compare_ttr(original_sentences, simplified_sentences):
    def calculate_ttr(sentences):
        tokens = []
        for sentence in sentences:
            tokens.extend(sentence.split())
        total_tokens = len(tokens)
        unique_tokens = len(set(tokens))
        return unique_tokens / total_tokens if total_tokens > 0 else 0

    orig_ttr = calculate_ttr(original_sentences)
    simp_ttr = calculate_ttr(simplified_sentences)

    print(f"Original TTR     : {orig_ttr:.4f}")
    print(f"Simplified TTR   : {simp_ttr:.4f}")
    print(f"TTR Difference   : {orig_ttr - simp_ttr:.4f}")

    return orig_ttr, simp_ttr

def compression_ratio(original_sentences, simplified_sentences):
    ratios = [len(simp.split()) / len(orig.split()) if len(orig.split()) > 0 else 0
              for orig, simp in zip(original_sentences, simplified_sentences)]
    return sum(ratios) / len(ratios)

def avg_word_len(sentences):
    return sum(len(word) for s in sentences for word in s.split()) / \
           sum(len(s.split()) for s in sentences)

def avg_sentence_len(sentences):
    return sum(len(s.split()) for s in sentences) / len(sentences)

def compute_paraphrase_similarity(original_sentences, simplified_sentences, model_name="paraphrase-MiniLM-L6-v2"):
    assert len(original_sentences) == len(simplified_sentences), "Mismatch in sentence count."

    print("Loading paraphrase detection model...")
    model = SentenceTransformer(model_name)

    print("Encoding sentences...")
    orig_embeddings = model.encode(original_sentences, convert_to_tensor=True, batch_size=32, show_progress_bar=True)
    simp_embeddings = model.encode(simplified_sentences, convert_to_tensor=True, batch_size=32, show_progress_bar=True)

    print("Computing cosine similarities...")
    cosine_similarities = util.cos_sim(orig_embeddings, simp_embeddings).diagonal()

    avg_similarity = cosine_similarities.mean().item()

    print(f"Average Paraphrase Cosine Similarity: {avg_similarity:.4f}")
    return avg_similarity

# ---------------- SAMSA (spaCy dependency-based approximation) ----------------
import numpy as np
import nltk
nltk.download('punkt', quiet=True)

CORE_DEPS = {
    "nsubj","nsubjpass","csubj","csubjpass","obj","dobj","iobj","attr",
    "oprd","xcomp","ccomp","obl"  # keep broad; you can trim if needed
}

def _scene_spans_from_dep(doc):
    """
    Extract 'scenes' from a spaCy Doc as dictionaries of token-index sets.
    Each scene has at least a predicate 'V'; arguments are grouped as ARG1, ARG2...
    Returns: (tokens:list[str], scenes:list[dict[str,set[int]]])
    """
    tokens = [t.text for t in doc]
    scenes = []
    for tok in doc:
        if tok.pos_ in {"VERB","AUX"}:
            # collect core argument subtrees attached to this verb
            args = []
            for child in tok.children:
                if child.dep_ in CORE_DEPS:
                    # use the span of the dependent's subtree as the argument
                    arg_idxs = set([t.i for t in child.subtree])
                    # filter punctuation-only args
                    if any(doc[i].is_alpha for i in arg_idxs):
                        args.append(arg_idxs)
            # keep predicates that look like true clauses (has ≥1 core arg or is root)
            if args or tok.dep_ == "ROOT":
                scene = {"V": {tok.i}}
                # name args ARG1, ARG2, ... in order of appearance
                for k, s in enumerate(sorted(args, key=lambda s: min(s))):
                    scene[f"ARG{k+1}"] = s
                scenes.append(scene)
    return tokens, scenes

def _align_src_to_tgt_indices(aligner, src_tokens, tgt_tokens):
    """
    Returns mapping: src_index -> set(tgt_indices) using SimAlign ('itermax').
    """
    aligns = aligner.get_word_aligns(src_tokens, tgt_tokens)
    m = {}
    for i, j in aligns.get("itermax", []):
        m.setdefault(i, set()).add(j)
    return m

def _element_preserved(element_src_idxs, src2tgt_map):
    """True if any source token in the element aligns to some target token."""
    return any((i in src2tgt_map and src2tgt_map[i]) for i in element_src_idxs)

def _scene_preservation_fraction(scene, src2tgt_map):
    """
    Fraction of scene elements preserved: count V and each ARG* present.
    """
    labels = list(scene.keys())  # e.g., ["V","ARG1","ARG2"]
    if not labels:
        return 0.0
    preserved = 0
    for lab in labels:
        if _element_preserved(scene[lab], src2tgt_map):
            preserved += 1
    return preserved / len(labels)

def _samsa_for_pair(aligner, nlp, src_sent, tgt_line):
    """
    src_sent: one source sentence (string)
    tgt_line: simplified line possibly containing multiple sentences (string)
    Returns (samsa, split_term, structure_term)
    """
    # parse source
    src_doc = nlp(src_sent)
    src_tokens, scenes = _scene_spans_from_dep(src_doc)
    if len(scenes) == 0:
        return 0.0, 0.0, 0.0

    # split target into sentences and tokenize
    tgt_doc = nlp(tgt_line)
    tgt_sents = [span.text for span in tgt_doc.sents] or [tgt_line]
    tgt_tokenized = [[t.text for t in nlp.make_doc(s)] for s in tgt_sents]

    # align source to each target sentence separately
    src2tgt_maps = [
        _align_src_to_tgt_indices(aligner, src_tokens, tgt_tokens)
        for tgt_tokens in tgt_tokenized
    ]

    # assign each scene to the target sentence that best preserves it
    assigned_idxs = []
    pres_scores = []
    for scene in scenes:
        best_j, best_frac = 0, -1.0
        for j, m in enumerate(src2tgt_maps):
            frac = _scene_preservation_fraction(scene, m)
            if frac > best_frac:
                best_frac = frac
                best_j = j
        assigned_idxs.append(best_j)
        pres_scores.append(best_frac)

    # splitting term: how many different target sentences received scenes
    split_term = min(1.0, len(set(assigned_idxs)) / len(scenes))
    struct_term = float(np.mean(pres_scores)) if pres_scores else 0.0
    return split_term * struct_term, split_term, struct_term

def compute_samsa(original_sentences, simplified_sentences, batch_init_cache={}):
    """
    Compute corpus-level SAMSA (avg), plus average split and structure terms.
    """
    assert len(original_sentences) == len(simplified_sentences), "Line count mismatch."

    # lazy init & cache heavy components
    if "nlp" not in batch_init_cache:
        try:
            batch_init_cache["nlp"] = spacy.load("en_core_web_sm")
        except Exception as e:
            raise RuntimeError("spaCy model 'en_core_web_sm' not found. Run: "
                               "`python -m spacy download en_core_web_sm`") from e
    if "aligner" not in batch_init_cache:
        # BERT-based aligner; uses your installed torch
        batch_init_cache["aligner"] = SentenceAligner(model="bert", token_type="bpe", matching_methods="mai")

    nlp = batch_init_cache["nlp"]
    aligner = batch_init_cache["aligner"]

    samsa_vals, split_vals, struct_vals = [], [], []
    for src, tgt in zip(original_sentences, simplified_sentences):
        s, sp, st = _samsa_for_pair(aligner, nlp, src, tgt)
        samsa_vals.append(s); split_vals.append(sp); struct_vals.append(st)

    return float(np.mean(samsa_vals)), float(np.mean(split_vals)), float(np.mean(struct_vals))
# -----------------------------------------------------------------



def evaluate(original, simplified, reference=None):
    original_text = read_file_lines(original)
    candidate_text = read_file_lines(simplified)
    bleu_score = compute_bleu(original_text, candidate_text)
    sim_fk_grade_level, sim_fk_reading_ease = compute_fk_metrics(candidate_text)

    # Read files line by line
    original_sentences = read_file_lines(original)
    simplified_sentences = read_file_lines(simplified)

    # Handle optional reference
    if reference:
        reference_sentences = read_file_lines(reference)
        assert len(reference_sentences) == len(original_sentences), "Reference and original file must align"
        refs = [[ref] for ref in reference_sentences]
    else:
        refs = None

    sari_score = calculate_sari_from_files(original_sentences, simplified_sentences, refs)


    bertscore = load("bertscore")
    results = bertscore.compute(model_type="distilbert-base-uncased", predictions=simplified_sentences, references=original_sentences, lang="en", device="cpu")


    avg_precision = sum(results["precision"]) / len(results["precision"])
    avg_recall = sum(results["recall"]) / len(results["recall"])
    avg_f1 = sum(results["f1"]) / len(results["f1"])

    #P, R, F1 = score(simplified_sentences, original_sentences, lang="en", model_type="facebook/bart-large-mnli")
    #avg_precision = P.mean().item()
    #avg_recall = R.mean().item()
    #avg_f1 = F1.mean().item()
    truncated_original = [truncate_sentence_nltk(s, max_len_allowed) for s in original_text]
    truncated_predicted = [truncate_sentence_nltk(s, max_len_allowed) for s in candidate_text]
    #lens_score = lens_eval.score(truncated_original, truncated_predicted,batch_size=1)
    flat_scores = []

    '''
    for item in lens_score:
        if isinstance(item, dict):
            # Extract the score value - change 'score' to the actual key name if different
            if 'score' in item:
                flat_scores.append(float(item['score']))
            else:
                # If dict has multiple keys, extract values and flatten
                for val in item.values():
                    try:
                        flat_scores.append(float(val))
                    except Exception:
                        pass
        elif isinstance(item, list):
            for subitem in item:
                if isinstance(subitem, dict):
                    if 'score' in subitem:
                        flat_scores.append(float(subitem['score']))
                    else:
                        for val in subitem.values():
                            try:
                                flat_scores.append(float(val))
                            except Exception:
                                pass
                else:
                    try:
                        flat_scores.append(float(subitem))
                    except Exception:
                        pass
        else:
            try:
                flat_scores.append(float(item))
            except Exception:
                pass
    
'''
    #if len(flat_scores) == 0:
    #    raise ValueError("No numeric scores found in lens_score")

    #avg_score = sum(flat_scores) / len(flat_scores)

    org_ttr, simp_ttr = compare_ttr(original_sentences, simplified_sentences)
    comp_ratio = compression_ratio(original_sentences, simplified_sentences)
    org_word_length = avg_word_len(original_sentences)
    org_sentence_length = avg_sentence_len(original_sentences)
    simp_word_length = avg_word_len(simplified_sentences)
    simp_sentence_length = avg_sentence_len(simplified_sentences)

    avg_similarity = compute_paraphrase_similarity(original_sentences, simplified_sentences)

        # --- SAMSA (SRL-based) ---
    samsa_avg, samsa_split, samsa_struct = compute_samsa(original_sentences, simplified_sentences)

    with open("final_evaluation_score_GPT.txt", "w", encoding="utf-8") as f:
        f.write(f"BLEU Score: {bleu_score:.4f}\n")
        f.write(f"Flesch-Kincaid Simplified Grade Level: {sim_fk_grade_level:.2f}\n")
        f.write(f"Flesch Simplified Reading Ease: {sim_fk_reading_ease:.2f}\n")
        f.write(f"Average SARI Score: {sari_score}\n")
        #f.write(f"LENS Avg Score: {avg_score:.4f}\n")
        f.write(f"Average BERTScore Precision: {avg_precision:.4f}\n")
        f.write(f"Average BERTScore Recall: {avg_recall:.4f}\n")
        f.write(f"Average BERTScore F1: {avg_f1:.4f}\n")
        f.write(f"Original TTR score : {org_ttr:.4f}\n")
        f.write(f"Simplified TTR score : {simp_ttr:.4f}\n")
        f.write(f"Average Compresion Ratio : {comp_ratio:.4f}\n")
        f.write(f"Original Word Length : {org_word_length:.4f}\n")
        f.write(f"Original Sentence Length : {org_sentence_length:.4f}\n")
        f.write(f"Simplified Word Length : {simp_word_length:.4f}\n")
        f.write(f"Simplified Sentence Length : {simp_sentence_length:.4f}\n")
        f.write(f"Average Pharaphrase Similarity : {avg_similarity:.4f}\n")
        f.write(f"SAMSA (dep-based) Average: {samsa_avg:.4f}\n")
        f.write(f"  ├─ Split term (avg): {samsa_split:.4f}\n")
        f.write(f"  └─ Structure term (avg): {samsa_struct:.4f}\n")


def main(args):
    original_file = args.original_file
    simplified_file = args.simplified_file
    reference_file = args.ref_file if args.ref_file else None

    evaluate(original_file, simplified_file, reference_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for evaluation.')
    
    parser.add_argument('--simplified_file', type=str, required=True, help='Path to simplified text file.')
    parser.add_argument('--original_file', type=str, required=True, help='Path to original text file.')
    parser.add_argument('--ref_file', type=str, required=False, help='(Optional) Path to reference text file.')

    args = parser.parse_args()
    main(args)
